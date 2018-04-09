# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys

sys.path.append('../../')

import xml.etree.cElementTree as ET
from libs.configs import cfgs
import numpy as np
import tensorflow as tf
import glob
import cv2
from libs.label_name_dict.label_dict import *
from help_utils.tools import *

tf.app.flags.DEFINE_string('VOC_dir', '/home/ai-i-hanmingfei/datasets/ODAI-ICPR/split_origin/val/', 'Voc dir')
tf.app.flags.DEFINE_string('xml_dir', 'Annotations', 'xml dir')
tf.app.flags.DEFINE_string('txt_dir', 'labelTxt', 'txt dir')
tf.app.flags.DEFINE_string('image_dir', 'images', 'image dir')
tf.app.flags.DEFINE_string('save_name', 'val', 'save name')
tf.app.flags.DEFINE_string('save_dir', cfgs.ROOT_PATH + '/data/tfrecord/', 'save name')
tf.app.flags.DEFINE_string('img_format', '.png', 'format of image')
tf.app.flags.DEFINE_string('dataset', 'dota', 'dataset')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_xml_gtbox_and_label(xml_path):

    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = NAME_LABEL_MAP[child_item.text]
                if child_item.tag == 'bndbox':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(int(node.text))  # [x1, y1. x2, y2, x3, y3, x4, y4]
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)  # [x1, y1. x2, y2, x3, y3, x4, y4, label]
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)  # [x1, y1. x2, y2, x3, y3, x4, y4, label]

    return img_height, img_width, gtbox_label

def read_txt_gtbox_and_label(txt_path):

    """
    :param xtxt_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """
    with open(txt_path, 'r') as anno_handle:
        anno_lines = anno_handle.readlines()


    box_list = []
    for anno_line in anno_lines:
        anno_line = anno_line.strip()
        if anno_line == "": # last empty line of the anno file
            continue

        annos = anno_line.split(' ')
        if len(annos) != 10: # not fullfil the format of DOTA
            continue

        if annos[:8] == "2": # the gtbox is spilt and is too small
            continue

        tmp_box = []
        for node in annos[:8]:
            if '.' in node:
                tmp_box.append(int(node[:-2]))
            else:
                tmp_box.append(int(node))  # [x1, y1. x2, y2, x3, y3, x4, y4]
        label = NAME_LABEL_MAP[annos[8]]
        tmp_box.append(label)  # [x1, y1. x2, y2, x3, y3, x4, y4, label]
        box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)  # [x1, y1. x2, y2, x3, y3, x4, y4, label]

    return gtbox_label


def convert_pascal_to_tfrecord():

    xml_path = FLAGS.VOC_dir + FLAGS.xml_dir
    image_path = FLAGS.VOC_dir + FLAGS.image_dir
    save_path = FLAGS.save_dir + FLAGS.dataset + '_' + FLAGS.save_name + '.tfrecord'
    mkdir(FLAGS.save_dir)

    writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)

    for count, xml in enumerate(glob.glob(xml_path + '/*.xml')):
        # to avoid path error in different development platform
        xml = xml.replace('\\', '/')

        img_name = xml.split('/')[-1].split('.')[0] + FLAGS.img_format
        img_path = image_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)

        # img = np.array(Image.open(img_path))
        img = cv2.imread(img_path)

        feature = tf.train.Features(feature={
            # maybe do not need encode() in linux
            # 'img_name': _bytes_feature(img_name.encode()),
            'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(glob.glob(xml_path + '/*.xml')))

    print('\nConversion is complete!')


def convert_dota_to_tfrecord():

    txt_path = FLAGS.VOC_dir + FLAGS.txt_dir
    image_path = FLAGS.VOC_dir + FLAGS.image_dir
    save_path = FLAGS.save_dir + FLAGS.dataset + '_' + FLAGS.save_name + '.tfrecord'
    mkdir(FLAGS.save_dir)

    writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)

    for count, txt in enumerate(glob.glob(txt_path + '/*.txt')):
        # to avoid path error in different development platform
        txt = txt.replace('\\', '/')

        img_name = txt.split('/')[-1].split('.')[0] + FLAGS.img_format
        img_path = image_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        # img = np.array(Image.open(img_path))
        img = cv2.imread(img_path)

        # img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)
        gtbox_label = read_txt_gtbox_and_label(txt)
        img_height, img_width, _ = img.shape

        feature = tf.train.Features(feature={
            # maybe do not need encode() in linux
            'img_name': _bytes_feature(img_name.encode()),
            # 'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(glob.glob(txt_path + '/*.txt')))

    print('\nConversion is complete!')


if __name__ == '__main__':
    # print(NAME_LABEL_MAP)
    # xml_path = '/home/ai-i-hanmingfei/datasets/ODAI-ICPR/split_origin/train/labelTxt/P0000__1__0___0.txt'
    # read_txt_gtbox_and_label(xml_path).shape
    # tf.app.run()
    convert_dota_to_tfrecord()
