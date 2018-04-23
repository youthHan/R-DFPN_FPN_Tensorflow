# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys

sys.path.append('../')

from libs.configs import cfgs
import numpy as np
import tensorflow as tf
import glob
import cv2
from libs.label_name_dict.label_dict import *
from tools.DOTA_ToolKit import dota_utils as utils
from tools.DOTA_ToolKit.DOTA.DOTA import *
from help_utils.tools import *

tf.app.flags.DEFINE_string('VOC_dir', '/home/ai-i-hanmingfei/datasets/ODAI-ICPR/split_origin/val/', 'Voc dir')
tf.app.flags.DEFINE_string('xml_dir', 'Annotations', 'xml dir')
tf.app.flags.DEFINE_string('txt_dir', 'labelTxt', 'txt dir')
tf.app.flags.DEFINE_string('image_dir', 'images', 'image dir')
tf.app.flags.DEFINE_string('img_format', '.png', 'format of image')
tf.app.flags.DEFINE_string('dataset', 'dota', 'dataset')
FLAGS = tf.app.flags.FLAGS


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

def analysis_scale(label_name=None):

    return


def analysis_aspect_ratio(label_name=None):

    return


if __name__ == '__main__':
    dota_analysis = DOTA('/home/ai-i-hanmingfei/datasets/ODAI-ICPR/split_half/train')
    dota_analysis.loadAnns("plane")

