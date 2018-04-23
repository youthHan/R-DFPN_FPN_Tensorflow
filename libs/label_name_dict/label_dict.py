# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from libs.configs import cfgs

if cfgs.DATASET_NAME == 'ship':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'ship': 1
    }
elif cfgs.DATASET_NAME == 'aeroplane':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1
    }
elif cfgs.DATASET_NAME == 'pascal':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
elif cfgs.DATASET_NAME == 'dota':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'plane': 1,
        'ship': 2,
        'storage-tank': 3,
        'baseball-diamond': 4,
        'tennis-court': 5,
        'basketball-court': 6,
        'ground-track-field': 7,
        'harbor': 8,
        'bridge': 9,
        'large-vehicle': 10,
        'small-vehicle': 11,
        'helicopter': 12,
        'roundabout': 13,
        'soccer-ball-field': 14,
        'swimming-pool': 15
    }
elif cfgs.DATASET_NAME == 'vedia':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        '1': 1,
        '2': 2,
        '4': 3,
        '5': 4,
        '7': 5,
        '8': 6,
        '9': 7,
        '10': 8,
        '11': 9,
        '23': 10,
        '31': 11
    }
else:
    assert 'please set label dict!'


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEl_NAME_MAP = get_label_name_map()