import os
import cv2
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt


def _is_box_intersect(box1, box2):
    if abs(box1[0] - box2[0]) < box1[2] + box2[2] and abs(box1[1] - box2[1]) < box1[3] + box2[3]:
        return True
    else:
        return False


def _get_area(box):
    return box[2] * box[3]


def _get_intersection_area(box1, box2):
    # intersection area
    return abs(max(box1[0], box2[0]) -
               min(box1[0] + box1[2], box2[0] + box2[2])) \
           * abs(max(box1[1], box2[1]) -
                 min(box1[1] + box1[3], box2[1] + box2[3]))


def _get_union_area(box1, box2, inter_area=None):
    area_a = _get_area(box1)
    area_b = _get_area(box2)
    if inter_area is None:
        inter_area = _get_intersection_area(box1, box2)

    return float(area_a + area_b - inter_area)


def iou(box1, box2):
    # if boxes dont intersect
    if _is_box_intersect(box1, box2) is False:
        # print("zero")
        return 0
    inter_area = _get_intersection_area(box1, box2)
    print(inter_area)
    union = _get_union_area(box1, box2, inter_area=inter_area)
    # intersection over union
    iou = inter_area / union
    # print(f"iou: {iou}")
    assert iou >= 0

    return iou


a = [0, 179, 441, 483]
b = [0, 182, 438, 306]
print(iou(a, b))