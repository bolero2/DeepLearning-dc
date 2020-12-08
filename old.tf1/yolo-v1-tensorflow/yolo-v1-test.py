import cv2
import os
import random
import numpy as np
import tensorflow as tf
import time
from yolo1_darknet import network, loss_layer
import xml.etree.ElementTree as xml
import config as cfg


label_full = cfg.label_full
label_size = cfg.label_size

batchsize = 1
image_Width = cfg.image_Width
image_Height = cfg.image_Height
channel = cfg.channel

grid = cfg.grid

threshold = cfg.threshold
iou_threshold = cfg.iou_threshold
box_per_cell = cfg.box_per_cell        # one cell have 2 box
boundary1 = cfg.boundary1  # 7 * 7 * 20
boundary2 = cfg.boundary2  # 7 * 7 * 20 + 7 * 7 *2

############


def iou(box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
        max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
        max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    inter = 0 if tb < 0 or lr < 0 else tb * lr
    return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)


def interpret_output(output):
    probs = np.zeros((grid, grid,
                      box_per_cell, label_size))
    class_probs = np.reshape(output[0:boundary1], (grid, grid, label_size))
    scales = np.reshape(
        output[boundary1:boundary2],
        (grid, grid, box_per_cell))
    boxes = np.reshape(
        output[boundary2:],
        (grid, grid, box_per_cell, 4))
    offset = np.array(
        [np.arange(grid)] * grid * box_per_cell)
    offset = np.transpose(
        np.reshape(
            offset,
            [box_per_cell, grid, grid]),
        (1, 2, 0))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / grid
    boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

    boxes *= image_Width

    for i in range(box_per_cell):
        for j in range(label_size):
            probs[:, :, i, j] = np.multiply(
                class_probs[:, :, j], scales[:, :, i])

    filter_mat_probs = np.array(probs >= threshold, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0],
                           filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(
        filter_mat_probs, axis=3)[
        filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if iou(boxes_filtered[i], boxes_filtered[j]) > iou_threshold:
                probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append(
            [label_full[classes_num_filtered[i]],
             boxes_filtered[i][0],
             boxes_filtered[i][1],
             boxes_filtered[i][2],
             boxes_filtered[i][3],
             probs_filtered[i]])

    return result


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    test = cv2.imread("airplane.jpg")
    img_h, img_w, _ = test.shape
    test = cv2.resize(test, (image_Height, image_Width)) / 255.0
    x = np.zeros([1, image_Height, image_Width, 3])
    x[0, :, :, :] = test

    imgs = tf.compat.v1.placeholder(tf.uint8, [batchsize, image_Width, image_Height, channel])
    imgs = tf.math.divide(tf.cast(imgs, tf.float32), 255.0, name='input_node')
    istraining = tf.compat.v1.placeholder(tf.bool, name='istraining')

    result = network(imgs, istraining, "D:\\0+2020ML\\1+Saver\\4lab_detection1\\4lab_detection1_Epoch_6.ckpt", sess)

    output = sess.run([result.fc2], feed_dict={result.imgs: x, result.training: False})[0]
    print(f'output={output.shape}')     # (1, 1470)
    results = []
    for i in range(output.shape[0]):
        results.append(interpret_output(output[i]))
    result = results[0]
    print(f'result={result}')
    print("result1:", result[0][1])
    print("result2:", result[0][2])
    print("result3:", result[0][3])
    print("result4:", result[0][4])

    for i in range(len(result)):
        result[i][1] *= (1.0 * img_w / image_Width)  # x_center
        result[i][2] *= (1.0 * img_h / image_Width)  # y_center
        result[i][3] *= (1.0 * img_w / image_Width)  # width
        result[i][4] *= (1.0 * img_h / image_Width)  # height

    for i in range(len(result)):
        x = int(result[i][1])
        y = int(result[i][2])
        w = int(result[i][3] / 2)
        h = int(result[i][4] / 2)
        cv2.rectangle(test, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(test, (x - w, y - h - 20),
                      (x + w, y - h), (125, 125, 125), -1)
        cv2.putText(
            test, result[i][0] + ' : %.2f' % result[i][5],
            (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow("result", test)
    cv2.waitKey(0)
