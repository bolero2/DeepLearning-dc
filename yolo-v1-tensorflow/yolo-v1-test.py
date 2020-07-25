import cv2
import os
import random
import numpy as np
import tensorflow as tf
import time
from yolo1_darknet import network, loss_layer
import xml.etree.ElementTree as xml


TrainDir = "C:\\dataset\\VOC2012\\JPEGImages\\"  # 300,000 images
TrainDir_Annot = "C:\\dataset\\VOC2012\\Annotations\\"

# The names of this variables(=ModelDir, ModelName) must come from the script name.
ModelName = "4lab_detection1"
ModelDir = "..\\1+Saver\\" + ModelName + "\\"

Filenames_Eval = []
Filenames_Train = []

index_train = 0
index_eval = 0

ForEpoch = 40

label_full = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat',
              'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

label_size = 20     # => len(label_full)
Total_Train = 17125
# Total_Eval = 4000

batchsize = 1
image_Width = 448
image_Height = 448
channel = 3
Learning_Rate = 0.00001

grid = 7


############
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    test = cv2.imread("airplane.jpg")
    test = cv2.resize(test, (image_Height, image_Width)) / 255.0
    x = np.zeros([1, image_Height, image_Width, 3])
    x[0, :, :, :] = test
    cv2.imshow("Input image", x[0, :, :, :])
    cv2.waitKey(0)

    X = tf.compat.v1.placeholder(tf.uint8, [batchsize, image_Width, image_Height, channel])
    X = tf.math.divide(tf.cast(X, tf.float32), 255.0, name='input_node')
    istraining = tf.compat.v1.placeholder(tf.bool, name='istraining')

    result = network(X, istraining, "D:\\0+2020ML\\1+Saver\\4lab_detection1\\4lab_detection1_Epoch_3.ckpt", sess)



    output = sess.run([result.fc2], feed_dict={network.imgs: x, network.training: False})
    print(output)
