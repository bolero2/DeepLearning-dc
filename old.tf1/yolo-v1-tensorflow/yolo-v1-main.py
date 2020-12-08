import cv2
import os
import random
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as xml
from matplotlib import pyplot as plt
import time

from yolo1_darknet import network, loss_layer
import config as cfg

weight_file = cfg.weight_file

TrainDir = cfg.TrainDir
TrainDir_Annot = cfg.TrainDir_Annot

# The names of this variables(=ModelDir, ModelName) must come from the script name.
ModelName = cfg.ModelName
ModelDir = cfg.ModelDir

Filenames_Eval = cfg.Filenames_Eval
Filenames_Train = cfg.Filenames_Train

index_train = cfg.index_train
index_eval = cfg.index_eval

ForEpoch = cfg.ForEpoch

label_full = cfg.label_full

label_size = cfg.label_size
Total_Train = cfg.Total_Train
# Total_Eval = 4000

batchsize = cfg.batchsize
image_Width = cfg.image_Width
image_Height = cfg.image_Height
channel = cfg.channel
Learning_Rate = cfg.Learning_Rate

grid = cfg.grid


def load_image():
    global Filenames_Train
    global Filenames_Eval
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    print("###############################################")
    print("Start Image Loading ...")
    print("###############################################")
    templist = list()
    filelist = os.listdir(TrainDir)
    print(f'Total number of Images : {len(filelist)}')

    for f in range(0, len(filelist)):
        tree = xml.ElementTree(file=TrainDir_Annot + filelist[f][:-4] + '.xml')
        for elem in tree.iter(tag='object'):
            label = list(elem)[0].text
            for tag in range(0, len(list(elem))):
                if list(elem)[tag].tag == 'bndbox':
                    for count in range(0, len(list(list(elem)[tag]))):
                        if list(list(elem)[tag])[count].tag == 'xmin':
                            xmin = int(float(list(list(elem)[tag])[count].text))
                        elif list(list(elem)[tag])[count].tag == 'ymin':
                            ymin = int(float(list(list(elem)[tag])[count].text))
                        elif list(list(elem)[tag])[count].tag == 'xmax':
                            xmax = int(float(list(list(elem)[tag])[count].text))
                        elif list(list(elem)[tag])[count].tag == 'ymax':
                            ymax = int(float(list(list(elem)[tag])[count].text))
                    templist.append([xmin, ymin, xmax, ymax, label_full.index(label)])
                else:
                    continue
            else:
                continue
        Filenames_Train.append([TrainDir + filelist[f], templist])
        templist = list()
    random.shuffle(Filenames_Train)
    print(Filenames_Train)
    print("Finish Image Loading !")
    print("###############################################")


def batch_train(batchsize):
    global index_train
    x_data = np.zeros([batchsize, image_Width, image_Height, channel], dtype=np.float32)
    y_data = np.zeros((batchsize, grid, grid, 5 + label_size), dtype=np.float32)  # -> one hot encoding
    for i in range(0, batchsize):
        value = cv2.imread(Filenames_Train[index_train + i][0])

        original_h = value.shape[0]     # height = row
        original_w = value.shape[1]     # width = column(=col)

        value = cv2.resize(value, (image_Height, image_Width))
        value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB).astype(np.float32)
        value = (value / 255.0) * 2.0 - 1.0

        size_x = float(image_Width / grid)
        size_y = float(image_Height / grid)

        """
        You should put label value like this form : [1, x_center, y_center, width, height] <- object center grid
        """
        for count in Filenames_Train[index_train + i][1]:
            xmin = count[0]
            ymin = count[1]
            xmax = count[2]
            ymax = count[3]
            label = count[4]

            # Update coordinate (original -> [448, 448])

            center_x = (xmin + xmax) / 2.0
            center_y = (ymin + ymax) / 2.0
            box_width = max(xmin, xmax) - min(xmin, xmax)
            box_height = max(ymin, ymax) - min(ymin, ymax)

            new_center_x = center_x * image_Width / original_w
            new_center_y = center_y * image_Height / original_h
            new_box_width = box_width * image_Width / original_w
            new_box_height = box_height * image_Height / original_h

            offset_x = new_center_x / size_x
            offset_y = new_center_y / size_y
            label_value = np.zeros(shape=(5 + label_size))
            label_value[0] = 1
            label_value[1] = new_center_x
            label_value[2] = new_center_y
            label_value[3] = new_box_width
            label_value[4] = new_box_height
            label_value[5 + label] = 1
            y_data[i, int(offset_x), int(offset_y), :] = label_value
        x_data[i, :, :, :] = value
    index_train = index_train + batchsize
    if index_train + batchsize >= Total_Train:
        index_train = 0

    return x_data, y_data


############

load_image()

with tf.compat.v1.Session() as sess:
    X = tf.compat.v1.placeholder(tf.float32, [batchsize, image_Width, image_Height, channel], name='input')
    Y = tf.compat.v1.placeholder(tf.float32, [batchsize, grid, grid, 5 + label_size], name='label')
    istraining = tf.compat.v1.placeholder(tf.bool, name='istraining')

    result = network(X, istraining)
    print(f" *** result={result.fc2}")

    loss_layer(predicts=result.fc2, labels=Y)
    loss = tf.compat.v1.losses.get_total_loss()
    print(f" *** loss = {loss}")
    optimizer = tf.train.AdamOptimizer(learning_rate=Learning_Rate)
    global_step = tf.train.create_global_step()

    train_step = tf.contrib.training.create_train_op(loss, optimizer, global_step=global_step)

    tf.compat.v1.summary.scalar("loss", loss)

    merge_summary_op = tf.compat.v1.summary.merge_all()
    merged = tf.compat.v1.summary.merge_all()

    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()  # Network model Save
    writer = tf.compat.v1.summary.FileWriter(ModelDir + "logs", sess.graph)

    if weight_file is not None:
        print(f"Weight file Loading Start! -> {weight_file}")
        meta_saver = tf.train.import_meta_graph(weight_file + ".meta")
        save_path = saver.restore(sess, weight_file)
        print(f"Weight file Loading is successful")

    # ==========================================================================================================
    # Training!
    # ==========================================================================================================
    for epoch in range(0, ForEpoch):  # epoch 1 ~ epoch 20
        count = 0
        for i in range(int(Total_Train / batchsize)):
        # for i in range(1):
            count += 1
            bx, by = batch_train(batchsize)
            bx = np.reshape(bx, [batchsize, image_Width, image_Height, channel])
            cost, _ = sess.run([loss, train_step], feed_dict={X: bx, Y: by, istraining.name: True})

            now = time.localtime()
            print("%04d/%02d/%02d %02d:%02d:%02d" % (
                now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

            print('[' + str(count) + '] ',
                  'Epoch %d    ' % (epoch + 1),
                  # 'Training accuracy %g     ' % train_accuracy,
                  'loss %g        ' % cost)

        save_path2 = saver.save(sess, ModelDir + "\\" + ModelName + "_Epoch_" + str(epoch + 1) + ".ckpt")
        tf.io.write_graph(sess.graph_def, ModelDir, "trained_" + ModelName + ".pb", as_text=False)

        # count = 0
        #
        # for j in range(int(Total_Eval / batchsize)):
        #     count += 1
        #     Start = time.time()  # For Time Checking!
        #     ex, ey = batch_eval(batchsize)
        #     ex = np.reshape(ex, [batchsize, image_Width, image_Height, channel])
        #     l, y, acc, summary = sess.run([ly, ay, accuracy, merged], feed_dict={X: ex, Y: ey, istraining.name: False})
        #     writer.add_summary(summary)
        #
        #     End = time.time() - Start
        #     print('[' + str(count) + '] ', "epoch ", (epoch + 1), "     mini accuracy : ", acc, "     mini time(sec) : ", End)
        #     total_time = total_time + End
        #     accuracy_sum = accuracy_sum + acc
        #
        # print('Epoch %d' % (epoch + 1), 'test : %f' % (accuracy_sum / (Total_Eval / batchsize) * 100))
        # accuracy_list.append(accuracy_sum / (Total_Eval / batchsize) * 100)
        # accuracy_sum = 0
        # for i in range(0, epoch):
        #     print('Epoch %d' % (i + 1), 'test : %f' % accuracy_list[i])

    print("### Finish Training! ###")
