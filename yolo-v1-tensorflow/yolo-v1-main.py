import cv2
import os
import random
import numpy as np
import tensorflow as tf
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

batchsize = 4
image_Width = 448
image_Height = 448
channel = 3
Learning_Rate = 0.00001

grid = 7


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
        size_x = float(original_w / grid)
        size_y = float(original_h / grid)

        for count in Filenames_Train[index_train + i][1]:
            xmin = count[0]
            ymin = count[1]
            xmax = count[2]
            ymax = count[3]
            label = count[4]

            box_width = abs(xmin - xmax) / 2.0
            box_height = abs(ymin - ymax) / 2.0

            center_x = xmin + box_width
            center_y = ymin + box_height

            offset_x = center_x / size_x
            offset_y = center_y / size_y
            label_value = np.zeros(shape=(5 + label_size))
            label_value[0] = 1
            label_value[1] = center_x
            label_value[2] = center_y
            label_value[3] = box_width
            label_value[4] = box_height
            label_value[5 + label] = 1
            y_data[i, int(offset_x), int(offset_y), :] = label_value

        value = cv2.resize(value, (image_Height, image_Width))
        x_data[i, :, :, :] = value
    index_train += batchsize
    if index_train + batchsize >= Total_Train:
        index_train = 0
    return x_data, y_data


def batch_norm(input, n_out, training, scope='bn'):
    with tf.compat.v1.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training, true_fn=mean_var_with_update,
                            false_fn=lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
    return normed


############

load_image()

with tf.compat.v1.Session() as sess:
    X = tf.compat.v1.placeholder(tf.uint8, [batchsize, image_Width, image_Height, channel])
    X = tf.math.divide(tf.cast(X, tf.float32), 255.0, name='input_node')
    Y = tf.compat.v1.placeholder(tf.float32, [batchsize, grid, grid, 5 + label_size], name='Y')
    # Y = tf.compat.v1.placeholder(tf.float32, [batchsize, 1, 1, label_size], name='Y')
    istraining = tf.compat.v1.placeholder(tf.bool, name='istraining')

    result = network(X, istraining)
    print(f" *** result={result.fc2}")

    loss_layer(predicts=result.fc2, labels=Y)
    loss = tf.losses.get_total_loss()
    print(f" *** loss = {loss}")
    train_step = tf.compat.v1.train.AdamOptimizer(Learning_Rate * batchsize).minimize(loss)

    tf.compat.v1.summary.scalar("loss", loss)

    merge_summary_op = tf.compat.v1.summary.merge_all()
    merged = tf.compat.v1.summary.merge_all()

    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()  # Network model Save
    writer = tf.compat.v1.summary.FileWriter(ModelDir + "logs", sess.graph)

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
            ts, cost = sess.run([train_step, loss], feed_dict={X: bx, Y: by, istraining.name: True})

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

    print('*** Total Accuracy List ***\n')
    for i in range(0, ForEpoch):
        print('Epoch %d' % (i + 1), 'test : %f' % accuracy_list[i])

    print("### Finish Training! ###")
