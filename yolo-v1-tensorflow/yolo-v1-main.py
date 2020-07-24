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
    templist = []
    filelist = os.listdir(TrainDir)
    print(len(filelist))

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
                    templist.append([xmin, ymin, xmax, ymax, str(label_full.index(label))])
                else:
                    continue
            else:
                continue
        Filenames_Train.append([TrainDir + filelist[f], templist])
    # random.shuffle(Filenames_Train)
    print(Filenames_Train)
    print("Finish Image Loading !")
    print("###############################################")


def batch_train(batchsize):
    """
    > about Filenames_Train[index_train + i][0] <

    [0] : filename(XXX.jpg)
    [1] : annotation - X min
    [2] : annotation - Y min
    [3] : annotation - X max
    [4] : annotation - Y man
    [5] : annotation - label value

    We should normalize(0 ~ 1) [1]~[4] and make [5] one-hot encoding(0 ~ label size)
    """
    global index_train
    x_data = np.zeros([batchsize, image_Width, image_Height, channel], dtype=np.float32)
    y_data = np.zeros((batchsize, grid, grid, 5 + label_size), dtype=np.float32)  # one hot encoding을 위해 0으로 채워진 리스트를 만듭니다
    # y_data = np.zeros((batchsize, 1, 1, label_size), dtype=np.float32)  # one hot encoding을 위해 0으로 채워진 리스트를 만듭니다
    for i in range(0, batchsize):
        # print(f'target file={Filenames_Train[index_train + i][0]}')
        value = cv2.imread(Filenames_Train[index_train + i][0])
        original_h = value.shape[0]     # height = row
        original_w = value.shape[1]     # width = column(=col)
        # width <-> xmin/xmax
        # height <-> ymin/ymax
        print(original_h)
        print(original_w)
        val_label = Filenames_Train[index_train + i][5]
        xmin = int(int(Filenames_Train[index_train + i][1]) / original_w * image_Width)
        ymin = int(int(Filenames_Train[index_train + i][2]) / original_h * image_Height)
        xmax = int(int(Filenames_Train[index_train + i][3]) / original_w * image_Width)
        ymax = int(int(Filenames_Train[index_train + i][4]) / original_h * image_Height)
        print(xmin, ymin, xmax, ymax, val_label)
        norm_xmin = xmin / original_w
        norm_ymin = ymin / original_h
        norm_xmax = xmax / original_w
        norm_ymax = ymax / original_h
        print(norm_xmin, norm_ymin, norm_xmax, norm_ymax)
        xrange = [norm_xmin, norm_xmax]
        yrange = [norm_ymin, norm_ymax]
        size = float(1 / 7)
        print(size)

        # for r in range(0, grid):
        #     for c in range(0, grid):
        #         now = [r * size, c * size]
        #

        one_hot = np.squeeze(np.zeros(shape=(1, label_size)))
        one_hot[int(val_label)] = 1
        print(one_hot)

        value = cv2.resize(value, (image_Height, image_Width))
        x_data[i] = value
        y_data[i][0] = Filenames_Train[index_train + i][1]
        y_data[i][1] = int(int(Filenames_Train[index_train + i][2]) / original_w * image_Width)
        y_data[i][2] = int(int(Filenames_Train[index_train + i][3]) / original_h * image_Height)
        y_data[i][3] = int(int(Filenames_Train[index_train + i][4]) / original_w * image_Width)
        y_data[i][4] = int(int(Filenames_Train[index_train + i][5]) / original_h * image_Height)
        # y_data[i, :, :, Filenames_Train[index_train + i][1]] = 1
    # print(y_data)
    index_train += batchsize
    if index_train + batchsize >= Total_Train:
        index_train = 0
    return x_data, y_data


# def batch_eval(batchsize):
#     global index_eval
#     x_data = np.zeros([batchsize, image_Width, image_Height, channel], dtype=np.uint8)
#     y_data = np.zeros((batchsize, label_size), dtype=np.float32)  # one hot encoding을 위해 0으로 채워진 리스트를 만듭니다
#     # y_data = np.zeros((batchsize, 1, 1, label_size), dtype=np.float32)  # one hot encoding을 위해 0으로 채워진 리스트를 만듭니다
#     for i in range(0, batchsize):
#         rand = random.randrange(0, 2)
#         value = cv2.imread(Filenames_Eval[index_eval + i][0])
#         if rand == 1:
#             value = cv2.flip(value, 1)
#         value = cv2.resize(value, (image_Height, image_Width))
#         x_data[i] = value
#         y_data[i][Filenames_Eval[index_eval + i][1]] = 1
#         # y_data[i, :, :, Filenames_Eval[index_eval + i][1]] = 1
#     index_eval += batchsize
#     if index_eval + batchsize >= Total_Eval:
#         index_eval = 0
#     return x_data, y_data


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

X = tf.compat.v1.placeholder(tf.uint8, [batchsize, image_Width, image_Height, channel])
X = tf.math.divide(tf.cast(X, tf.float32), 255.0, name='input_node')
Y = tf.compat.v1.placeholder(tf.float32, [batchsize, grid, grid, 5 + label_size], name='Y')
print("Y:", Y)
# Y = tf.compat.v1.placeholder(tf.float32, [batchsize, 1, 1, label_size], name='Y')
istraining = tf.compat.v1.placeholder(tf.bool, name='istraining')

result = network(X, istraining)
print(f" *** result={result.fc2}")

# for b in range(0, batchsize):
# res = sess.run([result.fc3])
# print(res)
# print(result.fc3.shape)
# # result = np.array(result.fc3)
# # Y = np.array(Y)
# print(result.fc3)
# print(Y)

loss = loss_layer(predicts=result.fc2, labels=Y)
print(f" *** loss = {loss}")
cross_entropy = tf.reduce_mean(loss)
# cross_entropy = tf.reduce_mean(loss)
train_step = tf.compat.v1.train.AdamOptimizer(Learning_Rate * batchsize).minimize(cross_entropy)
#
# print(" *** Softmax(X):", tf.nn.softmax(result.fc3))
# print(" *** Softmax(Y):", tf.nn.softmax(Y))
#
# ay = tf.argmax(tf.nn.softmax(result.fc3), 1)
# ly = tf.argmax(tf.nn.softmax(Y), 1)
#
# print(" *** Argmax(X):", ay)
# print(" *** Argmax(Y):", ly)

# correct_prediction = tf.equal(tf.nn.softmax(result.fc3), Y)
# correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(result.fc3), 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.compat.v1.summary.scalar("loss", cross_entropy)
# tf.compat.v1.summary.scalar("accuracy", accuracy)

merge_summary_op = tf.compat.v1.summary.merge_all()
total_time = 0
accuracy_sum = 0

merged = tf.compat.v1.summary.merge_all()

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    accuracy_list = []

    saver = tf.compat.v1.train.Saver()  # Network model Save
    # save_path = saver.restore(sess, "D:\\Saver\\3lab_fingertip_vgg19_2\\3lab_fingertip_vgg19_2_Epoch_10.ckpt")

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
            # print(f'bx={bx}')
            # print(f'by={by}')
            bx = np.reshape(bx, [batchsize, image_Width, image_Height, channel])
            ts, cost, res = sess.run([train_step, cross_entropy, result.fc2], feed_dict={X: bx, Y: by, istraining.name: True})

            print('[' + str(count) + '] ',
                  'Epoch %d    ' % (epoch + 1),
                  # 'Training accuracy %g     ' % train_accuracy,
                  'loss %g        ' % cost,
                  'result=', res)

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
