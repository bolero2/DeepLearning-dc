"""
Title : shufflenet_1.py
Date : 2019-01-15
Author : Noh Dae Cheol

Network : ShuffleNet(2D)
Dataset : 2D Hand Gesture
Dropout : Don't use.
"""

import cv2
import os
import random
import numpy as np
import tensorflow as tf
import time

TrainDir = "D:\\2D_DATA\\RGB\\"  # 120,000 images
# TrainDir = "D:\\2D_DATA\\RGB\\"  # 120,000 images
EvalDir = "D:\\2D_DATA\\BRIGHT\\"  # 12,000 images

# The names of this variables(=ModelDir, ModelName) must come from the script name.
ModelName = "ShuffleNet_v1"
ModelDir = "D:\\SavedModel\\" + ModelName + "\\"


Filenames_Train = []
Filenames_Eval = []

index_train = 0
index_eval = 0

BatchSize = 64

Total_Train = 120000
Total_Eval = 12000
# Total_List = np.zeros([16], dtype=int)

# Declaring Image Width and Image Height.
image_Width = 224
image_Height = 224

label_size = 6
group_num = 4
output_dim = 0

channel = 3
exp = 6  # expansion ratio

# Count of Epoch(Epoch 1 ~ Epoch ?)
ForEpoch = 20
dropout_rate = 0.5


def load_image():
    global Filenames_Train
    global Filenames_Eval
    print("###############################################")
    print("Start Image Loading ...")
    print("###############################################")
    templist = []
    for i in range(0, label_size):
        filelist = os.listdir(TrainDir + str(i))
        for j in range(0, len(filelist)):
            templist.append([TrainDir + str(i) + '/' + filelist[j], i])
    Filenames_Train = templist
    random.shuffle(Filenames_Train)
    templist = []

    for i in range(0, label_size):
        filelist = os.listdir(EvalDir + str(i))
        for j in range(0, len(filelist)):
            templist.append([EvalDir + str(i) + '/' + filelist[j], i])
    Filenames_Eval = templist
    random.shuffle(Filenames_Eval)
    print("Finish Image Loading !")
    print("###############################################")


def batch_train(batchsize):
    global index_train
    x_data = np.zeros([batchsize, image_Width, image_Height, channel], dtype=np.float32)
    y_data = np.zeros((batchsize, label_size), dtype=np.float32)  # one hot encoding을 위해 0으로 채워진 리스트를 만듭니다
    for i in range(0, batchsize):
        value = cv2.imread(Filenames_Train[index_train + i][0])
        value = value/255
        value = cv2.resize(value, (image_Height, image_Width))
        x_data[i] = value
        y_data[i][Filenames_Train[index_train + i][1]] = 1
    index_train += batchsize
    if index_train + batchsize >= Total_Train:
        index_train = 0
    return x_data, y_data


def batch_eval(batchsize):
    global index_eval
    x_data = np.zeros([batchsize, image_Width, image_Height, channel], dtype=np.float32)
    y_data = np.zeros((batchsize, label_size), dtype=np.float32)  # one hot encoding을 위해 0으로 채워진 리스트를 만듭니다
    for i in range(0, batchsize):
        value = cv2.imread(Filenames_Eval[index_eval + i][0])
        value = value / 255
        value = cv2.resize(value, (image_Height, image_Width))
        x_data[i] = value
        y_data[i][Filenames_Eval[index_eval + i][1]] = 1
    index_eval += batchsize
    if index_eval + batchsize >= Total_Eval:
        index_eval = 0
    return x_data, y_data


def batch_norm(input, n_out, training, scope='bn'):
    with tf.variable_scope(scope):
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


# ================================================================================================================
def conv2d(input, weight, strides=1, padding='SAME', name=None):
    return tf.nn.conv2d(input, weight, strides=[1, strides, strides, 1], padding=padding, name=name)


def conv2d_1x1(input, output_dim, strides=1, padding='SAME', name=None):
    input_tensor_channel = input.get_shape().as_list()[-1]
    w = tf.Variable(tf.truncated_normal(shape=[1, 1, input_tensor_channel, output_dim], stddev=0.1))
    return conv2d(input, w, strides, padding, name)


def return_tensor_channel(input):
    return input.get_shape().as_list()[-1]


def pw_conv(input, output_dim, strides=1, padding='SAME', name=None):
    return conv2d_1x1(input, output_dim, strides, padding, name)


def dw_conv(input, depth_filter_channel=1, strides=1, padding='SAME', name=None):
    input_tensor_channel = input.get_shape().as_list()[-1]
    w = tf.Variable(tf.truncated_normal(shape=[3, 3, input_tensor_channel, depth_filter_channel], stddev=0.1))
    return tf.nn.depthwise_conv2d(input, w, strides=[1, strides, strides, 1], padding=padding, name=name)


# depthwise separable convolution
def separable_conv(input, output_dim, depth_filter_channel=1, strides=1, padding='SAME', name=None):
    input_tensor_channel = input.get_shape().as_list()[-1]
    dw_kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, input_tensor_channel, depth_filter_channel], stddev=0.1))
    pw_kernel = tf.Variable(tf.truncated_normal(shape=[1, 1, input_tensor_channel * depth_filter_channel, output_dim],
                                                stddev=0.1))
    return tf.nn.separable_conv2d(input, dw_kernel, pw_kernel,
                                  strides=[1, strides, strides, 1], padding=padding, name=name)


def bottleneck_conv(input, expansion_ratio, output_dim, istraining, strides=1, padding='SAME', name=None):
    input_tensor_channel = input.get_shape().as_list()[-1]
    bottleneck_dim = round(expansion_ratio * input_tensor_channel)

    # step 1. Pointwise Conv
    pw1 = tf.nn.relu(batch_norm(pw_conv(input, bottleneck_dim, strides=1, padding=padding, name=name),
                                n_out=bottleneck_dim, training=istraining))
    # step 2. Depthwise Conv
    dw1 = dw_conv(pw1, strides=strides, padding=padding, name=name)
    dw2 = batch_norm(dw1, n_out=dw1.get_shape().as_list()[-1], training=istraining)
    dw3 = tf.nn.relu(dw2)

    # step 3. Pointwise Conv & Linear(is called "pwl" in function)
    pwl1 = pw_conv(dw3, output_dim, strides=1, padding=padding, name=name)
    pwl2 = batch_norm(pwl1, n_out=output_dim, training=istraining)

    output = pwl2

    return output


def grouped_conv2d(input, output_dim=None, group_num=1, ksize=1, strides=1, name=None):
    input_tensor_channel = input.get_shape().as_list()[-1]
    input_group = tf.split(input, group_num, axis=-1)   # Divide the input image channel into [group_num]EA groups.
    # if output_dim is None(input nothing) -> follow input tensor's channel.
    # do you want to channel? -> input value which is what you want in var[output_dim] !!!
    if output_dim is None:
        w = tf.Variable(tf.truncated_normal(shape=[ksize, ksize, return_tensor_channel(input_group[0]),
                                                   int(input_tensor_channel / group_num)], stddev=0.1))
    else:
        w = tf.Variable(tf.truncated_normal(shape=[ksize, ksize, return_tensor_channel(input_group[0]), int(output_dim / group_num)],
                                            stddev=0.1))
    output_group = [tf.nn.conv2d(input_group[num], w, strides=[1, strides, strides, 1], padding='SAME', name=name)
                    for num in range(0, group_num)]
    output = tf.concat(output_group, axis=-1)           # Channel Augmentation(before channel * group_num)

    return output


def channel_shuffle(input, group_num):
    n, h, w, c = input.shape.as_list()
    x_reshaped = tf.reshape(input, [-1, h, w, group_num, c // group_num])
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(x_transposed, [-1, h, w, c])

    return output


def shuffle_unit(input, first_output_dim=None, second_output_dim=None, group_num=1, strides=1, istraining=None, name=None):

    if strides == 1:
        su1 = grouped_conv2d(input=input, output_dim=first_output_dim, group_num=group_num, ksize=1, strides=1)
        su1_1 = tf.nn.relu(batch_norm(su1, n_out=return_tensor_channel(su1), training=istraining))
        # print("su1_1:", su1_1)
        su2 = channel_shuffle(su1_1, group_num=group_num)
        # print("su2:", su2)
        su3 = dw_conv(input=su2, depth_filter_channel=1, strides=1, padding='SAME')
        su3_1 = batch_norm(su3, n_out=return_tensor_channel(su3), training=istraining)
        su4 = grouped_conv2d(input=su3_1, output_dim=second_output_dim, group_num=group_num, ksize=1, strides=1)
        su4_1 = batch_norm(su4, n_out=return_tensor_channel(su4), training=istraining)
        # print('su4_1 and input:', su4_1, input)
        su5 = tf.nn.relu(tf.add(input, su4_1), name=name)

        return su5

    elif strides == 2:
        su1 = grouped_conv2d(input=input, output_dim=first_output_dim, group_num=group_num, ksize=1, strides=1)
        su1_1 = tf.nn.relu(batch_norm(su1, n_out=return_tensor_channel(su1), training=istraining))
        su2 = channel_shuffle(su1_1, group_num=group_num)
        su3 = dw_conv(input=su2, depth_filter_channel=1, strides=2, padding='SAME')
        su3_1 = batch_norm(su3, n_out=return_tensor_channel(su3), training=istraining)
        su4 = grouped_conv2d(input=su3_1, output_dim=second_output_dim, group_num=group_num, ksize=1, strides=1)
        # print("su4:", su4)
        su4_1 = batch_norm(su4, n_out=return_tensor_channel(su4), training=istraining)
        unit_avg_pool = tf.nn.avg_pool(input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name='shuffle_unit_avg_pooling')
        # print("unit_avg_pool:", unit_avg_pool)
        su5 = tf.nn.relu(tf.concat((su4_1, unit_avg_pool), axis=-1), name=name)

        return su5


def ShuffleNet(input, weight, b_training, group_num):

    # ================================================================================
    #                           N   E   T   W   O   R   K                            #
    # ================================================================================

    C1 = tf.nn.relu(batch_norm(conv2d(input, weight['wc1'], strides=2, padding='SAME'), n_out=24, training=b_training),
                    name='C1')
    print("C1: ", C1)   # 112 * 112

    L1 = tf.nn.max_pool(C1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    print("L1: ", L1)   # 56 * 56

    # ==============================
    #      S  T  A  G  E     2     #
    # ==============================
    C2_1 = shuffle_unit(L1, first_output_dim=62, second_output_dim=248, group_num=group_num, strides=2, istraining=b_training, name='shuffle1-1')
    print("C2_1:", C2_1)
    C2_2 = shuffle_unit(C2_1, first_output_dim=68, second_output_dim=272, group_num=group_num, strides=1, istraining=b_training, name='shuffle1-2')
    print("C2_2:", C2_2)
    C2_3 = shuffle_unit(C2_2, first_output_dim=68, second_output_dim=272, group_num=group_num, strides=1, istraining=b_training, name='shuffle1-3')
    print("C2_3:", C2_3)
    C2_4 = shuffle_unit(C2_3, first_output_dim=68, second_output_dim=272, group_num=group_num, strides=1, istraining=b_training, name='shuffle1-4')
    print("C2_4:", C2_4)    # 28 * 28 * 272

    # ==============================
    #      S  T  A  G  E     3     #
    # ==============================
    C3_1 = shuffle_unit(C2_4, first_output_dim=68, second_output_dim=272, group_num=group_num, strides=2, istraining=b_training, name='shuffle2-1')
    print("C3_1:", C3_1)
    C3_2 = shuffle_unit(C3_1, first_output_dim=136, second_output_dim=544, group_num=group_num, strides=1, istraining=b_training, name='shuffle2-2')
    print("C3_2:", C3_2)
    C3_3 = shuffle_unit(C3_2, first_output_dim=136, second_output_dim=544, group_num=group_num, strides=1, istraining=b_training, name='shuffle2-3')
    print("C3_3:", C3_3)
    C3_4 = shuffle_unit(C3_3, first_output_dim=136, second_output_dim=544, group_num=group_num, strides=1, istraining=b_training, name='shuffle2-4')
    print("C3_4:", C3_4)
    C3_5 = shuffle_unit(C3_4, first_output_dim=136, second_output_dim=544, group_num=group_num, strides=1, istraining=b_training, name='shuffle2-5')
    print("C3_5:", C3_5)
    C3_6 = shuffle_unit(C3_5, first_output_dim=136, second_output_dim=544, group_num=group_num, strides=1, istraining=b_training, name='shuffle2-6')
    print("C3_6:", C3_6)
    C3_7 = shuffle_unit(C3_6, first_output_dim=136, second_output_dim=544, group_num=group_num, strides=1, istraining=b_training, name='shuffle2-7')
    print("C3_7:", C3_7)
    C3_8 = shuffle_unit(C3_7, first_output_dim=136, second_output_dim=544, group_num=group_num, strides=1, istraining=b_training, name='shuffle2-8')
    print("C3_8:", C3_8)

    # ==============================
    #      S  T  A  G  E     4     #
    # ==============================
    C4_1 = shuffle_unit(C3_8, first_output_dim=136, second_output_dim=544, group_num=group_num, strides=2, istraining=b_training, name='shuffle3-1')
    print("C4_1:", C4_1)
    C4_2 = shuffle_unit(C4_1, first_output_dim=272, second_output_dim=1088, group_num=group_num, strides=1, istraining=b_training, name='shuffle3-2')
    print("C4_2:", C4_2)
    C4_3 = shuffle_unit(C4_2, first_output_dim=272, second_output_dim=1088, group_num=group_num, strides=1, istraining=b_training, name='shuffle3-3')
    print("C4_3:", C4_3)
    C4_4 = shuffle_unit(C4_3, first_output_dim=272, second_output_dim=1088, group_num=group_num, strides=1, istraining=b_training, name='shuffle3-4')
    print("C4_4:", C4_4)

    L2 = tf.nn.avg_pool(value=C4_4, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID', name='global_average_pooling1')
    print("GAP:", L2)

    FC1 = tf.reshape(L2, shape=[-1, L2.shape[1] * L2.shape[2] * L2.shape[3]])
    print("FC1:", FC1)

    FC2 = tf.matmul(FC1, weight['wfc1'])
    print("FC2:", FC2)

    return FC2   # 56 * 56


weights = {
    # 'wc1' : tf.Variable(tf.truncated_normal([3, 7, 7, channel, 64], stddev=0.1))

    'wc1': tf.Variable(tf.truncated_normal([3, 3, channel, 24], stddev=0.1)),
    'wfc1': tf.Variable(tf.truncated_normal([1 * 1 * 1088, label_size], stddev=0.1))

    # 3D : [frameSize, width, height, input_channel, output_channel]
    # 2D : [width, height, input_channel, output_channel]
}

if __name__ == '__main__':
    load_image()

    X = tf.placeholder(tf.float32, [BatchSize, image_Width, image_Height, channel], name='X')
    Y = tf.placeholder(tf.float32, [BatchSize, label_size], name='Y')
    istraining = tf.placeholder(tf.bool, name='istraining')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    result = ShuffleNet(input=X,
                        weight=weights,
                        b_training=istraining,
                        group_num=group_num)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result,
                                                                           labels=Y))
    print("* Cross Entropy SIZE : " + str(cross_entropy))

    Result_argmax = tf.argmax(tf.nn.softmax(result), 1)
    Label_argmax = tf.argmax(Y, 1)
    print("* Result Argmax : ", Result_argmax)
    print("* Label Argmax : ", Label_argmax)

    ay = tf.argmax(tf.nn.softmax(result), 1)
    ly = tf.argmax(tf.nn.softmax(Y), 1)
    correct_prediction = tf.equal(Result_argmax, Label_argmax)
    print("* tf.argmax : " + str(Result_argmax))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_step = tf.train.AdamOptimizer(0.00001 * BatchSize).minimize(cross_entropy)
    accuracy_list = []
    accuracy_sum = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_time = 0
        accuracy_list = []

        # ==========================================================================================================
        # Training!
        # ==========================================================================================================
        for epoch in range(0, ForEpoch):  # epoch 1 ~ epoch 20
            for i in range(int(Total_Train / BatchSize)):
                bx, by = batch_train(BatchSize)
                bx = np.reshape(bx, [BatchSize, image_Width, image_Height, channel])
                ts, cost, train_accuracy, ra, la = sess.run([train_step, cross_entropy, accuracy, Result_argmax,
                                                             Label_argmax], feed_dict={X: bx,
                                                                                       Y: by,
                                                                                       istraining.name: True})
                # for r in range(0, 5):
                #     print("Result Weight [" + str(r * 3 + 0) + "] " + str(RS[0][r * 3 + 0]) + "     " +
                #           "Result Weight [" + str(r * 3 + 1) + "] " + str(RS[0][r * 3 + 1]) + "     " +
                #           "Result Weight [" + str(r * 3 + 2) + "] " + str(RS[0][r * 3 + 2]) + "     ")

                print('Epoch %d    ' % (epoch + 1),
                      'Training accuracy %g     ' % train_accuracy,
                      'loss %g        ' % cost)

            for j in range(int(Total_Eval / BatchSize)):
                Start = time.time()  # For Time Checking!
                ex, ey = batch_eval(BatchSize)
                ex = np.reshape(ex, [BatchSize, image_Width, image_Height, channel])
                l, y, acc = sess.run([ly, ay, accuracy], feed_dict={X: ex,
                                                                    Y: ey,
                                                                    istraining.name: False})
                End = time.time() - Start
                print("epoch ", (epoch + 1), "     mini accuracy : ", acc, "     mini time(sec) : ", End)
                total_time = total_time + End
                accuracy_sum = accuracy_sum + acc

            print('Epoch %d' % (epoch + 1), 'test : %f' % (accuracy_sum / (Total_Eval / BatchSize) * 100))
            accuracy_list.append(accuracy_sum / (Total_Eval / BatchSize) * 100)
            accuracy_sum = 0
            index_eval = 0

        for i in range(0, ForEpoch):
            print('Epoch %d' % (i + 1), 'test : %f' % accuracy_list[i])

            # if epoch < 5:
            #     save_path = saver1.save(sess, ModelDir + "EPOCH_" + str(
            #         epoch + 1) + "\\" + ModelName + "_Epoch_" + str(epoch + 1) + ".ckpt")
            # elif epoch < 10:
            #     save_path = saver2.save(sess, ModelDir + "EPOCH_" + str(
            #         epoch + 1) + "\\" + ModelName + "_Epoch_" + str(epoch + 1) + ".ckpt")
            # elif epoch < 15:
            #     save_path = saver3.save(sess, ModelDir + "EPOCH_" + str(
            #         epoch + 1) + "\\" + ModelName + "_Epoch_" + str(epoch + 1) + ".ckpt")
            # elif epoch < 20:
            #     save_path = saver4.save(sess, ModelDir + "EPOCH_" + str(
            #         epoch + 1) + "\\" + ModelName + "_Epoch_" + str(epoch + 1) + ".ckpt")
            #
            # for t in range(int(Total_Test / BatchSize)):
            #     ex, ey, ex2 = Batch_Eval(BatchSize)
            #     ex = np.reshape(ex, [BatchSize, FrameSize, image_Width, image_Height, channel])
            #     ex2 = np.reshape(ex2, [BatchSize, FrameSize, image_Width, image_Height, 1])
            #     acc = sess.run(accuracy, feed_dict={X: ex, X2: ex2, Y: ey, istraining.name: False, dropout_rate: 1.0})
            #     print("epoch ", (epoch + 1), "          mini accuracy : ", acc)
            #     accuracy_sum = accuracy_sum + acc
            # print('Epoch %d' % (epoch + 1), 'test : %f' % (accuracy_sum / (Total_Test / BatchSize) * 100))
            #
            # accuracy_list.append(accuracy_sum / (Total_Test / BatchSize) * 100)
            # accuracy_sum = 0
            # index_eval = 0

        print("### Finish Training! ###")




