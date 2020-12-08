"""
Title : mobilenetv2_1.py
Date : 2019-02-25
Author : Noh Dae Cheol

Network : MobileNet V2(2D)
Dataset : ImageNet 100 Labels
Dropout : 0.5
"""

import cv2
import os
import random
import numpy as np
import tensorflow as tf
import time

TrainDir = "D:\\Tiny_ImageNet\\Tiny_ImageNet\\Train\\"  # 1,537,631 images
EvalDir = "D:\\Tiny_ImageNet\\Tiny_ImageNet\\Test\\"  # 119,424 images

# The names of this variables(=ModelDir, ModelName) must come from the script name.
ModelName = "MobileNetv2_dropout_190225"
ModelDir = "D:\\SavedModel\\" + ModelName + "\\"


Filenames_Train = []
Filenames_Eval = []

index_train = 0
index_eval = 0

BatchSize = 64

Total_Train = 1537631
Total_Eval = 119424
# Total_List = np.zeros([16], dtype=int)

# Declaring Image Width and Image Height.
image_Width = 112
image_Height = 112

label_size = 100

channel = 3
exp = 6  # expansion ratio

# Count of Epoch(Epoch 1 ~ Epoch ?)
ForEpoch = 30
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


def pw_conv(input, output_dim, strides=1, padding='SAME', name=None):
    return conv2d_1x1(input, output_dim, strides, padding, name)


def dw_conv(input, depth_filter_channel=1, strides=1, padding='SAME', name=None):
    input_tensor_channel = input.get_shape().as_list()[-1]
    s = [1, strides, strides, 1]
    p = padding
    w = tf.Variable(tf.truncated_normal(shape=[3, 3, input_tensor_channel, depth_filter_channel], stddev=0.1))
    return tf.nn.depthwise_conv2d(input, w, strides=s, padding=p, name=name)


def separable_conv(input, output_dim, depth_filter_channel=1, strides=1, padding='SAME', name=None):
    input_tensor_channel = input.get_shape().as_list()[-1]
    dw_kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, input_tensor_channel, depth_filter_channel], stddev=0.1))
    pw_kernel = tf.Variable(tf.truncated_normal(shape=[1, 1, input_tensor_channel * depth_filter_channel, output_dim],
                                                stddev=0.1))
    s = [1, strides, strides, 1]
    p = padding
    return tf.nn.separable_conv2d(input, dw_kernel, pw_kernel, strides=s, padding=p, name=name)


def bottleneck_conv(input, expansion_ratio, output_dim, istraining, strides=1, padding='SAME', name=None):
    input_tensor_channel = input.get_shape().as_list()[-1]
    bottleneck_dim = round(expansion_ratio * input_tensor_channel)

    p = padding

    # step 1. Pointwise Conv
    pw1 = tf.nn.relu(batch_norm(pw_conv(input, bottleneck_dim, strides=1, padding=p, name=name),
                                n_out=bottleneck_dim, training=istraining))
    # step 2. Depthwise Conv
    dw1 = dw_conv(pw1, strides=strides, padding=p, name=name)
    dw2 = batch_norm(dw1, n_out=dw1.get_shape().as_list()[-1], training=istraining)
    dw3 = tf.nn.relu(dw2)

    # step 3. Pointwise Conv & Linear
    pwl1 = pw_conv(dw3, output_dim, strides=1, padding=p, name=name)
    pwl2 = batch_norm(pwl1, n_out=output_dim, training=istraining)

    output = pwl2

    return output


def MobileNetV2(input, weight, b_training, expansion_rate, label_size, keep_prob):
    """
    :param input: Input Image
    :param weight: Weight
    :param b_training: bool variable(istraining, True or False)
    :param expansion_rate: expansion rate
    :param label_size: label size
    :return: Last Fully-Connected Layer
    """

    # ================================================================================
    #                           N   E   T   W   O   R   K                            #
    # ================================================================================

    C1 = tf.nn.relu(batch_norm(conv2d(input, weight['wc1'], strides=2, padding='SAME'), n_out=32,
                               training=b_training), name='C1')
    print(C1)  # 112 * 112

    C2 = bottleneck_conv(C1, 1, 16, istraining=b_training, strides=1, padding='SAME', name='C2')
    print(C2)  # 112 * 112

    C3_1 = bottleneck_conv(C2, expansion_rate, 24, istraining=b_training, strides=2, padding='SAME', name='C3_1')
    C3_2 = bottleneck_conv(C3_1, expansion_rate, 24, istraining=b_training, strides=1, padding='SAME', name='C3_2')
    print(C3_2)  # 56 * 56

    C4_1 = bottleneck_conv(C3_2, expansion_rate, 32, istraining=b_training, strides=2, padding='SAME', name='C4_1')
    C4_2 = bottleneck_conv(C4_1, expansion_rate, 32, istraining=b_training, strides=1, padding='SAME', name='C4_2')
    C4_3 = bottleneck_conv(C4_2, expansion_rate, 32, istraining=b_training, strides=1, padding='SAME', name='C4_3')
    print(C4_3)  # 28 * 28

    C5_1 = bottleneck_conv(C4_3, expansion_rate, 64, istraining=b_training, strides=2, padding='SAME', name='C5_1')
    C5_2 = bottleneck_conv(C5_1, expansion_rate, 64, istraining=b_training, strides=1, padding='SAME', name='C5_2')
    C5_3 = bottleneck_conv(C5_2, expansion_rate, 64, istraining=b_training, strides=1, padding='SAME', name='C5_3')
    C5_4 = bottleneck_conv(C5_3, expansion_rate, 64, istraining=b_training, strides=1, padding='SAME', name='C5_4')
    print(C5_4)  # 14 * 14

    C6_1 = bottleneck_conv(C5_4, expansion_rate, 96, istraining=b_training, strides=1, padding='SAME', name='C6_1')
    C6_2 = bottleneck_conv(C6_1, expansion_rate, 96, istraining=b_training, strides=1, padding='SAME', name='C6_2')
    C6_3 = bottleneck_conv(C6_2, expansion_rate, 96, istraining=b_training, strides=1, padding='SAME', name='C6_3')
    print(C6_3)  # 14 * 14

    C7_1 = bottleneck_conv(C6_3, expansion_rate, 160, istraining=b_training, strides=2, padding='SAME', name='C7_1')
    C7_2 = bottleneck_conv(C7_1, expansion_rate, 160, istraining=b_training, strides=1, padding='SAME', name='C7_2')
    C7_3 = bottleneck_conv(C7_2, expansion_rate, 160, istraining=b_training, strides=1, padding='SAME', name='C7_3')
    print(C7_3)  # 7 * 7

    C8 = bottleneck_conv(C7_3, expansion_rate, 320, istraining=b_training, strides=1, padding='SAME', name='C8')
    print(C8)

    C9 = conv2d_1x1(C8, 1280, strides=1, padding='SAME', name='C9')
    print(C9)

    AVP = tf.nn.avg_pool(C9, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID', name='avg_pool')
    print(AVP)

    C10 = conv2d_1x1(AVP, label_size, strides=1, padding='SAME', name='C10')
    print(C10)

    C11 = tf.nn.dropout(C10, keep_prob=keep_prob)

    FC1 = tf.reshape(C11, shape=[-1, weight['wfc1'].get_shape().as_list()[0]])
    print(FC1)

    FC2 = tf.matmul(FC1, weight['wfc1'])
    print(FC2)

    return FC2

# ================================================================================================================


weights = {
    # 'wc1' : tf.Variable(tf.truncated_normal([3, 7, 7, channel, 64], stddev=0.1))

    'wc1': tf.Variable(tf.truncated_normal([3, 3, channel, 32], stddev=0.1)),
    'wfc1': tf.Variable(tf.truncated_normal([1 * 1 * label_size, label_size], stddev=0.1))

    # 3D : [frameSize, width, height, input_channel, output_channel]
    # 2D : [width, height, input_channel, output_channel]
}

if __name__ == '__main__':
    load_image()

    X = tf.placeholder(tf.float32, [BatchSize, image_Width, image_Height, channel], name='X')
    Y = tf.placeholder(tf.float32, [BatchSize, label_size], name='Y')
    istraining = tf.placeholder(tf.bool, name='istraining')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    result = MobileNetV2(input=X,
                         weight=weights,
                         b_training=istraining,
                         expansion_rate=exp,
                         label_size=label_size,
                         keep_prob=keep_prob)

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
    time_list = []
    time_sum = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()  # Network model Save
    # save_path = saver.restore(sess, ModelDir + ModelName + "_Epoch_7.ckpt")

    # ==========================================================================================================
    # Training!
    # ==========================================================================================================
    for epoch in range(0, ForEpoch):  # epoch 1 ~ epoch 20
        count = 0
        # =============================================
        percentage = epoch / ForEpoch
        dropout_rate = (dropout_rate * percentage + 0.17)
        dropout_rate = round(dropout_rate, 2)
        # =============================================
        for i in range(int(Total_Train / BatchSize)):
            count = count + 1
            bx, by = batch_train(BatchSize)
            bx = np.reshape(bx, [BatchSize, image_Width, image_Height, channel])
            ts, loss, train_accuracy, ra, la = sess.run([train_step, cross_entropy, accuracy, Result_argmax,
                                                         Label_argmax], feed_dict={X: bx,
                                                                                   Y: by,
                                                                                   istraining.name: True,
                                                                                   keep_prob: 0.5})
            # for r in range(0, 5):
            #     print("Result Weight [" + str(r * 3 + 0) + "] " + str(RS[0][r * 3 + 0]) + "     " +
            #           "Result Weight [" + str(r * 3 + 1) + "] " + str(RS[0][r * 3 + 1]) + "     " +
            #           "Result Weight [" + str(r * 3 + 2) + "] " + str(RS[0][r * 3 + 2]) + "     ")

            print('[%d]' % count,
                  'Epoch %d    ' % (epoch + 1),
                  'Training accuracy %g     ' % train_accuracy,
                  'loss %g        ' % loss)

        # save_path = saver.save(sess, ModelDir + "\\" + ModelName + "_Epoch_" + str(epoch + 1) + ".ckpt")
        count = 0
        for j in range(int(Total_Eval / BatchSize)):
            count = count + 1
            Start = time.time()  # For Time Checking!
            ex, ey = batch_eval(BatchSize)
            ex = np.reshape(ex, [BatchSize, image_Width, image_Height, channel])
            loss, l, y, eval_accuracy = sess.run([cross_entropy, ly, ay, accuracy],
                                                 feed_dict={X: ex,
                                                            Y: ey,
                                                            istraining.name: False,
                                                            keep_prob: 1.0})
            End = time.time() - Start
            print("[" + str(count) + "] Epoch", (epoch + 1),
                  "    mini accuracy", eval_accuracy,
                  "       mini time(sec)", End,
                  "     loss %g" % loss)
            time_sum = time_sum + End
            accuracy_sum = accuracy_sum + eval_accuracy

        print('Epoch %d' % (epoch + 1), 'test : %f' % (accuracy_sum / (Total_Eval / BatchSize) * 100))
        print('Time per Image : ', time_sum / Total_Eval)
        accuracy_list.append(accuracy_sum / (Total_Eval / BatchSize) * 100)
        time_list.append(time_sum / Total_Eval)
        accuracy_sum = 0
        time_sum = 0
        index_eval = 0

        # dropout_rate = 0.75

        print("===================== Accuracy List =====================")
        for i in range(0, epoch + 1):
            print('Epoch %d' % (i + 1), '   Test : %f' % accuracy_list[i], '   Time per Image : %f' % time_list[i])
        print("=========================================================")

    for i in range(0, ForEpoch):
        print('Epoch %d' % (i + 1), '   Test : %f' % accuracy_list[i], '   Time per Image : %f' % time_list[i])

    print("### Finish Training! ###")




