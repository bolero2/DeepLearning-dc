import tensorflow as tf
import numpy as np
import cv2
import os
import time
import random

TrainDir = "D:\\3D_NUMBER_3\\Train\\"
EvalDir = "D:\\3D_NUMBER_3\\Test\\"

Filenames_Train = []
Filenames_Eval = []
index_train = 0
index_eval = 0
batchsize = 1
xList = []
oList = []
label_size = 10
Total_List = np.zeros([label_size], dtype=int)
framesize = 20
Total_Batch = int(162000 / (framesize))
Total_Test = int(3000 / (framesize))
image_Width = 112
image_Height = 112

channel = 3

Weights = {
    'wc1': tf.Variable(tf.truncated_normal([3, 7, 7, channel, 64], stddev=0.1)),
    'wc2': tf.Variable(tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.1)),
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 3, 64, 128], stddev=0.1)),
    'wc4': tf.Variable(tf.truncated_normal([3, 3, 3, 128, 128], stddev=0.1)),
    'wc5': tf.Variable(tf.truncated_normal([3, 3, 3, 128, 256], stddev=0.1)),
    'wc6': tf.Variable(tf.truncated_normal([3, 3, 3, 256, 256], stddev=0.1)),
    'wc7': tf.Variable(tf.truncated_normal([3, 3, 3, 256, 256], stddev=0.1)),
    'wc8': tf.Variable(tf.truncated_normal([3, 3, 3, 256, 512], stddev=0.1)),
    'wc9': tf.Variable(tf.truncated_normal([3, 3, 3, 512, 512], stddev=0.1)),
    'wc10': tf.Variable(tf.truncated_normal([3, 3, 3, 512, 512], stddev=0.1)),
    'wc11': tf.Variable(tf.truncated_normal([3, 3, 3, 512, 512], stddev=0.1)),
    'wc12': tf.Variable(tf.truncated_normal([3, 3, 3, 512, 512], stddev=0.1)),
    'wc13': tf.Variable(tf.truncated_normal([3, 3, 3, 512, 512], stddev=0.1)),
    'wfc1': tf.Variable(tf.truncated_normal([1 * 1 * 512, 512], stddev=0.1)),
    'wfc2': tf.Variable(tf.truncated_normal([512, label_size], stddev=0.1)),
}


def Load_Image():
    TempList = []

    for i in range(0, label_size):
        FileList = os.listdir(EvalDir + str(i))
        for j in range(0, len(FileList)):

            TempList.append(EvalDir + str(i) + '/' + FileList[j])
            if (j + 1) % framesize == 0 and j != 0:
                Filenames_Eval.append([TempList, i])
                TempList = []


def MS_camera():
    cv2.namedWindow('MyWindow')
    count = 0
    x_data = np.zeros([batchsize, framesize, image_Height, image_Width, channel], dtype=np.float32)
    cameraCapture = cv2.VideoCapture(0)
    success, frame = cameraCapture.read()

    while success and cv2.waitKey(1) == -1:
        print("frame count:", count)
        success, frame = cameraCapture.read()
        frame = cv2.resize(frame, (image_Height, image_Width))
        frame = frame / 255
        x_data[batchsize - 1][count] = frame
        count = count + 1
        cv2.imshow('MyWindow', cv2.resize(frame, (640, 480)))
        cv2.waitKey(100)
    cameraCapture.release()

    return x_data


def Batch_Eval(sess, batchsize):
    global index_eval
    y_data = np.zeros((batchsize, label_size), dtype=np.float32)  # one hot encoding
    for i in range(0, batchsize):
        y_data[i][Filenames_Eval[index_eval+i][1]] = 1
    index_eval += batchsize
    if index_eval + batchsize >= Total_Test:
        index_eval = 0
    return y_data


def batch_norm(input, n_out, training, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(training, true_fn=mean_var_with_update, false_fn=lambda :(ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
    return normed


Load_Image()

X = tf.placeholder(tf.float32, [batchsize, framesize, image_Height, image_Width, channel])
Y_ = tf.placeholder(tf.float32, [batchsize, label_size])
istraining = tf.placeholder(tf.bool, name='istraining')

# L1
C1 = tf.nn.relu(batch_norm(tf.nn.conv3d(X,Weights['wc1'],strides=[1,1,2,2,1],padding='SAME'),n_out=64,training=istraining))
print(C1)
C2 = tf.nn.relu(batch_norm(tf.nn.conv3d(C1,Weights['wc2'],strides=[1,1,1,1,1],padding='SAME'),n_out=64,training=istraining))
print(C2)
L1 = tf.nn.max_pool3d(C2,ksize=[1,1,2,2,1],strides=[1,1,2,2,1],padding='SAME')
print(L1)

# L2
C3 = tf.nn.relu(batch_norm(tf.nn.conv3d(L1,Weights['wc3'],strides=[1,1,1,1,1],padding='SAME'),n_out=128,training=istraining))
print(C3)
C4 = tf.nn.relu(batch_norm(tf.nn.conv3d(C3,Weights['wc4'],strides=[1,1,1,1,1],padding='SAME'),n_out=128,training=istraining))
print(C4)
L2 = tf.nn.max_pool3d(C4,ksize=[1,1,2,2,1],strides=[1,1,2,2,1],padding='SAME')
print(L2)

# L3
C5 = tf.nn.relu(batch_norm(tf.nn.conv3d(L2,Weights['wc5'],strides=[1,1,1,1,1],padding='SAME'),n_out=256,training=istraining))
print(C5)
C6 = tf.nn.relu(batch_norm(tf.nn.conv3d(C5,Weights['wc6'],strides=[1,1,1,1,1],padding='SAME'),n_out=256,training=istraining))
print(C6)
C7 = tf.nn.relu(batch_norm(tf.nn.conv3d(C6,Weights['wc7'],strides=[1,1,1,1,1],padding='SAME'),n_out=256,training=istraining))
print(C7)
L3 = tf.nn.max_pool3d(C7,ksize=[1,1,2,2,1],strides=[1,1,2,2,1],padding='SAME')
print(L3)


# L4
C8 = tf.nn.relu(batch_norm(tf.nn.conv3d(L3,Weights['wc8'],strides=[1,1,1,1,1],padding='SAME'),n_out=512,training=istraining))
print(C8)
C9 = tf.nn.relu(batch_norm(tf.nn.conv3d(C8,Weights['wc9'],strides=[1,1,1,1,1],padding='SAME'),n_out=512,training=istraining))
print(C9)
C10 = tf.nn.relu(batch_norm(tf.nn.conv3d(C9,Weights['wc10'],strides=[1,1,1,1,1],padding='SAME'),n_out=512,training=istraining))
print(C10)
L4 = tf.nn.max_pool3d(C10,ksize=[1,1,2,2,1],strides=[1,1,2,2,1],padding='SAME')
print(L4)

# L4
C11 = tf.nn.relu(batch_norm(tf.nn.conv3d(L4,Weights['wc11'],strides=[1,1,1,1,1],padding='SAME'),n_out=512,training=istraining))
C12 = tf.nn.relu(batch_norm(tf.nn.conv3d(C11,Weights['wc12'],strides=[1,1,1,1,1],padding='SAME'),n_out=512,training=istraining))
C13 = tf.nn.relu(batch_norm(tf.nn.conv3d(C12,Weights['wc13'],strides=[1,1,1,1,1],padding='SAME'),n_out=512,training=istraining))
L5 = tf.nn.max_pool3d(C13,ksize=[1,1,2,2,1],strides=[1,1,2,2,1],padding='SAME')
print("L5 SIZE : " + str(L5))


#FC
Global_Avg = tf.nn.avg_pool3d(L5, ksize=[1, framesize, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding='VALID', name='Global_Avg2')
print(Global_Avg)
FC1 = tf.reshape(Global_Avg, [-1, Weights['wfc1'].get_shape().as_list()[0]])
FC1 = tf.nn.relu(tf.matmul(FC1, Weights['wfc1']), name='fc1')
print(FC1)
FC2 = tf.matmul(FC1, Weights['wfc2'], name="outlabel")
print(FC2)

# SoftMax
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=FC2, labels=Y_))
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(FC2), 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.AdamOptimizer(0.00001 * batchsize).minimize(cross_entropy)
accuracy_sum = 0

# tf.summary.scalar("loss", cross_entropy)
# tf.summary.scalar("accuracy", accuracy)
# merge_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()  # network model saver variable
    accuracy_list = []

    save_path = saver.restore(sess, "D:/Saver/VGG16_3D/EPOCH_15/VGG_3D_Epoch_15.ckpt")

    while (True):
        ex = MS_camera()
        ey = Batch_Eval(sess, batchsize)
        # ex = np.reshape(ex, [batchsize, framesize, image_Height, image_Width, channel])
        start = time.time()
        ep, acc = sess.run([tf.argmax(FC2, 1), accuracy], feed_dict={X: ex, Y_: ey, istraining.name: False})
        end = time.time() - start
        for i in range(0, 1):
            print("label:", ep, "time:", end, "Accuracy: %f" % acc)   # accuracy is not correct!!! -- problem 1
        print("END")

