import tensorflow as tf
from class_utils import sigmoid
from class_darknet53 import block_conv, block_darknet, batch_norm

image_Width = 416
image_Height = 416
channel = 3
label_size = 20  # pascal VOC 2012 Dataset


def block_darknet_route(input, ch_input, ch_output, stride, istraining, name):
    ksize1 = [1, 1, ch_input, int(int(ch_input) / 2)]
    ksize2 = [3, 3, int(int(ch_input) / 2), ch_output]
    strides = [1, stride, stride, 1]

    kernel1 = tf.Variable(tf.random.truncated_normal(ksize1, stddev=0.1), name=name + '_weight_1')
    conv1 = tf.nn.conv2d(input, kernel1, strides, padding='SAME')
    bn1 = batch_norm(conv1, n_out=ksize1[-1], training=istraining)
    af1 = tf.nn.leaky_relu(bn1, name=name + '_leaky-RELU')

    kernel2 = tf.Variable(tf.random.truncated_normal(ksize2, stddev=0.1), name=name + '_weight_2')
    conv2 = tf.nn.conv2d(af1, kernel2, strides, padding='SAME')
    bn2 = batch_norm(conv2, n_out=ksize2[-1], training=istraining)
    af2 = tf.nn.leaky_relu(bn2, name=name + '_leaky-RELU')

    total = tf.add(input, af2)
    total = sigmoid(total)
    b, h, w, c = total.shape
    print(name + " output ->", "[" + str(h) + ", " + str(w) + ", " + str(c) + "]")

    return af1, total


def block_yolo(input, ch_input, ch_output, stride, istraining, name):
    return block_darknet_route(input, ch_input, ch_output, stride, istraining, name)


class yolo:
    """
    Build Yolo block
    """

    def __init__(self, input, training):
        self.label_size = label_size
        self.input = input
        self.training = training
        self.yololayers()
        # self.gap_layers()
        # self.fc_layers()
        # self.probs = tf.nn.softmax(self.fc)

    def yololayers(self):
        # yolo1
        with tf.name_scope('yolo1') as scope:
            self.yolo1 = block_darknet(self.input, ch_input=self.input.shape[-1],
                                       ch_output=self.input.shape[-1],
                                       stride=1,
                                       istraining=self.training,
                                       name=scope[:-1])

        # yolo2
        with tf.name_scope('yolo2') as scope:
            self.yolo2 = block_darknet(self.yolo1, ch_input=self.yolo1.shape[-1],
                                       ch_output=self.yolo1.shape[-1],
                                       stride=1,
                                       istraining=self.training,
                                       name=scope[:-1])

        # yolo3
        with tf.name_scope('yolo3') as scope:
            self.route, self.yolo3 = block_yolo(self.yolo2, ch_input=self.yolo2.shape[-1],
                                                ch_output=self.yolo2.shape[-1],
                                                stride=1,
                                                istraining=self.training,
                                                name=scope[:-1])
