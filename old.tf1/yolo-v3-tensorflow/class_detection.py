import tensorflow as tf
from class_utils import sigmoid
from class_darknet53 import block_conv, block_darknet, batch_norm
from class_yolo import yolo

image_Width = 416
image_Height = 416
channel = 3
label_size = 20     # pascal VOC 2012 Dataset
label_full = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat',
              'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

anchors = [(10, 13), (16, 30), (33, 23),
           (30, 61), (62, 45), (59, 119),
           (116, 90), (156, 198), (373, 326)]
grid = 13


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


class detection:
    def __init__(self, input, training):
        self.label_size = label_size
        self.input = input
        self.training = training
        self.anchors = anchors
        self.num_anchors = len(self.anchors)
        self.detectionlayers()
        # self.gap_layers()
        # self.fc_layers()
        # self.probs = tf.nn.softmax(self.fc)

    def detectionlayers(self):
        self.predictions = block_conv(self.input, ch_input=self.input.shape[-1],
                                      ch_output=self.num_anchors * (5 + self.label_size),
                                      stride=1,
                                      istraining=self.training,
                                      name='detection_pred')
        self.shape = self.predictions.get_shape().as_list()
        dim = self.shape[1] * self.shape[2]
        bbox_attrs = 5 + self.label_size
        self.predictions = tf.reshape(self.predictions, [-1, self.num_anchors * dim, bbox_attrs])
        stride = (image_Height // self.shape[1], image_Width // self.shape[2])
        """
        box_centers = (x, y)
        box_sizes = (w, h)
        """
        box_centers, box_sizes, confidence, classes = tf.split(self.predictions, [2, 2, 1, self.label_size], axis=-1)
        box_centers = sigmoid(box_centers)
        confidence = sigmoid(confidence)

        grid_x = tf.range(self.shape[1], dtype=tf.float32)
        grid_y = tf.range(self.shape[2], dtype=tf.float32)
        a, b = tf.meshgrid(grid_x, grid_y)

        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))

        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, self.num_anchors]), [1, -1, 2])

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * stride

        anchors = tf.tile(self.anchors, [dim, 1])
        box_sizes = tf.exp(box_sizes) * anchors
        box_sizes = box_sizes * stride

        detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

        classes = tf.nn.sigmoid(classes)
        predictions = tf.concat([detections, classes], axis=-1)

        # grid_size =