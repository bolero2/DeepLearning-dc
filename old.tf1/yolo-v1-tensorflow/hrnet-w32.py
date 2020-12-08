import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

image_Width = 448
image_Height = 448
channel = 3
label_size = 20     # pascal VOC 2012 Dataset
grid = 7
batchsize = 1
Learning_Rate = 0.00001

box_per_cell = 2        # one cell have 2 box
boundary1 = grid * grid * label_size  # 7 * 7 * 20
boundary2 = boundary1 + grid * grid * box_per_cell  # 7 * 7 * 20 + 7 * 7 *2

w = 32


def sigmoid(x):
    y = tf.math.sigmoid(x)

    return y


def batch_norm(input, n_out, training, scope='bn'):
    with tf.compat.v1.variable_scope(scope):
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
############


def block_conv(input, ksize, ch_input, ch_output, stride, istraining, name):
    ksize = [ksize, ksize, int(ch_input), ch_output]
    strides = [1, stride, stride, 1]
    n_out = ksize[-1]

    kernel = tf.Variable(tf.random.truncated_normal(ksize, stddev=0.1), name=name + '_weight')
    conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
    bn = batch_norm(conv, n_out=int(conv.shape[-1]), training=istraining)
    conv = tf.nn.leaky_relu(bn, name=name + '_leaky-RELU')
    b, h, w, c = conv.shape
    print(name + " output ->", "[" + str(h) + ", " + str(w) + ", " + str(c) + "]")

    return conv


def block_residual(input, ch_output1, ch_output2, stride, istraining, name):
    ksize1 = [1, 1, int(input.shape[-1]), ch_output1]
    ksize2 = [3, 3, ch_output1, ch_output2]
    strides = [1, stride, stride, 1]

    kernel1 = tf.Variable(tf.random.truncated_normal(ksize1, stddev=0.1), name=name + '_weight_1')
    conv1 = tf.nn.conv2d(input, kernel1, strides, padding='SAME')
    bn1 = batch_norm(conv1, n_out=ksize1[-1], training=istraining)
    af1 = tf.nn.leaky_relu(bn1, name=name + '_leaky-RELU')

    b, h, w, c = af1.shape
    print(name + " output ->", "[" + str(h) + ", " + str(w) + ", " + str(c) + "]")

    kernel2 = tf.Variable(tf.random.truncated_normal(ksize2, stddev=0.1), name=name + '_weight_2')
    conv2 = tf.nn.conv2d(af1, kernel2, strides, padding='SAME')
    bn2 = batch_norm(conv2, n_out=ksize2[-1], training=istraining)
    af2 = tf.nn.leaky_relu(bn2, name=name + '_leaky-RELU')

    b, h, w, c = af2.shape
    print(name + " output ->", "[" + str(h) + ", " + str(w) + ", " + str(c) + "]")

    return af2


def block_upsample(input, name, method="deconv"):
    assert method in ["resize", "deconv"]

    with tf.variable_scope(name):
        if method == "resize":      # case in resize
            input_shape = tf.shape(input)
            output = tf.image.resize_nearest_neighbor(input, (input_shape[1] * 2, input_shape[2] * 2))

            return output

        if method == "deconv":    # case in deconvolution(transpose)
            # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
            numm_filter = input.shape.as_list()[-1]
            output = tf.layers.conv2d_transpose(input, numm_filter, kernel_size=2, padding='same',
                                                strides=(2, 2), kernel_initializer=tf.random_normal_initializer())
            return output


def block_maxpool(input, name):
    ret = tf.nn.max_pool2d(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    b, h, w, c = ret.shape
    print(name + " output ->", "[" + str(h) + ", " + str(w) + ", " + str(c) + "]")

    return ret


def block_bottleneck(input, ch_output, istraining, residual=None, name='bottleneck'):
    exp = 4
    res = input
    ch_input = int(input.shape[-1])

    out = block_conv(input=input, ksize=1, ch_input=ch_input, ch_output=ch_output, stride=1, istraining=istraining,
                     name=name)
    out = block_conv(input=out, ksize=3, ch_input=ch_output, ch_output=ch_output, stride=1, istraining=istraining,
                     name=name)
    out = block_conv(input=out, ksize=1, ch_input=ch_output, ch_output=ch_output * exp, stride=1, istraining=istraining,
                     name=name)

    if residual is not None:
        res = residual

    out += res
    out = tf.nn.relu(out)

    return out


def block_basic(input, ch_output, istraining, residual=None, name=None):
    exp = 1
    res = input
    ch_input = int(input.shape[-1])

    out = block_conv(input=input, ksize=3, ch_input=ch_input, ch_output=ch_output, stride=1, istraining=istraining,
                     name=name)
    out = block_conv(input=out, ksize=3, ch_input=ch_output, ch_output=ch_output, stride=1, istraining=istraining,
                     name=name)

    if residual is not None:
        res = residual

    out += res
    out = tf.nn.relu(out)

    return out


def block_stage(input, stage, output_branches, w, istraining, name=None):
    branches = list()

    for i in range(0, stage):
        stage_w = w * (2 ** i)
        out = block_basic(input, stage_w, istraining=istraining, residual=None, name=name)
        out = block_basic(out, stage_w, istraining=istraining, residual=None, name=name)
        out = block_basic(out, stage_w, istraining=istraining, residual=None, name=name)
        out = block_basic(out, stage_w, istraining=istraining, residual=None, name=name)
        branches.append(out)

    fuse_layers = list()
    input22 = branches[0]
    for i in range(0, output_branches):
        fuse_layers = list()
        for j in range(0, stage):
            if i == j:
                pass
            elif i < j:
                temp1 = block_conv(input=input22, ksize=3, ch_input=input22.shape[-1], ch_output=w * (2 ** i), stride=1,
                                   istraining=istraining, name=name)
                b, w, h, c = temp1.shape
                scale_factor = 2.0 ** (j - i)
                temp2 = tf.keras.layers.UpSampling2D(size=(w * scale_factor, h * scale_factor),
                                                     data_format="channels_last", interpolation='nearest')(temp1)
                fuse_layers[-1].append(temp2)
            elif i > j:
                ops = list()
                for k in range(i - j - 1):
                    temp2 = block_conv(input=input22, ksize=3, ch_input=input22.shape[-1], ch_output=w * (2 ** j),
                                       stride=2, istraining=istraining, name=name)
                    ops.append(temp2)
                temp3 = block_conv(input=temp2, ksize=3, ch_input=temp2.shape[-1], ch_output=w * (2 ** i),
                                   stride=2, istraining=istraining, name=name)
                ops.append(temp3)
                fuse_layers[-1].append(ops)

    return fuse_layers[-1]


class network:
    """
    Builds Darknet-53 model.
    """
    def __init__(self, imgs, training, weights=None, sess=None):
        self.w = w
        self.label_size = label_size
        self.imgs = imgs
        self.training = training
        self.grid = grid
        self.reshapelayers()
        self.hrlayers()
        # self.gap_layers()
        # self.fc_layers()
        # self.probs = tf.nn.softmax(self.fc)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def reshapelayers(self):
        print("reshape layers")
        ################################################################################################################
        # initialization input node
        self.imgs = tf.reshape(self.imgs, shape=[-1, image_Height, image_Width, 3], name='input_node')
        ################################################################################################################

    def hrlayers(self):
        print("hr layers")
        out = block_conv(self.imgs, ksize=3, ch_input=self.imgs.shape[-1], ch_output=64, stride=2,
                         istraining=self.training, name='conv1')
        out = block_conv(out, ksize=3, ch_input=out.shape[-1], ch_output=64, stride=2,
                         istraining=self.training, name='conv2')
        # out = block_conv(out, ksize=3, ch_input=out.shape[-1], ch_output=64, stride=2,
        #                  istraining=self.training, name='conv3')
        downsample = block_conv(out, ksize=3, ch_input=out.shape[-1], ch_output=256, stride=1,
                                istraining=self.training, name='downsample')

        # Layer 1
        out = block_bottleneck(out, 64, downsample)
        out = block_bottleneck(out, 64, istraining=self.training)
        out = block_bottleneck(out, 64, istraining=self.training)
        out = block_bottleneck(out, 64, istraining=self.training)

        ################################################################################################################
        # Fusion Layer 1
        ################################################################################################################
        out = block_conv(out, ksize=3, ch_input=out.shape[-1], ch_output=self.w, stride=1,
                         istraining=self.training, name='transition1-1')
        out = block_conv(out, ksize=3, ch_input=out.shape[-1], ch_output=self.w * (2 ** 1), stride=2,
                         istraining=self.training, name='transition1-2')

        ################################################################################################################
        # stage 2
        ################################################################################################################
        out = block_stage(out, stage=2, output_branches=2, w=self.w, istraining=self.training, name='stage2')

        ################################################################################################################
        # Fusion Layer 2
        ################################################################################################################
        out = block_conv(out, ksize=3, ch_input=out.shape[-1], ch_output=self.w * (2 ** 2), stride=2,
                         istraining=self.training, name='transition2-1')

        ################################################################################################################
        # stage 3
        ################################################################################################################
        out = block_stage(out, stage=3, output_branches=3, w=self.w, istraining=self.training, name='stage3')
        out = block_stage(out, stage=3, output_branches=3, w=self.w, istraining=self.training, name='stage3')
        out = block_stage(out, stage=3, output_branches=3, w=self.w, istraining=self.training, name='stage3')
        out = block_stage(out, stage=3, output_branches=3, w=self.w, istraining=self.training, name='stage3')

        ################################################################################################################
        # Fusion Layer 3
        ################################################################################################################
        out = block_conv(out, ksize=3, ch_input=out.shape[-1], ch_output=self.w * (2 ** 3), stride=2,
                         istraining=self.training, name='transition3-1')

        ################################################################################################################
        # stage 4
        ################################################################################################################
        out = block_stage(out, stage=4, output_branches=4, w=self.w, istraining=self.training, name='stage4')
        out = block_stage(out, stage=4, output_branches=4, w=self.w, istraining=self.training, name='stage4')
        out = block_stage(out, stage=4, output_branches=1, w=self.w, istraining=self.training, name='stage4')

        ################################################################################################################
        # Final Layer
        ################################################################################################################
        self.output = block_conv(out, ksize=1, ch_input=out.shape[-1], ch_output=self.label_size, stride=1,
                                 istraining=self.training, name='final')

    def load_weights(self, weight_file, sess):
        print(f"Weight Loading Start! -> {weight_file}")
        saver = tf.compat.v1.train.Saver()  # Network model Save
        meta_saver = tf.compat.v1.train.import_meta_graph(weight_file + ".meta")
        save_path = saver.restore(sess, weight_file)
        print(f"Weight Loading is successful")


def calc_iou(boxes1, boxes2, scope='iou'):
    """calculate ious
    Args:
      boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
    Return:
      iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    with tf.variable_scope(scope):
        # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
        boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                             boxes1[..., 1] - boxes1[..., 3] / 2.0,
                             boxes1[..., 0] + boxes1[..., 2] / 2.0,
                             boxes1[..., 1] + boxes1[..., 3] / 2.0],
                            axis=-1)

        boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                             boxes2[..., 1] - boxes2[..., 3] / 2.0,
                             boxes2[..., 0] + boxes2[..., 2] / 2.0,
                             boxes2[..., 1] + boxes2[..., 3] / 2.0],
                            axis=-1)

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
        rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[..., 0] * intersection[..., 1]

        # calculate the boxs1 square and boxs2 square
        square1 = boxes1[..., 2] * boxes1[..., 3]
        square2 = boxes2[..., 2] * boxes2[..., 3]

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def loss_layer(predicts, labels, scope='loss_layer'):
    object_scale = 1.0
    noobject_scale = 1.0
    class_scale = 2.0
    coord_scale = 5.0

    with tf.variable_scope(scope):
        # print(boundary1)
        # print(predicts[:, :boundary1])
        predict_classes = tf.reshape(predicts[:, :boundary1], [batchsize, grid, grid, label_size])
        predict_scales = tf.reshape(predicts[:, boundary1:boundary2], [batchsize, grid, grid, box_per_cell])
        predict_boxes = tf.reshape(predicts[:, boundary2:], [batchsize, grid, grid, box_per_cell, 4])

        response = tf.reshape(labels[..., 0], [batchsize, grid, grid, 1])       # response = confidence score
        boxes = tf.reshape(labels[..., 1:5], [batchsize, grid, grid, 1, 4])
        boxes = tf.tile(boxes, [1, 1, 1, box_per_cell, 1]) / image_Width
        classes = labels[..., 5:]
        offset = np.transpose(np.reshape(np.array([np.arange(grid)] * grid * box_per_cell), (box_per_cell, grid, grid)),
                              (1, 2, 0))
        offset = tf.reshape(tf.constant(offset, dtype=tf.float32), [1, grid, grid, box_per_cell])    # Reshape from 7*7*2 to 1*7*7*2
        offset = tf.tile(offset, [batchsize, 1, 1, 1])  # Copy on the first dimension and become [batchsize, 7, 7, 2]
        offset_tran = tf.transpose(offset, (0, 2, 1, 3))  # The dimension is [batchsize, 7, 7, 2]

        predict_boxes_tran = tf.stack(
            [(predict_boxes[..., 0] + offset) / grid,
             (predict_boxes[..., 1] + offset_tran) / grid,
             tf.square(predict_boxes[..., 2]),
             tf.square(predict_boxes[..., 3])], axis=-1)

        iou_predict_truth = calc_iou(predict_boxes_tran, boxes)

        # calculate I tensor [batchsize, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        boxes_tran = tf.stack(
            [boxes[..., 0] * grid - offset,
             boxes[..., 1] * grid - offset_tran,
             tf.sqrt(boxes[..., 2]),
             tf.sqrt(boxes[..., 3])], axis=-1)

        # class_loss, calculate the loss of the category
        class_delta = response * (predict_classes - classes)
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * class_scale

        object_delta = object_mask * (predict_scales - iou_predict_truth)
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * object_scale

        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * noobject_scale

        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * coord_scale

        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobject_loss)
        tf.losses.add_loss(coord_loss)

        tf.summary.scalar('class_loss', class_loss)
        tf.summary.scalar('object_loss', object_loss)
        tf.summary.scalar('noobject_loss', noobject_loss)
        tf.summary.scalar('coord_loss', coord_loss)

        tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
        tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
        tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
        tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
        tf.summary.histogram('iou', iou_predict_truth)

    return class_loss + object_loss + noobject_loss + coord_loss
