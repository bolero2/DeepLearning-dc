import tensorflow as tf

image_Width = 416
image_Height = 416
channel = 3
label_size = 20     # pascal VOC 2012 Dataset


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


def block_conv(input, ch_input, ch_output, stride, istraining, name):
    ksize = [3, 3, ch_input, ch_output]
    strides = [1, stride, stride, 1]
    n_out = ksize[-1]

    kernel = tf.Variable(tf.random.truncated_normal(ksize, stddev=0.1), name=name + '_weight')
    conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
    bn = batch_norm(conv, n_out=n_out, training=istraining)
    conv = tf.nn.leaky_relu(bn, name=name + '_leaky-RELU')
    b, h, w, c = conv.shape
    print(name + " output ->", "[" + str(h) + ", " + str(w) + ", " + str(c) + "]")

    return conv


def block_yolo(input, ch_input, ch_output, stride, istraining, name):
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
    b, h, w, c = total.shape
    print(name + " output ->", "[" + str(h) + ", " + str(w) + ", " + str(c) + "]")

    return total


class darknet53:
    def __init__(self, imgs, training, weights=None, sess=None):
        self.label_size = label_size
        self.imgs = imgs
        self.training = training
        self.convlayers()
        self.gap_layers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def convlayers(self):
        self.parameters = []
        self.layers = {}

        # initialization input node
        self.imgs = tf.reshape(self.imgs, shape=[-1, image_Height, image_Width, 3], name='input_node')
########################################################################################################################
        # conv1
        with tf.name_scope('conv1') as scope:
            self.conv1 = block_conv(self.imgs, ch_input=self.imgs.shape[-1],
                                    ch_output=32,
                                    stride=1,
                                    istraining=self.training,
                                    name=scope[:-1])

        # conv2
        with tf.name_scope('conv2') as scope:
            self.conv2 = block_conv(self.conv1, ch_input=self.conv1.shape[-1],
                                    ch_output=64,
                                    stride=2,
                                    istraining=self.training, name=scope[:-1])
########################################################################################################################
        # YOLO BLOCK
        # conv3
        with tf.name_scope('conv3') as scope:
            self.conv3 = block_yolo(self.conv2, ch_input=self.conv2.shape[-1],
                                    ch_output=self.conv2.shape[-1],
                                    stride=1,
                                    istraining=self.training, name=scope[:-1])
########################################################################################################################
        # conv4
        with tf.name_scope('conv4') as scope:
            self.conv4 = block_conv(self.conv3, ch_input=self.conv3.shape[-1],
                                    ch_output=128,
                                    stride=2,
                                    istraining=self.training, name=scope[:-1])
########################################################################################################################
        # YOLO BLOCK
        # conv5
        with tf.name_scope('conv5') as scope:
            self.conv5 = block_yolo(self.conv4, ch_input=self.conv4.shape[-1],
                                    ch_output=self.conv4.shape[-1],
                                    stride=1,
                                    istraining=self.training, name=scope[:-1])

        # conv6
        with tf.name_scope('conv6') as scope:
            self.conv6 = block_yolo(self.conv5, ch_input=self.conv5.shape[-1],
                                    ch_output=self.conv5.shape[-1],
                                    stride=1,
                                    istraining=self.training, name=scope[:-1])
########################################################################################################################
        # conv7
        with tf.name_scope('conv7') as scope:
            self.conv7 = block_conv(self.conv6, ch_input=self.conv6.shape[-1],
                                    ch_output=256,
                                    stride=2,
                                    istraining=self.training, name=scope[:-1])
########################################################################################################################
        # YOLO BLOCK
        # conv8
        with tf.name_scope('conv8') as scope:
            self.conv8 = block_yolo(self.conv7, ch_input=self.conv7.shape[-1],
                                    ch_output=self.conv7.shape[-1],
                                    stride=1,
                                    istraining=self.training, name=scope[:-1])

        # conv9
        with tf.name_scope('conv9') as scope:
            self.conv9 = block_yolo(self.conv8, ch_input=self.conv8.shape[-1],
                                    ch_output=self.conv8.shape[-1],
                                    stride=1,
                                    istraining=self.training, name=scope[:-1])

        # conv10
        with tf.name_scope('conv10') as scope:
            self.conv10 = block_yolo(self.conv9, ch_input=self.conv9.shape[-1],
                                     ch_output=self.conv9.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv11
        with tf.name_scope('conv11') as scope:
            self.conv11 = block_yolo(self.conv10, ch_input=self.conv10.shape[-1],
                                     ch_output=self.conv10.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv12
        with tf.name_scope('conv12') as scope:
            self.conv12 = block_yolo(self.conv11, ch_input=self.conv11.shape[-1],
                                     ch_output=self.conv11.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv13
        with tf.name_scope('conv13') as scope:
            self.conv13 = block_yolo(self.conv12, ch_input=self.conv12.shape[-1],
                                     ch_output=self.conv12.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv14
        with tf.name_scope('conv14') as scope:
            self.conv14 = block_yolo(self.conv13, ch_input=self.conv13.shape[-1],
                                     ch_output=self.conv13.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv15
        with tf.name_scope('conv15') as scope:
            self.conv15 = block_yolo(self.conv14, ch_input=self.conv14.shape[-1],
                                     ch_output=self.conv14.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])
########################################################################################################################
        # conv16
        with tf.name_scope('conv16') as scope:
            self.conv16 = block_conv(self.conv15, ch_input=self.conv15.shape[-1],
                                     ch_output=512,
                                     stride=2,
                                     istraining=self.training, name=scope[:-1])
########################################################################################################################
        # YOLO BLOCK
        # conv17
        with tf.name_scope('conv17') as scope:
            self.conv17 = block_yolo(self.conv16, ch_input=self.conv16.shape[-1],
                                    ch_output=self.conv16.shape[-1],
                                    stride=1,
                                    istraining=self.training, name=scope[:-1])

        # conv18
        with tf.name_scope('conv18') as scope:
            self.conv18 = block_yolo(self.conv17, ch_input=self.conv17.shape[-1],
                                    ch_output=self.conv17.shape[-1],
                                    stride=1,
                                    istraining=self.training, name=scope[:-1])

        # conv19
        with tf.name_scope('conv19') as scope:
            self.conv19 = block_yolo(self.conv18, ch_input=self.conv18.shape[-1],
                                     ch_output=self.conv18.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv20
        with tf.name_scope('conv20') as scope:
            self.conv20 = block_yolo(self.conv19, ch_input=self.conv19.shape[-1],
                                     ch_output=self.conv19.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv21
        with tf.name_scope('conv21') as scope:
            self.conv21 = block_yolo(self.conv20, ch_input=self.conv20.shape[-1],
                                     ch_output=self.conv20.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv22
        with tf.name_scope('conv22') as scope:
            self.conv22 = block_yolo(self.conv21, ch_input=self.conv21.shape[-1],
                                     ch_output=self.conv21.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv23
        with tf.name_scope('conv23') as scope:
            self.conv23 = block_yolo(self.conv22, ch_input=self.conv22.shape[-1],
                                     ch_output=self.conv22.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv24
        with tf.name_scope('conv24') as scope:
            self.conv24 = block_yolo(self.conv23, ch_input=self.conv23.shape[-1],
                                     ch_output=self.conv23.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])
########################################################################################################################
        # conv25
        with tf.name_scope('conv25') as scope:
            self.conv25 = block_conv(self.conv24, ch_input=self.conv24.shape[-1],
                                     ch_output=1024,
                                     stride=2,
                                     istraining=self.training, name=scope[:-1])
########################################################################################################################
        # YOLO BLOCK
        # conv26
        with tf.name_scope('conv26') as scope:
            self.conv26 = block_yolo(self.conv25, ch_input=self.conv25.shape[-1],
                                     ch_output=self.conv25.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv27
        with tf.name_scope('conv27') as scope:
            self.conv27 = block_yolo(self.conv26, ch_input=self.conv26.shape[-1],
                                     ch_output=self.conv26.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv28
        with tf.name_scope('conv28') as scope:
            self.conv28 = block_yolo(self.conv27, ch_input=self.conv27.shape[-1],
                                     ch_output=self.conv27.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])

        # conv29
        with tf.name_scope('conv29') as scope:
            self.conv29 = block_yolo(self.conv28, ch_input=self.conv28.shape[-1],
                                     ch_output=self.conv28.shape[-1],
                                     stride=1,
                                     istraining=self.training, name=scope[:-1])
########################################################################################################################

    def gap_layers(self):
        with tf.name_scope('gap') as scope:
            size = [1, self.conv28.shape[1], self.conv28.shape[2], 1]

            gap1 = tf.nn.avg_pool2d(self.conv29, ksize=size, strides=size, padding='VALID', name=scope)
            self.gap = gap1
            self.layers[scope[:-1]] = self.gap
            b, h, w, c = self.gap.shape
            print(scope[:-1] + " output ->", "[" + str(h) + ", " + str(w) + ", " + str(c) + "]")

    def fc_layers(self):
        with tf.name_scope('fc') as scope:
            # ksize = [1, 1, 512, 512]
            # strides = [1, 1, 1, 1]
            ksize = [self.gap.shape[-1], label_size]

            kernel = tf.Variable(tf.random.truncated_normal(ksize, stddev=0.1), name='weights_fc1')
            # conv = tf.nn.conv2d(self.gap, kernel, strides, padding='SAME')
            fc1 = tf.reshape(self.gap, shape=[-1, self.gap.shape[-1]])
            fc2 = tf.matmul(fc1, kernel)
            self.fc = fc2
            b, c = self.fc.shape
            # b, h, w, c = self.fc1.shape
            print(scope[:-1] + " output ->", "[" + str(b) + ", " + str(c) + "]")
            # print(scope[:-1] + " output ->", "[" + str(h) + ", " + str(w) + ", " + str(c) + "]")

    def load_weights(self, weight_file, sess):
        saver = tf.train.Saver()  # Network model Save
        meta_saver = tf.train.import_meta_graph(weight_file + ".meta")
        save_path = saver.restore(sess, weight_file)
        # weights = np.load(weight_file)
        # keys = sorted(weights.keys())
        # for i, k in enumerate(keys):
        #     print(i, k, np.shape(weights[k]))
        #     sess.run(self.parameters[i].assign(weights[k]))
