"""
purpose : only one image
"""

from class_vgg import vgg19
# from class_densevgg19 import vgg19
import tensorflow as tf
import numpy as np
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
from classes import class_names
import cv2
import copy
import time

before_pointer = list()
batchsize = 1
image_Height = 224
image_Width = 224
channel = 3
contour_flag = 0


def threshold(img):
    img_gray = copy.deepcopy(255 * img)
    img_gray = img_gray.astype(np.uint8)
    ret, img_binary = cv2.threshold(img_gray, (255 * 0.1), 255, 0)

    return img_binary


def MS_camera(cameraCapture):
    success, frame = cameraCapture.read()  # discard first frame image!
    frame_original = copy.deepcopy(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (image_Height, image_Width))
    frame_original = cv2.resize(frame_original, (image_Height, image_Width))
    x = np.expand_dims(frame, axis=0)
    x = x[:, :, :, ::-1]
    return x, frame, frame_original


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


def grad_cam(x, network, sess, predicted_class, layer_name, nb_classes):
    print("Setting gradients to 1 for target class and rest to 0")
    # Conv layer tensor [?,7,7,512]
    conv_layer = network.layers[layer_name]
    # [1000]-D tensor with target class index set to 1 and rest as 0
    one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
    signal = tf.math.multiply(network.layers['fc3'], one_hot)
    loss = tf.reduce_mean(signal)

    grads = tf.gradients(loss, conv_layer)[0]
    # Normalizing the gradients
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={network.imgs: x, network.training: False})
    output = output[0]           # [7,7,512]
    grads_val = grads_val[0]	 # [7,7,512]

    weights = np.mean(grads_val, axis=(0, 1)) 			    # [512]
    cam = np.ones(output.shape[0: 2], dtype=np.float32) 	# [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = resize(cam, (224, 224))

    # Converting grayscale to 3-D
    cam3 = np.expand_dims(cam, axis=2)
    cam3 = np.tile(cam3, [1, 1, 3])

    return cam3


def find_location(img):
    global contour_flag
    global before_pointer
    data = 255 * img
    img = data.astype(np.uint8)
    if img.shape[-1] == 3:
        img_color = copy.deepcopy(img)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        contour_flag = 3
        thresh_hold = 127
    else:
        img_gray = copy.deepcopy(img)
        contour_flag = 1
        thresh_hold = 127
    ret, img_binary = cv2.threshold(img_gray, thresh_hold, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    temparray_x = list()
    temparray_y = list()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        temparray_x += [x, x + w]
        temparray_y += [y, y + h]
    print("temparray_X:", temparray_x)
    print("temparray_Y:", temparray_y)

    if temparray_x == [] or temparray_y == []:
        return before_pointer

    else:
        min_x = min(temparray_x) - 1
        max_x = max(temparray_x) + 1
        min_y = min(temparray_y) - 1
        max_y = max(temparray_y) + 1
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x > img.shape[0]:
            max_x = img.shape[0]
        if max_y > img.shape[1]:
            max_y = img.shape[1]
        pointer = [min_x, min_y, max_x, max_y]
        before_pointer = pointer

        return pointer


def find_location2(img):
    global contour_flag
    global before_pointer
    pointer = list()
    extpointer = list()
    img = img.astype(np.uint8)
    if img.shape[-1] == 3:
        img_color = copy.deepcopy(img)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        contour_flag = 3
    else:
        img_gray = copy.deepcopy(img)
        contour_flag = 1
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 255, 255), 5)
        # pointer.append([x, y, w, h])
        if w * h < 224 * 224 * 0.01:
            continue
        else:
            cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 255, 255), 5)
            pointer.append([x, y, w, h])

            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
            bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
            extpointer.append([leftmost, rightmost, topmost, bottommost])

    print("Pointer in find_location2 ->", pointer)
    print("Extreme Pointer in find_location2 ->", extpointer)

    return pointer, extpointer


def main(_):
    cv2.namedWindow("result window")

    sess = tf.Session()

    imgs = tf.placeholder(tf.float32, [batchsize, 224, 224, 3])
    istraining = tf.placeholder(tf.bool, name='istraining')
    # network = vgg19(imgs, istraining, "D:\\Saver\\3lab_fingertip_vgg19_3\\3lab_fingertip_vgg19_3_Epoch_6.ckpt", sess)
    # network = vgg19(imgs, istraining, "D:\\Saver\\3lab_fingertip_vgg19_4\\3lab_fingertip_vgg19_4_Epoch_21.ckpt", sess)
    network = vgg19(imgs, istraining, "D:\\Saver\\3lab_fingertip_originalVGGNet19_1\\3lab_fingertip_originalVGGNet19_1_Epoch_18.ckpt", sess)

    conv_layer = network.layers['conv16']
    one_hot = tf.sparse_to_dense(1, [3], 1.0)       # correct label, [label size], percentage
    signal = tf.math.multiply(network.layers['fc3'], one_hot)
    loss = tf.reduce_mean(signal)

    grads = tf.gradients(loss, conv_layer)[0]
    print("\n")
    print(" *** LOSS:", loss)
    print(" *** conv_layer:", conv_layer)
    print(" *** grads:", grads)
    # grads = tf.gradients(loss, grads)[0]
    # Normalizing the gradients
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    total_time = 0

    for nc in range(0, 102):
        # test = cv2.imread("C:\\Users\\bolero\\Desktop\\paper_images\\test_point.jpg")
        start = time.time()
        test = cv2.imread("C:\\Users\\bolero\\Desktop\\works0903\\work\\segmentation\\data2\\exp" + str(nc) + ".jpg")
        # test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
        test = cv2.resize(test, (image_Height, image_Width))
        test2 = copy.deepcopy(test)
        x = np.zeros([1, image_Height, image_Width, 3])
        x[0, :, :, :] = test

        prob, output, grads_val = sess.run([network.probs, conv_layer, norm_grads], feed_dict={network.imgs: x, network.training: False})
        print('\nTop 1 classes:', prob[0].tolist().index(np.max(prob)), "\nAccuracy:",
              str(round(prob[0][prob[0].tolist().index(np.max(prob))] * 100, 2)) + "%")

        if prob[0].tolist().index(np.max(prob)) == 1:
            output = output[0]  # [7,7,512]
            grads_val = grads_val[0]  # [7,7,512]
            # Target class
            # predicted_class = preds[0]
            # # Target layer for visualization
            # layer_name = FLAGS.layer_name
            # # Number of output classes of model being used
            # nb_classes = 6

            """
            @@@@@ Grad-CAM Making Process! @@@@@
            """
            weights = np.mean(grads_val, axis=(0, 1))  # [512]
            cam = np.ones(output.shape[0: 2], dtype=np.float32)  # [7,7]

            # Taking a weighted average
            for i, w in enumerate(weights):
                cam += w * output[:, :, i]

            # Passing through ReLU
            cam = np.maximum(cam, 0)
            cam = cam / np.max(cam)
            cam = resize(cam, (224, 224))

            # Converting grayscale to 3-D
            cam3 = np.expand_dims(cam, axis=2)
            cam3 = np.tile(cam3, [1, 1, 3])

            img = test.astype(float)
            img /= img.max()

            # Superimposing the visualization with the image.
            new_img = img + 3 * cam3
            new_img /= new_img.max()

            """
            @@@@@ SHOW! @@@@@= 
            """
            # Display and save
            # new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
            pointer = find_location(new_img)        # grad-cam position

            x = np.zeros([1, image_Height, image_Width, 3], np.uint8)
            img_focus = cv2.resize(copy.deepcopy(test[pointer[1]:pointer[3], pointer[0]:pointer[2], :]), (224, 224))

            # cv2.rectangle(test2, (pointer[0], pointer[1]), (pointer[2], pointer[3]), color=(0, 0, 255), thickness=3)

            # cv2.imwrite("C:\\Users\\bolero\\Desktop\\grad-cam.jpg", img_focus)
            # cv2.imwrite("C:\\Users\\bolero\\Desktop\\rectangle_test2.jpg", test2)

            # cv2.imshow("result window", test)
            # cv2.imshow("result window2", test2)
            cv2.imshow("grad-cam", img_focus)
            # cv2.waitKey(0)

            x[0, :, :, :] = copy.deepcopy(img_focus)

            fm_conv4, fm_conv7, fm_conv10, fm_conv13, fm_conv16 = sess.run(
                [network.conv4, network.conv7, network.conv10, network.conv13, network.conv16],
                feed_dict={network.imgs: x, network.training: False})
            print('\nTop 1 classes:', prob[0].tolist().index(np.max(prob)), "\nAccuracy:",
                  str(round(prob[0][prob[0].tolist().index(np.max(prob))] * 100, 2)) + "%")

            print("fm_conv4:", fm_conv4.shape)
            print("fm_conv7:", fm_conv7.shape)
            print("fm_conv10:", fm_conv10.shape)
            print("fm_conv13:", fm_conv13.shape)
            print("fm_conv16:", fm_conv16.shape)

            fm_temp4 = np.zeros(shape=[112, 112])
            fm_temp7 = np.zeros(shape=[56, 56])
            fm_temp10 = np.zeros(shape=[28, 28])
            fm_temp13 = np.zeros(shape=[14, 14])
            fm_temp16 = np.zeros(shape=[14, 14])

            if prob[0].tolist().index(np.max(prob)) == 1:
                ############################################################################################################

                for i in range(0, fm_conv4.shape[-1]):
                    fm_temp4 = fm_temp4 + fm_conv4[0, :, :, i]

                fm_temp4 = fm_temp4 / np.max(fm_temp4)
                fm_temp4_2 = cv2.resize(fm_temp4, (224, 224), cv2.INTER_LINEAR)
                # ret, fm_temp13_2 = cv2.threshold(fm_temp13_2, thresh=0.5, maxval=1, type=0)
                # plt.subplot(221)
                # plt.imshow(fm_temp13_2)

                ############################################################################################################

                for i in range(0, fm_conv7.shape[-1]):
                    fm_temp7 = fm_temp7 + fm_conv7[0, :, :, i]

                fm_temp7 = fm_temp7 / np.max(fm_temp7)
                fm_temp7_2 = cv2.resize(fm_temp7, (224, 224), cv2.INTER_LINEAR)
                # ret, fm_temp10_2 = cv2.threshold(fm_temp10_2, thresh=0.5, maxval=1, type=0)
                # plt.subplot(222)
                # plt.imshow(fm_temp10_2)

                ############################################################################################################

                for i in range(0, fm_conv10.shape[-1]):
                    fm_temp10 = fm_temp10 + fm_conv10[0, :, :, i]

                fm_temp10 = fm_temp10 / np.max(fm_temp10)
                fm_temp10_2 = cv2.resize(fm_temp10, (224, 224), cv2.INTER_LINEAR)
                # plt.subplot(223)
                # plt.imshow(fm_temp7_2)

                ############################################################################################################

                for i in range(0, fm_conv13.shape[-1]):
                    fm_temp13 = fm_temp13 + fm_conv13[0, :, :, i]

                fm_temp13 = fm_temp13 / np.max(fm_temp13)
                fm_temp13_2 = cv2.resize(fm_temp13, (224, 224), cv2.INTER_LINEAR)
                # plt.subplot(224)
                # plt.imshow(fm_temp4_2)
                # plt.show()

                ############################################################################################################

                for i in range(0, fm_conv16.shape[-1]):
                    fm_temp16 = fm_temp16 + fm_conv16[0, :, :, i]

                fm_temp16 = fm_temp16 / np.max(fm_temp16)
                fm_temp16_2 = cv2.resize(fm_temp16, (224, 224), cv2.INTER_LINEAR)
                # plt.subplot(224)
                # plt.imshow(fm_temp4_2)
                # plt.show()

                ############################################################################################################

                """
                @@@@@ configure your feature  map! @@@@@
                """
                new_fm = fm_temp4_2 * fm_temp7_2 * fm_temp10_2 * fm_temp13_2 * fm_temp16_2
                new_fm_copy = copy.deepcopy(new_fm)

                # cv2.imwrite("C:\\Users\\bolero\\Desktop\\fm_temp4_2.jpg", fm_temp4_2 * 255)
                # cv2.imwrite("C:\\Users\\bolero\\Desktop\\fm_temp7_2.jpg", fm_temp7_2 * 255)
                # cv2.imwrite("C:\\Users\\bolero\\Desktop\\fm_temp10_2.jpg", fm_temp10_2 * 255)
                # cv2.imwrite("C:\\Users\\bolero\\Desktop\\fm_temp13_2.jpg", fm_temp13_2 * 255)
                # cv2.imwrite("C:\\Users\\bolero\\Desktop\\fm_temp16_2.jpg", fm_temp16_2 * 255)
                # cv2.imwrite("C:\\Users\\bolero\\Desktop\\new_fm.jpg", new_fm * 255)

                """
                @@@@@ SHOW! @@@@@
                """
                print("new_fm shape:", new_fm.shape)
                # plt.subplot(331).set_title("Original Image")
                # plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
                # plt.xticks([])
                # plt.yticks([])
                #
                # plt.subplot(332).set_title("grad-cam output")
                # plt.imshow(cam)
                # plt.xticks([])
                # plt.yticks([])
                #
                # plt.subplot(333).set_title("feature map 4 sum")
                # plt.imshow(fm_temp4_2)
                # plt.xticks([])
                # plt.yticks([])
                #
                # plt.subplot(334).set_title("feature map 7 sum")
                # plt.imshow(fm_temp7_2)
                # plt.xticks([])
                # plt.yticks([])
                #
                # plt.subplot(335).set_title("feature map 10 sum")
                # plt.imshow(fm_temp10_2)
                # plt.xticks([])
                # plt.yticks([])
                #
                # plt.subplot(336).set_title("feature map 13 sum")
                # plt.imshow(fm_temp13_2)
                # plt.xticks([])
                # plt.yticks([])
                #
                # plt.subplot(337).set_title("feature map 16 sum")
                # plt.imshow(fm_temp16_2)
                # plt.xticks([])
                # plt.yticks([])
                #
                # plt.subplot(338 ).set_title("feature map sum")
                # plt.imshow(new_fm_copy)
                # plt.xticks([])
                # plt.yticks([])

                # plt.show()
                # Display and save
                th_new_fm = threshold(new_fm)
                pointer2, exp_pointer = find_location2(th_new_fm)
                print(pointer2)
                end = time.time() - start
                if nc > 0:
                    total_time = total_time + end

                for c in range(0, len(pointer2)):
                    # cv2.rectangle(cp_new_fm1, (pointer2[c][0], pointer2[c][1]), (pointer2[c][0] + pointer2[c][2], pointer2[c][1] + pointer2[c][3]), (1, 1, 1), 2)
                    pct_x = exp_pointer[c][2][0] / 224
                    pct_y = exp_pointer[c][2][1] / 224
                    cv2.rectangle(new_fm_copy, (int(pct_x * 224), int(pct_y * 224)), (int(pct_x * 224), int(pct_y * 224)),
                                  (1, 1, 1), 10)
                    cv2.rectangle(test2, (pointer[0] + int((pointer[2] - pointer[0]) * pct_x),
                                                  pointer[1] + int((pointer[3] - pointer[1]) * pct_y)),
                                  (pointer[0] + int((pointer[2] - pointer[0]) * pct_x),
                                   pointer[1] + int((pointer[3] - pointer[1]) * pct_y)), color=(0, 0, 255), thickness=10)
                # cv2.rectangle(new_fm, (pointer2[0], pointer2[1]), (pointer2[2], pointer2[3]), color=(255, 255, 255), thickness=3)
                # cv2.rectangle(original, (pointer[0] + pointer2[0], pointer[1] + pointer2[1]), (pointer[0] + pointer2[2], pointer[0] + pointer2[3]), color=(255, 0, 0), thickness=3)
                # cv2.rectangle(test, (pointer[0] + pointer2[0], pointer[1] + pointer2[1]),
                #               (pointer[0] + pointer2[0], pointer[1] + pointer2[1]), color=(255, 255, 255), thickness=10)


                cv2.imshow("original window", test)
                cv2.imshow("result window2", test2)
                cv2.imshow("th_new_fm", th_new_fm)
                cv2.imshow("fm_focus", img_focus)
                cv2.waitKey(1)

                # cv2.imwrite('./densenet+vgg_exp_result/result/output' + str(nc) + '.jpg', test2)
                # cv2.imwrite('./densenet+vgg_exp_result/fm/fm' + str(nc) + '.jpg', new_fm * 255)

                # cv2.imwrite("C:\\Users\\bolero\\Desktop\\th_fm.jpg", th_new_fm)
                # cv2.imwrite("C:\\Users\\bolero\\Desktop\\test_pointed.jpg", test2)
    print("Average time:", total_time / 100)


if __name__ == '__main__':
    print("main.py start...")
    tf.app.run()
    print("main.py end...")
