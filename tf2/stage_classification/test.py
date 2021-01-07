import os
import random
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model


def stage(tensor, multiplier, stride=1):
    residual = tensor
    in_channels = residual.shape[-1]

    origin = keras.layers.Conv2D(filters=int(in_channels) * multiplier, kernel_size=1, strides=(stride, stride), padding='SAME')(tensor)
    origin = keras.layers.BatchNormalization()(origin)
    origin = keras.layers.ReLU()(origin)

    residual = keras.layers.Conv2D(filters=int(in_channels) * multiplier, kernel_size=3, strides=(stride, stride), padding='SAME')(residual)
    residual = keras.layers.BatchNormalization()(residual)
    residual = keras.layers.ReLU()(residual)

    elem1 = keras.layers.DepthwiseConv2D(kernel_size=3, strides=(stride, stride), padding='SAME', depth_multiplier=multiplier)(tensor)
    elem2 = keras.layers.DepthwiseConv2D(kernel_size=3, strides=(stride, stride), padding='SAME', depth_multiplier=multiplier, dilation_rate=2)(tensor)
    elem3 = keras.layers.DepthwiseConv2D(kernel_size=3, strides=(stride, stride), padding='SAME', depth_multiplier=multiplier, dilation_rate=4)(tensor)
    elem4 = keras.layers.DepthwiseConv2D(kernel_size=3, strides=(stride, stride), padding='SAME', depth_multiplier=multiplier, dilation_rate=8)(tensor)

    elem1 = keras.layers.BatchNormalization()(elem1)
    elem2 = keras.layers.BatchNormalization()(elem2)
    elem3 = keras.layers.BatchNormalization()(elem3)
    elem4 = keras.layers.BatchNormalization()(elem4)

    elem1 = keras.layers.ReLU()(elem1)
    elem2 = keras.layers.ReLU()(elem2)
    elem3 = keras.layers.ReLU()(elem3)
    elem4 = keras.layers.ReLU()(elem4)

    print(origin.shape, residual.shape, elem1.shape, elem2.shape, elem3.shape, elem4.shape)
    total = keras.layers.Add()([origin, residual, elem1, elem2, elem3, elem4])

    return total


def network():
    multiplier = 0

    input_tensor = keras.layers.Input(shape=(image_size, image_size, channel))
    out = keras.layers.Conv2D(filters=32, kernel_size=7, strides=(2, 2), padding='SAME')(input_tensor)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.ReLU()(out)

    out = keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='SAME')(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.ReLU()(out)

    out = keras.layers.MaxPool2D(pool_size=(2, 2))(out)

    for i in range(0, 3):
        if i == 0:
            multiplier = 2
            stride = 2
        else:
            multiplier = 1
            stride = 1
        out = stage(out, multiplier)

    out = keras.layers.MaxPool2D(pool_size=(2, 2))(out)

    for i in range(0, 6):
        if i == 0:
            multiplier = 2
            stride = 2
        else:
            multiplier = 1
            stride = 1
        out = stage(out, multiplier)

    out = keras.layers.MaxPool2D(pool_size=(2, 2))(out)

    for i in range(0, 5):
        if i == 0:
            multiplier = 2
            stride = 2
        else:
            multiplier = 1
            stride = 1
        out = stage(out, multiplier)
     
    out = keras.layers.MaxPool2D(pool_size=(2, 2))(out)

    for i in range(0, 4):
        if i == 0:
            multiplier = 2
            stride = 2
        else:
            multiplier = 1
            stride = 1
        out = stage(out, multiplier)

    out = keras.layers.MaxPool2D(pool_size=(2, 2))(out)

    for i in range(0, 3):
        if i == 0:
            multiplier = 2
            stride = 2
        else:
            multiplier = 1
            stride = 1
        out = stage(out, multiplier)
    new_size = out.shape[1] 
    avgpool = keras.layers.AveragePooling2D(pool_size=(new_size, new_size))(out)

    flatten = keras.layers.Flatten()(avgpool)
    dense1 = keras.layers.Dense(units=1024)(flatten)
    dense1 = keras.layers.Dropout(rate=dropout_rate)(dense1)
    dense2 = keras.layers.Dense(units=label_size, activation='softmax')(dense1)
    model = keras.models.Model(input_tensor, dense2)

    return model


if __name__ == "__main__":
    # C16
    data_path = f'/home/joyhyuk/dc/c16_cls/test/'
    model = "weights/cyberlogitec_best_1.hdf5"
    # model = "weights/c16_best.hdf5"

    # c18
    # data_path = f'/home/joyhyuk/dc/c18_cls/test/'
    # model = "weights/clt_dc_best_2.hdf5"
    # model = "weights/c18_best.hdf5"

    keras.backend.clear_session()
    model_best = keras.models.load_model(model)
    print('[model layers]\n', model_best.layers)
    plot_model(model_best, to_file='output/model.jpg')

    test_dataset = [0, 0, 0, 0]
    test_label = sorted(os.listdir(data_path))
    print(f'test label= {test_label}')
    print(f'before sort= {os.listdir(data_path)}')
    total_count = [0, 0, 0, 0]

    for label in test_label:
        print(f'\n * target test label: [{label}]')
        img_list = os.listdir(f'{data_path}{label}/') 
        count = [0, 0, 0, 0]
        test_dataset[test_label.index(label)] = len(img_list)

        for image in img_list:
            img = f'{data_path}{label}/{image}'
            img = cv2.imread(img)
            img = img / 255.0
            img = cv2.resize(img, (448, 448))
            img = np.expand_dims(img, axis=0)
            # print(img.shape)
            res = model_best.predict(img, batch_size=1, verbose=0)
            # print(f"result= {res[0]}")
            index = np.argmax(res)
            print(f' * test image: {image} | Result: {index}\n * Softmax: {res[0]}\n')
            count[int(index)] += 1
        temp_result = count[int(label)]
        print(f'result: {count} ---> {temp_result}')
        total_count[int(label)] += count[int(label)]
    print(f"\n ========== Total Result ==========\n"
            "Label 0: {total_count[0]} / {test_dataset[0]}\n"
            "Label 1: {total_count[1]} / {test_dataset[1]}\n"
            "Label 2: {total_count[2]} / {test_dataset[2]}\n"
            "Label 3: {total_count[3]} / {test_dataset[3]}")
    acc = round((sum(total_count) / sum(test_dataset)) * 100, 2)
    print(f'\nAcc: {acc} %')
