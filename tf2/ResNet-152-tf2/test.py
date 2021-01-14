import os
import random
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model


if __name__ == "__main__":
    # C16
    # data_path = f'/home/joyhyuk/dc/c16_cls/test/'
    # model = "weights/c16_image_best.hdf5"
    # model = "weights/trained1_best.hdf5"

    # c18
    data_path = f'/home/joyhyuk/dc/c18_rid/test/'
    # model = "weights/c18_image_best.hdf5"
    model = "c18_rid_test1.hdf5"

    keras.backend.clear_session()
    model_best = keras.models.load_model(model)
    print('[model layers]\n', model_best.layers)
    plot_model(model_best, to_file='output/model.jpg')

    test_dataset = [0, 0, 0, 0]
    test_label = sorted(os.listdir(data_path))
    print(f'test label= {test_label}')
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
            f"Label 0: {total_count[0]} / {test_dataset[0]}\n"
            f"Label 1: {total_count[1]} / {test_dataset[1]}\n"
            f"Label 2: {total_count[2]} / {test_dataset[2]}\n"
            f"Label 3: {total_count[3]} / {test_dataset[3]}")
    acc = round((sum(total_count) / sum(test_dataset)) * 100, 2)
    print(f'\nAcc: {acc} %')
