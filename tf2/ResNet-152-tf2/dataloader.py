import os
import random
import numpy as np
import cv2
import config as cfg


def read_path():
    print("Reading path of dataset ... Start")
    train_buffer = list()
    eval_buffer = list()
    for class_name in cfg.classes:
        filelist = os.listdir(cfg.TrainDir + str(class_name))
        for j in range(len(filelist)):
            train_buffer.append([cfg.TrainDir + str(class_name) + '/' + filelist[j], cfg.classes.index(class_name)])
    random.shuffle(train_buffer)

    for class_name in cfg.classes:
        filelist = os.listdir(cfg.EvalDir + str(class_name))
        for j in range(len(filelist)):
            eval_buffer.append([cfg.EvalDir + str(class_name) + '/' + filelist[j], cfg.classes.index(class_name)])
    random.shuffle(eval_buffer)
    print("Reading path of dataset ... End")

    return np.array(train_buffer), np.array(eval_buffer)


def load_image(filenames, type):
    print("Loading dataset in Array ... Start")
    images = filenames[:, 0]
    labels = filenames[:, 1]

    total_size = 0

    if type == 'train':
        total_size = cfg.total_train
    elif type == 'eval':
        total_size = cfg.total_eval

    image_buffer = np.zeros(shape=(total_size, cfg.image_size, cfg.image_size, cfg.channel), dtype=np.float32)
    label_buffer = np.zeros(shape=(total_size, cfg.label_size), dtype=np.uint8)
    # images / 255.0
    # labels.astype('float32') or ('uint8'), don't care about label type

    for i in range(filenames.shape[0]):
        image_buffer[i, :, :, :] = cv2.resize(cv2.imread(images[i]), (cfg.image_size, cfg.image_size)) / 255.0
        label_buffer[i, int(labels[i])] = 1     # one-hot encoding

    print("Loading dataset in Array ... End")

    return image_buffer, label_buffer
