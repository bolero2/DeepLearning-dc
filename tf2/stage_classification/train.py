import os
import random
import numpy as np
import cv2
import tensorflow as tf
import keras
import collections
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight


########################################
# Hyper-Parameter for default setting #
########################################
# Windows 10 Version
# TrainDir = "C:/dataset/OpenedDataset/cifar10/train/"
# EvalDir = "C:/dataset/OpenedDataset/cifar10/eval/"

# Linux Version
# TrainDir = "/home/Cyberlogitec/dataset/classification/cifar5/train/"
TrainDir = "/home/Cyberlogitec/dc2/c16_rid_aug/train/"
# EvalDir = "/home/Cyberlogitec/dataset/classification/cifar5/eval/"
EvalDir = "/home/Cyberlogitec/dc2/c16_rid_aug/eval/"
TestImage = "./test.jpg"

train_counter = [len(os.listdir(filelist)) for filelist in [TrainDir + num_files + "/" for num_files in sorted(os.listdir(TrainDir))]] 
eval_counter = [len(os.listdir(filelist)) for filelist in [EvalDir + num_files + "/" for num_files in sorted(os.listdir(EvalDir))]]
total_train = sum(train_counter)
total_eval = sum(eval_counter)
print(f'Total train: {total_train}, Total Eval: {total_eval}')
classes = sorted(os.listdir(TrainDir))
print(f'classes= {classes}')
label_size = len(classes)

########################################
# Hyper-Parameter for Network #
########################################
image_size = 448 
channel = 3
dropout_rate = 0.5

########################################
# Hyper-Parameter for training #
########################################
num_epochs = 300 
batch_size = 8 
train_with_validation = True
verbose = 1
lr = 0.00001
optimizer = keras.optimizers.Adam(lr=lr)
loss_function = keras.losses.CategoricalCrossentropy()

########################################
# Hyper-Parameter for checkpoint #
########################################
pwd = os.getcwd()
ModelDir = "trained"

if not os.path.isdir(f'{pwd}/{ModelDir}'):
    print("Making trained directory...")
    os.mkdir(f'{pwd}/{ModelDir}')
else:
    pass

dir_num = len(os.listdir(f'{pwd}/{ModelDir}'))
ModelName = f"{ModelDir}{str(dir_num)}"

# saved_name = f'{pwd}/{ModelDir}/{ModelName}/{ModelName}_epoch_{epoch:04d}.hdf5'
saved_name = f'{pwd}/{ModelDir}/{ModelName}_best.hdf5'
# save_ckpt_interval = 2

########################################
# checkpoint name to load for test
########################################
# ckpt_name_testing = ModelDir + "/" + ModelName + "/" + ModelName + "_epoch_0020.ckpt"


def read_path():
    print("Reading path of dataset ... Start")
    train_buffer = list()
    eval_buffer = list()
    for class_name in classes:
        filelist = os.listdir(TrainDir + str(class_name))
        for j in range(len(filelist)):
            train_buffer.append([TrainDir + str(class_name) + '/' + filelist[j], classes.index(class_name)])
    random.shuffle(train_buffer)

    for class_name in classes:
        filelist = os.listdir(EvalDir + str(class_name))
        for j in range(len(filelist)):
            eval_buffer.append([EvalDir + str(class_name) + '/' + filelist[j], classes.index(class_name)])
    random.shuffle(eval_buffer)
    print("Reading path of dataset ... End")

    return np.array(train_buffer), np.array(eval_buffer)


def load_image(filenames, type):
    print("Loading dataset in Array ... Start")
    images = filenames[:, 0]
    labels = filenames[:, 1]

    total_size = 0

    if type == 'train':
        total_size = total_train
    elif type == 'eval':
        total_size = total_eval

    image_buffer = np.zeros(shape=(total_size, image_size, image_size, channel), dtype=np.float32)
    label_buffer = np.zeros(shape=(total_size, label_size), dtype=np.uint8)
    # images / 255.0
    # labels.astype('float32') or ('uint8'), don't care about label type

    for i in range(filenames.shape[0]):
        image_buffer[i, :, :, :] = cv2.resize(cv2.imread(images[i]), (image_size, image_size)) / 255.0
        label_buffer[i, int(labels[i])] = 1     # one-hot encoding

    print("Loading dataset in Array ... End")

    return image_buffer, label_buffer


def stage(tensor, multiplier, stride=1):
    residual = tensor
    in_channels = residual.shape[-1]

    origin = keras.layers.Conv2D(filters=int(in_channels) * multiplier, kernel_size=1, strides=(stride, stride), padding='SAME', 
            kernel_initializer=keras.initializers.he_normal(seed=None))(tensor)
    origin = keras.layers.BatchNormalization()(origin)
    origin = keras.layers.ReLU()(origin)

    residual = keras.layers.Conv2D(filters=int(in_channels) * multiplier, kernel_size=3, strides=(stride, stride), padding='SAME', 
            kernel_initializer=keras.initializers.he_normal(seed=None))(residual)
    residual = keras.layers.BatchNormalization()(residual)
    residual = keras.layers.ReLU()(residual)

    elem1 = keras.layers.DepthwiseConv2D(kernel_size=3, strides=(stride, stride), padding='SAME', depth_multiplier=multiplier, 
            kernel_initializer=keras.initializers.he_normal(seed=None))(tensor)
    elem2 = keras.layers.DepthwiseConv2D(kernel_size=3, strides=(stride, stride), padding='SAME', depth_multiplier=multiplier, dilation_rate=2, 
            kernel_initializer=keras.initializers.he_normal(seed=None))(tensor)
    elem3 = keras.layers.DepthwiseConv2D(kernel_size=3, strides=(stride, stride), padding='SAME', depth_multiplier=multiplier, dilation_rate=4, 
            kernel_initializer=keras.initializers.he_normal(seed=None))(tensor)
    elem4 = keras.layers.DepthwiseConv2D(kernel_size=3, strides=(stride, stride), padding='SAME', depth_multiplier=multiplier, dilation_rate=8,
            kernel_initializer=keras.initializers.he_normal(seed=None))(tensor)

    elem1 = keras.layers.BatchNormalization()(elem1)
    elem2 = keras.layers.BatchNormalization()(elem2)
    elem3 = keras.layers.BatchNormalization()(elem3)
    elem4 = keras.layers.BatchNormalization()(elem4)

    elem1 = keras.layers.ReLU()(elem1)
    elem2 = keras.layers.ReLU()(elem2)
    elem3 = keras.layers.ReLU()(elem3)
    elem4 = keras.layers.ReLU()(elem4)

    # print(origin.shape, residual.shape, elem1.shape, elem2.shape, elem3.shape, elem4.shape)
    total = keras.layers.Add()([origin, residual, elem1, elem2, elem3, elem4])

    return total


def network():
    multiplier = 0

    input_tensor = keras.layers.Input(shape=(image_size, image_size, channel))
    out = keras.layers.Conv2D(filters=32, kernel_size=7, strides=(2, 2), padding='SAME', 
            kernel_initializer=keras.initializers.he_normal(seed=None))(input_tensor)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.ReLU()(out)

    out = keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='SAME', 
            kernel_initializer=keras.initializers.he_normal(seed=None))(out)
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
    dense1 = keras.layers.Dense(units=1024,
            kernel_initializer=keras.initializers.he_normal(seed=None))(flatten)
    dense1 = keras.layers.Dropout(rate=dropout_rate)(dense1)
    dense2 = keras.layers.Dense(units=label_size, activation='softmax',
            kernel_initializer=keras.initializers.he_normal(seed=None))(dense1)
    model = keras.models.Model(input_tensor, dense2)

    return model


if __name__ == "__main__":
    filenames_train, filenames_eval = read_path()

    train_images, train_labels = load_image(filenames_train, type='train')
    eval_images, eval_labels = load_image(filenames_eval, type='eval')
    """
    train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            rotation_range=90,
            brightness_range=[0.75, 1.0],
            zoom_range=[0.2, 1.0],
            horizontal_flip=True,
            vertical_flip=True)

    valid_datagen = ImageDataGenerator(
            rescale=1. /255)

    train_generator = train_datagen.flow_from_directory(
            TrainDir[:-1], 
            target_size=(image_size, image_size),
            batch_size=batch_size, 
            class_mode='categorical')

    valid_generator = valid_datagen.flow_from_directory(
            EvalDir[:-1],
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical')
    """

    # Multi-GPU Model
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = network()
        model.summary()
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    # Single-GPU Model
    """
    model = network()
    # model = multi_gpu_model(model, gpus=4)

    model.summary()
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    """
    
    # Training with ImageDataGenerator
    """
    history = model.fit_generator(
            train_generator,
            steps_per_epoch = total_train // batch_size,
            validation_data=valid_generator,
            validation_steps = total_eval // batch_size,
            epochs=num_epochs,
            verbose=1,
            callbacks=[cb_ckpt, cb_logger])
    """
    cb_ckpt = keras.callbacks.ModelCheckpoint(filepath=saved_name, verbose=1, save_best_only=True)
    cb_logger = keras.callbacks.CSVLogger('history.log')

    # Class_weights params for Imbalanced Dataset
    class_weights = list()
    for count in train_counter:
        value = (1 / count) * (total_train) / label_size
        class_weights.append(value)

    d_class_weights = dict(enumerate(class_weights))
    print(f'd_class_weights= {d_class_weights}')

    history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(eval_images, eval_labels), class_weight=d_class_weights, callbacks=[cb_ckpt, cb_logger])

    model.save(saved_name)
    print(" *** END ***")
