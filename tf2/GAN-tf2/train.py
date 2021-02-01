import os
import random
import numpy as np
import cv2
import tensorflow as tf
import keras
import collections
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


########################################
# Hyper-Parameter for default setting #
########################################
# Windows 10 Version
# TrainDir = "C:/dataset/OpenedDataset/cifar10/train/"
# EvalDir = "C:/dataset/OpenedDataset/cifar10/eval/"

# Linux Version
# TrainDir = "/home/Cyberlogitec/dataset/classification/cifar5/train/"
TrainDir = "/home/clt_dc/dataset/classification/cifar-animal/train/"
# EvalDir = "/home/Cyberlogitec/dataset/classification/cifar5/eval/"
EvalDir = "/home/clt_dc/dataset/classification/cifar-animal/eval/"
TestImage = "./test.jpg"

train_counter = [len(os.listdir(filelist)) for filelist in [TrainDir + num_files + "/" for num_files in sorted(os.listdir(TrainDir))]] 
eval_counter = [len(os.listdir(filelist)) for filelist in [EvalDir + num_files + "/" for num_files in sorted(os.listdir(EvalDir))]]
total_train = sum(train_counter)
total_eval = sum(eval_counter)
print(f'Total train: {total_train}\nTotal Eval: {total_eval}')
classes = sorted(os.listdir(TrainDir))
print(f'classes= {classes}')
label_size = len(classes)

########################################
# Hyper-Parameter for Network #
########################################
image_size = 224 
channel = 3
dropout_rate = 0.5
######################################## # Hyper-Parameter for training #
########################################
num_epochs = 30 
batch_size = 64 
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


def generator_model():
    input_data = tf.keras.Input(shape=(image_size, image_size, channel))

    out = tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(image_size, image_size, channel))(input_data)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)

    out = tf.keras.layers.Reshape((7, 7, 256))(out)

    out = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)

    out = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)

    output = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=Falsem, activation='tanh')(out)

    model = tf.keras.Model(input_data, output)

    return model


def discriminator_model():
    input_data = tf.keras.Input(shape=(image_size, image_size, channel))

    out = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
            input_shape=[image_size, image_size, channel]))(input_data)

    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = tf.keras.layers.Flatten()(out)
    output = tf.keras.layers.Dense(1)(out)

    model = tf.keras.Model(input_data, output)
    
    return model


if __name__ == "__main__":
    filenames_train, filenames_eval = read_path()

    train_images, train_labels = load_image(filenames_train, type='train')
    eval_images, eval_labels = load_image(filenames_eval, type='eval')

    generator = generator_model()

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    plt.imshow(generated_image[0, :, :, 0], cmap='gray')

    discriminator = discriminator_model()
    decision = discriminator(generated_image)
    print(decision)


    
    # Multi-GPU Model
    from keras.utils import multi_gpu_model

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = network(training=True)
        model.summary()
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    """ 
    # Single-GPU Model
    model = network(training=True)
    model.summary()
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
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
