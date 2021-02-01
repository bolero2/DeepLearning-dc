import glob
import os
from sklearn.model_selection import train_test_split
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import *
from efficientnet.tfkeras import EfficientNetB4
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import utils
from keras import layers
import efficientnet.keras as efn
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

###############################################################
TrainDir = '/home/clt_dc/dataset/classification/cifar-animal/train'
EvalDir = '/home/clt_dc/dataset/classification/cifar-animal/eval'
train_counter = [len(os.listdir(filelist)) for filelist in [TrainDir + "/" + num_files + "/" for num_files in sorted(os.listdir(TrainDir))]]
eval_counter = [len(os.listdir(filelist)) for filelist in [EvalDir + "/" + num_files + "/" for num_files in sorted(os.listdir(EvalDir))]]
total_train = sum(train_counter)
total_eval = sum(eval_counter)
print(f'Total train: {total_train}, Total Eval: {total_eval}')
classes = sorted(os.listdir(TrainDir))
print(f'classes= {classes}')
label_size = len(classes)
batch_size = 8 
img_height = img_width = 300
###############################################################

train_datagen = ImageDataGenerator(
    rescale=1. / 255)
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    TrainDir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    EvalDir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

"""
model = Sequential()
model.add(efn.EfficientNetB3(weights="imagenet", include_top=False, pooling='avg'))
model.add(layers.Dense(label_size, activation="softmax"))
model = utils.multi_gpu_model(model, gpus=4)
model.compile(metrics=['acc'], loss='categorical_crossentropy', optimizer='adam')
"""

mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    model = Sequential()
    model.add(efn.EfficientNetB3(weights="imagenet", include_top=False, pooling='avg'))
    model.add(layers.Dense(label_size, activation="softmax"))
    model.compile(metrics=['acc'], loss='categorical_crossentropy', optimizer='adam')
# """

checkpointer = ModelCheckpoint(filepath='best.hdf5', verbose=1, save_best_only=True) # Save best weight file
csv_logger = CSVLogger('history.log')

history = model.fit(train_generator, validation_data=validation_generator, epochs=150, callbacks=[csv_logger, checkpointer])

model.save('last.hdf5') # Save last weight file
