import glob
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
n_classes = 4 
img_size = 299
train_data_dir = '/home/aigpuserver/img_cls/c_data4/train'
validation_data_dir = '/home/aigpuserver/img_cls/c_data4/valid'
nb_train_samples = 2461
nb_validation_samples = 274
batch_size = 16
###############################################################

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model = Sequential()
model.add(efn.EfficientNetB4(weights="imagenet", include_top=False, pooling='avg'))
model.add(layers.Dense(n_classes, activation="softmax"))
model = utils.multi_gpu_model(model, gpus=4)
model.compile(metrics=['acc'], loss='categorical_crossentropy', optimizer='adam')

checkpointer = ModelCheckpoint(filepath='best.hdf5', verbose=1, save_best_only=True) # Save best weight file
csv_logger = CSVLogger('history.log')

history = model.fit(train_generator, validation_data=validation_generator, epochs=150, callbacks=[csv_logger, checkpointer])

model.save('last.hdf5') # Save last weight file
