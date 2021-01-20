import glob
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import argparse
import keras
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


weight = "best.hdf5"
classes= ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
# classes = ['horse', 'bird', 'dog', 'deer', 'frog', 'cat']

model = keras.models.load_model(weight)
video = "video/animals.mp4"

cap = cv2.VideoCapture(video)
ret, frame = cap.read()

while ret: 
    cv2.imshow("aa", frame)
    frame = cv2.resize(frame, (300, 300))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    cv2.waitKey(1)

    res = model.predict(frame)
    index = np.argmax(res, axis=1)

    print(f'label= {index}, {classes[int(index)]}')

    ret, frame = cap.read()
