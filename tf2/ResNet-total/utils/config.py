import tensorflow as tf
import os

########################################
# Select a defined ResNet-model
########################################
"""
Supported models
1. resnet34
2. resnet101
3. resnet152
"""
_model = "resnet34"

########################################
# Hyper-Parameter for default setting #
########################################
# Windows 10 Version
# TrainDir = "C:/dataset/OpenedDataset/cifar10/train/"
# EvalDir = "C:/dataset/OpenedDataset/cifar10/eval/"

# Linux Version
TrainDir = "C:/dataset/OpenedDataset/cifar10/train/"
EvalDir = "C:/dataset/OpenedDataset/cifar10/eval/"
TestImage = "./test.jpg"

total_train = 50000
total_eval = 10000
classes = os.listdir(TrainDir)
label_size = len(classes)

########################################
# Hyper-Parameter for Network #
########################################
image_size = 224
channel = 3
dropout_rate = 0.5

########################################
# Hyper-Parameter for training #
########################################
num_epochs = 20
batch_size = 16
train_with_validation = True
verbose = 1
lr = 0.00001
optimizer = 'Adam'
loss_function = tf.keras.losses.CategoricalCrossentropy()

########################################
# Hyper-Parameter for checkpoint #
########################################
ModelDir = "trained"

if not os.path.isdir(ModelDir):
    print("Making trained directory...")
    os.mkdir(ModelDir)
else:
    pass

dir_num = len(os.listdir(ModelDir))
ModelName = f"trained_{str(dir_num)}_{_model}-tf2"

ckpt_name_training = ModelDir + "/" + ModelName + "/" + ModelName + "_epoch_{epoch:04d}.ckpt"
save_ckpt_interval = 2

########################################
# checkpoint name to load for test
########################################
ckpt_name_testing = ModelDir + "/" + ModelName + "/" + ModelName + "_epoch_0020.ckpt"
