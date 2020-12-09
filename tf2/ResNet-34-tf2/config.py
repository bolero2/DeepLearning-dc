import tensorflow as tf
import os

########################################
# Hyper-Parameter for default setting #
########################################
# Windows 10 Version
# TrainDir = "C:/dataset/OpenedDataset/cifar10/train/"
# EvalDir = "C:/dataset/OpenedDataset/cifar10/eval/"

# Linux Version
TrainDir = "/home/clt_dc/dataset/classification/cifar100/train/"
EvalDir = "/home/clt_dc/dataset/classification/cifar100/test/"
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
num_epochs = 50
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
ModelName = "trained_" + str(dir_num) + "_resnet34-tf2"

ckpt_name_training = ModelDir + "/" + ModelName + "/" + ModelName + "_epoch_{epoch:04d}.ckpt"
save_ckpt_interval = 2

########################################
# checkpoint name to load for test
########################################
ckpt_name_testing = ModelDir + "/" + ModelName + "/" + ModelName + "_epoch_0020.ckpt"
