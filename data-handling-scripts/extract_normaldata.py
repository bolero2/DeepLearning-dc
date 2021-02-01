import os
import shutil as sh
import glob
import random

path = "/home/yb/dc/AI_C16/normal/"

image_list = os.listdir(path)
random.shuffle(image_list)

for i in range(0, len(image_list)):
    if i < 580:
        sh.copy(path + image_list[i], '/home/yb/dc/c16_normal/')
    else:
        break
