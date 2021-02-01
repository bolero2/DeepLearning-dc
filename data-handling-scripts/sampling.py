import os
import random

path = "/home/Cyberlogitec/dc2/c16_rid_aug_smp/train/0/"
filelist = os.listdir(path)

random.shuffle(filelist)

for i in range(0, len(filelist)):
    if i < 4000:
        continue
    else:
        os.remove(path + filelist[i])
