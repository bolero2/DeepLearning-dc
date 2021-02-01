import glob
import os

path = '/home/yb/dc/yolo_c16/new_endo/'

filenames = [x + '\n' for x in glob.glob(f'{path}*.jpg')]
f = open('train.txt', 'w')
f.writelines(filenames)

