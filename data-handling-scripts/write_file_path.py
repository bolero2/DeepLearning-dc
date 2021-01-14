import glob
import os

path = '/home/yb/dc/yolo_c16/test/'

filenames = [x + '\n' for x in glob.glob(f'{path}*.jpg')]
f = open('test.txt', 'w')
f.writelines(filenames)

