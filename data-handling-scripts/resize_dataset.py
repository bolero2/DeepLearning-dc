import os
import glob
import cv2


path = '/home/clt_dc/dataset/yolo_c16/newendo/'
os.chdir(path)
imglist = [x for x in glob.glob('*.jpg')]

for imgname in imglist:
    img = cv2.imread(f'{path}{imgname}')
    print(img.shape)
