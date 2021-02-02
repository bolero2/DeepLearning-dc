import os
import glob
import cv2


path = '/home/clt_dc/dataset/detection/idc_cancer/yolo_c16/new_endo/new_endo_resized/'

os.chdir(path)
imglist = sorted([x for x in glob.glob('*.jpg')])

for imgname in imglist:
    img = cv2.imread(f'{path}{imgname}')

    img2 = cv2.resize(img, (640, 480))
    cv2.imwrite(f'{path}{imgname}', img2)
    print(img.shape, img2.shape)
