import os
import random
import glob
import shutil
import cv2
import numpy as np
import copy


def zoom_out(img, out_rate, lines):
    row, col, ch = img.shape
    print(f'Image width: {col}, Image height: {row}, Image channel: {ch}')

    bg = np.zeros((row, col, ch), dtype='uint8')
    scaled_w = int(col / out_rate)
    scaled_h = int(row / out_rate)
    new_img = cv2.resize(img, (scaled_w, scaled_h))
    # cv2.imshow("new_img", new_img)
    # cv2.waitKey(0)
    scaled_center_x = int(scaled_w)
    scaled_center_y = int(scaled_h)
    bg[int(scaled_center_y - (scaled_h / 2)):int(scaled_center_y + (scaled_h / 2)), int(scaled_center_x - (scaled_w / 2)):int(scaled_center_x + (scaled_w / 2)), :] = copy.deepcopy(new_img[:, :, :])
    # bg[int(center_x - (width / 2)):int(center_x + (width / 2)), int(center_y - (height / 2)):int(center_y + (height / 2)), :] = copy.deepcopy(new_img[:, :, :])

    new_coord = list()

    for line in lines:
        if line[-1] == "\n":
            line = line[:-1]
        line = line.split(' ')
        center_x = line[1]
        center_y = line[2]
        width = line[3]
        height = line[4]
        print(center_x)
        distance_x = abs(float(center_x) - 0.5)
        distance_y = abs(float(center_y) - 0.5)
        if float(center_x) > 0.5:
            new_x = 0.5 + (distance_x / out_rate)
        else:
            new_x = 0.5 - (distance_x / out_rate)
        if float(center_y) > 0.5:
            new_y = 0.5 + (distance_y / out_rate)
        else:
            new_y = 0.5 - (distance_y / out_rate)
        new_w = float(width) / 2
        new_h = float(height) / 2

        new_coord.append(f'0 {new_x} {new_y} {new_w} {new_h}\n')

    return bg, new_coord


if __name__ == "__main__":
    path = "/home/yb/dc/yolo_c16/train/"
    new_path = "/home/yb/dc/yolo_c16/train/"
    os.chdir(path)
    imglist = [x for x in glob.glob('*.jpg')]
    random.shuffle(imglist)
    limit = int(len(imglist) / 3)
    count = 0

    for image in imglist:
        if count < limit:
            annotfile = image[:-3] + "txt"
            f = open(path + annotfile, 'r')
            lines = f.readlines()
            f.close()

            img = cv2.imread(path + image)
            zout_img, zout_coord = zoom_out(img, 2, lines)
            print(lines)
            print(zout_coord)
            cv2.imwrite(new_path + image[:-4] + "_zout2.jpg", zout_img)
            f2 = open(new_path + image[:-4] + "_zout2.txt", 'w')
            f2.writelines(zout_coord)
            f2.close()
            # cv2.imshow("zoom_out", zout_img)
            # cv2.imshow("Original", img)

            os.remove(path + image)
            os.remove(path + annotfile)
            count += 1
        else:
            break
