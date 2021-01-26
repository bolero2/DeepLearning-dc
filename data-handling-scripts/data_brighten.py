import cv2
import numpy as np
import os
import glob
import shutil


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def brightening(path, save_path, gamma):
    os.chdir(path)
    original_filelist = [x for x in glob.glob('*.jpg')]

    for image in original_filelist:
        img = cv2.imread(path + image)
        img_bri = adjust_gamma(img, gamma)
        cv2.imshow("gamma", img_bri)
        cv2.imshow("original", img)
        key = cv2.waitKey(0)

        if key == ord('q'):
            exit(0)
        # cv2.imwrite(save_path + image[:-4] + "_brighten.jpg", img_bri)
        # shutil.copy(path + image[:-3] + "txt", save_path + image[:-4] + "_brighten.txt")
        
        print(f'{path + image} ===> {save_path + image[:-4] + "_brighten.jpg"}')


if __name__ == "__main__":
    path = "/home/bolero/.dc/dl/dataset/train_original/"
    save_path = "/home/bolero/.dc/dl/dataset/train_original/"
    gamma = 2.0

    brightening(path=path, save_path=save_path, gamma=gamma)

    exit(0)
