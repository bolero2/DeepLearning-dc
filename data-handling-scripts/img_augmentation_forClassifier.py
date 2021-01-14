"""
1. Crop( _crop1 ~ _crop4 )
2. Random RGB Scaling( _rgb )
3. weak Gaussian Blur( _blur )
4. rotation( _rot1 ~ _rot2 )

"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import random


def filtering(img, number):
    if number == 0:
        return cv2.GaussianBlur(img, ksize=(13, 13), sigmaX=0)
    if number == 1:
        return cv2.GaussianBlur(img, ksize=(25, 25), sigmaX=0)
    if number == 2:
        return adjust_gamma(img, random.randint(11, 16) / 10)
    if number == 3:
        return adjust_gamma(img, random.randint(6, 9) / 10)
    if number == 4:
        return salt_and_pepper(img, p=random.randint(0, 7) / 100)
    if number == 5:
        return salt_and_pepper(img, p=random.randint(8, 15) / 100)


# def dropout(image, count, rate_row, rate_col):
#     output = np.zeros(image.shape, np.uint8)
#     row, col, ch = image.shape
#     width = int(row * rate_row)
#     height = int(col * rate_col)
#     print(width, height)
#     for i in range(count):
#         random.seed(random.random())
#         rand_x = random.randint(1, row)
#         rand_y = random.randint(1, col)
#
#         xmin = int(rand_x - (width / 2))
#         ymin = int(rand_y - (height / 2))
#         xmax = int(rand_x + (width / 2))
#         ymax = int(rand_y + (height / 2))
#         print(xmin, ymin, xmax, ymax)
#
#         for r in range(image.shape[0]):
#             for c in range(image.shape[1]):
#                 if r >= xmin and r <= xmax and c >= ymin and c <= ymax:
#                     output[r, c, :] = 0
#                 else:
#                     output[r, c, :] = image[r, c, :]
#     return output


def salt_and_pepper(image, p):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - p
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < p:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def augment_for_classification(image_path, new_image_path, crop_rate):
    image_list = os.listdir(image_path)

    for num in range(0, len(image_list)):
        img = cv2.imread(image_path + image_list[num])    # Original Image
        # img = cv2.resize(img, (448, 448))
        col, row, ch = img.shape

        img_rot1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_rot2 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_flip1 = cv2.flip(img, -1)
        img_flip2 = cv2.flip(img, 1)
        img_flip3 = cv2.flip(img, 0)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_rot1 = filtering(img_rot1, random.randint(0, 5))
        cv2.imwrite(new_image_path + image_list[num][:-4] + "_rot_1" + ".jpg", img_rot1)

        img_rot2 = filtering(img_rot2, random.randint(0, 5))
        cv2.imwrite(new_image_path + image_list[num][:-4] + "_rot_2" + ".jpg", img_rot2)

        img_flip1 = filtering(img_flip1, random.randint(0, 5))
        cv2.imwrite(new_image_path + image_list[num][:-4] + "_flip_1" + ".jpg", img_flip1)

        img_flip2 = filtering(img_flip2, random.randint(0, 5))
        cv2.imwrite(new_image_path + image_list[num][:-4] + "_flip_2" + ".jpg", img_flip2)

        img_flip3 = filtering(img_flip3, random.randint(0, 5))
        cv2.imwrite(new_image_path + image_list[num][:-4] + "_flip_3" + ".jpg", img_flip3)

        if crop_rate != 0:
            img_crop1 = copy.deepcopy(img[0:int(col * (1 - crop_rate)), 0:int(row * (1 - crop_rate)), :])
            img_crop2 = copy.deepcopy(img[0:int(col * (1 - crop_rate)), int(row * crop_rate):row, :])
            img_crop3 = copy.deepcopy(img[int(col * crop_rate):col, 0:int(row * (1 - crop_rate)), :])
            img_crop4 = copy.deepcopy(img[int(col * crop_rate):col, int(row * crop_rate):row, :])

            img_crop1 = filtering(img_crop1, random.randint(0, 5))
            cv2.imwrite(new_image_path + image_list[num][:-4] + "_crop_1" + ".jpg", img_crop1)

            img_crop2 = filtering(img_crop2, random.randint(0, 5))
            cv2.imwrite(new_image_path + image_list[num][:-4] + "_crop_2" + ".jpg", img_crop2)

            img_crop3 = filtering(img_crop3, random.randint(0, 5))
            cv2.imwrite(new_image_path + image_list[num][:-4] + "_crop_3" + ".jpg", img_crop3)

            img_crop4 = filtering(img_crop4, random.randint(0, 5))
            cv2.imwrite(new_image_path + image_list[num][:-4] + "_crop_4" + ".jpg", img_crop4)


if __name__ == "__main__":
    image_path = "C:\\dataset\\MyDataset\\classifier_colon\\1\\"
    new_image_path = "C:\\dataset\\MyDataset\\classifier_colon\\new1\\"
    crop_rate = 0.15

    augment_for_classification(image_path=image_path, 
                               new_image_path=new_image_path,
                               crop_rate=crop_rate)
    exit(0)
