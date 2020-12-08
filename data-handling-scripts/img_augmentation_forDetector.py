"""
******** Annotation type = yolo type(ccwh, relat) ********

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


def draw_center(image, bbox):
    col, row, ch = image.shape
    pt = (int(row * bbox[1]), int(col * bbox[2]))

    return cv2.circle(image, center=pt, radius=1, color=(255, 0, 0), thickness=5)


def draw_bbox(image, bbox):
    col, row, ch = image.shape

    pt1 = (int(row * bbox[1] - (row * bbox[3]) / 2), int(col * bbox[2] - (col * bbox[4]) / 2))
    pt2 = (int(row * bbox[1] + (row * bbox[3]) / 2), int(col * bbox[2] + (col * bbox[4]) / 2))

    return cv2.rectangle(image, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=5)


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


def convert_coord(coord, name='none', original_image=None, converted_image=None):
    """
    :param coord: coord = [label_index, center_x, center_y, width, height]
    :param name:
    1) rot_90
    2) rev_rot_90
    3) rev_flip_1 (flip code = -1)
    4) flip_1 (flip code = 1)
    5) flip_0 (flip code = 0)
    6) crop_1 (top left)
    7) crop_2 (top right)
    8) crop_3 (bottom left)
    9) crop_4 (bottom right)
    :param original_image: This parameter is used for Crop.
    :param converted_image: This parameter is used for Crop.

    :return: coord = [label_index, center_x, center_y, width, height]
    """

    if name == 'rot_90':
        return [coord[0], 1 - coord[2], coord[1], coord[4], coord[3]]

    if name == 'rev_rot_90':
        return [coord[0], coord[2], 1 - coord[1], coord[4], coord[3]]

    if name == 'rev_flip_1':
        return [coord[0], 1 - coord[1], 1 - coord[2], coord[3], coord[4]]

    if name == 'flip_1':
        return [coord[0], 1 - coord[1], coord[2], coord[3], coord[4]]

    if name == 'flip_0':
        return [coord[0], coord[1], 1 - coord[2], coord[3], coord[4]]

    if name == 'crop_1':
        col, row, ch = original_image.shape
        new_col, new_row, new_ch = converted_image.shape

        center_x = coord[1] * col / new_col
        center_y = coord[2] * row / new_row
        width = coord[3] * col / new_col
        height = coord[4] * row / new_row

        if center_x + (width / 2) > 1:
            loss = (center_x + (width / 2)) - 1
            width = width - loss
            center_x = center_x - loss / 2

        if center_y + (height / 2) > 1:
            loss = (center_y + (height / 2)) - 1
            height = height - loss
            center_y = center_y - loss / 2
        return [coord[0], center_x, center_y, width, height]

    if name == 'crop_2':
        col, row, ch = original_image.shape
        new_col, new_row, new_ch = converted_image.shape

        gap = [col - new_col, row - new_row]
        center_x = (coord[1] * col - gap[0]) / new_col
        center_y = coord[2] * row / new_row
        width = coord[3] * col / new_col
        height = coord[4] * row / new_row

        if center_x - (width / 2) < 0:
            loss = (width / 2) - center_x
            width = width - loss
            center_x = center_x + loss / 2

        if center_y + (height / 2) > 1:
            loss = (center_y + (height / 2)) - 1
            height = height - loss
            center_y = center_y - loss / 2
        return [coord[0], center_x, center_y, width, height]

    if name == 'crop_3':
        col, row, ch = original_image.shape
        new_col, new_row, new_ch = converted_image.shape

        gap = [col - new_col, row - new_row]
        center_x = coord[1] * col / new_col
        center_y = (coord[2] * row - gap[1]) / new_row
        width = coord[3] * col / new_col
        height = coord[4] * row / new_row

        if center_x + (width / 2) > 1:
            loss = (center_x + (width / 2)) - 1
            width = width - loss
            center_x = center_x - loss / 2

        if center_y - (height / 2) < 0:
            loss = (height / 2) - center_y
            height = height - loss
            center_y = center_y + loss / 2
        return [coord[0], center_x, center_y, width, height]

    if name == 'crop_4':
        col, row, ch = original_image.shape
        new_col, new_row, new_ch = converted_image.shape

        gap = [col - new_col, row - new_row]
        center_x = (coord[1] * col - gap[0]) / new_col
        center_y = (coord[2] * row - gap[1]) / new_row
        width = coord[3] * col / new_col
        height = coord[4] * row / new_row

        if center_x - (width / 2) < 0:
            loss = (width / 2) - center_x
            width = width - loss
            center_x = center_x + loss / 2

        if center_y - (height / 2) < 0:
            loss = (height / 2) - center_y
            height = height - loss
            center_y = center_y + loss / 2
        return [coord[0], center_x, center_y, width, height]


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


def augment_for_detection(image_path, new_image_path, label_path, new_label_path,
                          crop_rate=0.15, save_coord='ccwh', save_type='relat'):
    """
    :param image_path: Original Image path
    :param new_image_path: Image path for saving augmented Images
    :param label_path: Original Label path
    :param new_label_path: Path for saving label coordinates of augmented image

    :param crop_rate: Crop-rate

    [How-to-Crop Images?]
    pivot = [top-left, top-right, bottom-left, bottom-right]
    IF) crop_rate is 0.15
    +--------------------+
    |                | c |
    |  Remain Sector | r |
    |................| o |
    |   crop   crop    p |
    +--------------------+

    [Augmentation Method]
    1) Geometric transformation
        - Cropping 1 ~ 4
        - Rotation 1 ~ 2
        - Flip 1 ~ 3
    2) Filtering
        - Gaussian Blurring(weak, strong)
        - Gamma Transformation(=RGB Scaling) (brightly, darkly)
        - Salt & Pepper noise(weak, strong)

    :param save_coord:
    1) ccwh = center_x, center_y, width, height
    2) xywh = xmin, ymin, width, height
    3) xyrb = xmin, ymin, xmax, ymax(Right, Bottom)

    :param save_type:
    1) relat = relative coordinate
        ex.
            center_x = 0.2
            center_y = 0.3
            width = 0.55
            height = 0.12
    2) abs = absolute coordinate
        ex.
            center_x = 300
            center_y = 350
            width = 120
            height = 180

    :return:
    Save...
    1. Augmented Image
    2. text file with label coordinate information for Augmented Image
    """
    # cv2.namedWindow("test")
    image_list = os.listdir(image_path)
    label_list = os.listdir(label_path)

    for num in range(0, len(image_list)):
        img = cv2.imread(image_path + image_list[num])    # Original Image
        col, row, ch = img.shape

        img_rot1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_rot2 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_flip1 = cv2.flip(img, -1)
        img_flip2 = cv2.flip(img, 1)
        img_flip3 = cv2.flip(img, 0)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Annotation Setting
        annot_file = label_path + image_list[num][:-4] + ".txt"
        lines = open(annot_file, 'r').readlines()
        coord_list = list()
        temp_list = list()
        for line in lines:
            temp = line[:-1].split(' ')
            for i in range(0, len(temp)):
                if i == 0:
                    temp_list.append(int(temp[i]))
                else:
                    temp_list.append(float(temp[i]))
            coord_list.append(temp_list)
            temp_list = list()
        print("\nOriginal coordinate list", coord_list)

        # Original Image
        temp_list = coord_list
        for count in temp_list:
            coord_list = list()
            coord_list.append([count[0], count[1], count[2], count[3], count[4]])
            print("original_coordinate:", coord_list[0])
            # img = draw_bbox(img, coord_list[0])
            # img = draw_center(img, coord_list[0])
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        coord_list = temp_list

        save_annot_list = list()
        temp_sentence = ''

        # rotation -> 90 clockwise
        temp_list = coord_list
        for count in temp_list:
            coord_list = list()
            coord_list.append(convert_coord(count, name='rot_90'))
            print("90 clockwise coordinate:", coord_list[0])
            # img_rot1 = draw_bbox(img_rot1, coord_list[0])
            # img_rot1 = draw_center(img_rot1, coord_list[0])
            for coindex in range(0, len(coord_list[0])):
                if coindex != 4:
                    temp_sentence = temp_sentence + str(coord_list[0][coindex]) + ' '
                else:
                    temp_sentence = temp_sentence + str(coord_list[0][coindex]) + '\n'
            save_annot_list.append(temp_sentence)
            temp_sentence = ''
        img_rot1 = filtering(img_rot1, random.randint(0, 5))
        cv2.imwrite(new_image_path + image_list[num][:-4] + "_rot_1" + ".jpg", img_rot1)
        f2 = open(new_label_path + image_list[num][:-4] + "_rot_1" + ".txt", 'w')
        for saveline in save_annot_list:
            f2.write(saveline)
        save_annot_list = list()
        temp_sentence = ''
        # cv2.imshow("img_rot1", img_rot1)
        # cv2.waitKey(0)
        coord_list = temp_list

        # rotation -> 90 counter-clockwise
        temp_list = coord_list
        for count in temp_list:
            coord_list = list()
            coord_list.append(convert_coord(count, name='rev_rot_90'))
            print("90 counter-clockwise coordinate:", coord_list[0])
            # img_rot2 = draw_bbox(img_rot2, coord_list[0])
            # img_rot2 = draw_center(img_rot2, coord_list[0])
            for coindex in range(0, len(coord_list[0])):
                if coindex != 4:
                    temp_sentence = temp_sentence + str(coord_list[0][coindex]) + ' '
                else:
                    temp_sentence = temp_sentence + str(coord_list[0][coindex]) + '\n'
            save_annot_list.append(temp_sentence)
            temp_sentence = ''
        img_rot2 = filtering(img_rot2, random.randint(0, 5))
        cv2.imwrite(new_image_path + image_list[num][:-4] + "_rot_2" + ".jpg", img_rot2)
        f2 = open(new_label_path + image_list[num][:-4] + "_rot_2" + ".txt", 'w')
        for saveline in save_annot_list:
            f2.write(saveline)
        save_annot_list = list()
        temp_sentence = ''
        # cv2.imshow("img_rot2", img_rot2)
        # cv2.waitKey(0)
        coord_list = temp_list

        # Filp = -1
        temp_list = coord_list
        for count in temp_list:
            coord_list = list()
            coord_list.append(convert_coord(count, name='rev_flip_1'))
            print("flip 1 coordinate:", coord_list[0])
            # img_flip1 = draw_bbox(img_flip1, coord_list[0])
            # img_flip1 = draw_center(img_flip1, coord_list[0])
            for coindex in range(0, len(coord_list[0])):
                if coindex != 4:
                    temp_sentence = temp_sentence + str(coord_list[0][coindex]) + ' '
                else:
                    temp_sentence = temp_sentence + str(coord_list[0][coindex]) + '\n'
            save_annot_list.append(temp_sentence)
            temp_sentence = ''
        img_flip1 = filtering(img_flip1, random.randint(0, 5))
        cv2.imwrite(new_image_path + image_list[num][:-4] + "_flip_1" + ".jpg", img_flip1)
        f2 = open(new_label_path + image_list[num][:-4] + "_flip_1" + ".txt", 'w')
        for saveline in save_annot_list:
            f2.write(saveline)
        save_annot_list = list()
        temp_sentence = ''
        # cv2.imshow("img_flip1", img_flip1)
        # cv2.waitKey(0)
        coord_list = temp_list

        # Filp = 1
        temp_list = coord_list
        for count in temp_list:
            coord_list = list()
            coord_list.append(convert_coord(count, name='flip_1'))
            print("flip 2 coordinate:", coord_list[0])
            # img_flip2 = draw_bbox(img_flip2, coord_list[0])
            # img_flip2 = draw_center(img_flip2, coord_list[0])
            for coindex in range(0, len(coord_list[0])):
                if coindex != 4:
                    temp_sentence = temp_sentence + str(coord_list[0][coindex]) + ' '
                else:
                    temp_sentence = temp_sentence + str(coord_list[0][coindex]) + '\n'
            save_annot_list.append(temp_sentence)
            temp_sentence = ''
        img_flip2 = filtering(img_flip2, random.randint(0, 5))
        cv2.imwrite(new_image_path + image_list[num][:-4] + "_flip_2" + ".jpg", img_flip2)
        f2 = open(new_label_path + image_list[num][:-4] + "_flip_2" + ".txt", 'w')
        for saveline in save_annot_list:
            f2.write(saveline)
        save_annot_list = list()
        temp_sentence = ''
        # cv2.imshow("img_flip2", img_flip2)
        # cv2.waitKey(0)
        coord_list = temp_list

        # Filp = 0
        temp_list = coord_list
        for count in temp_list:
            coord_list = list()
            coord_list.append(convert_coord(count, name='flip_0'))
            print("flip 3 coordinate:", coord_list[0])
            # img_flip3 = draw_bbox(img_flip3, coord_list[0])
            # img_flip3 = draw_center(img_flip3, coord_list[0])
            for coindex in range(0, len(coord_list[0])):
                if coindex != 4:
                    temp_sentence = temp_sentence + str(coord_list[0][coindex]) + ' '
                else:
                    temp_sentence = temp_sentence + str(coord_list[0][coindex]) + '\n'
            save_annot_list.append(temp_sentence)
            temp_sentence = ''
        img_flip3 = filtering(img_flip3, random.randint(0, 5))
        cv2.imwrite(new_image_path + image_list[num][:-4] + "_flip_3" + ".jpg", img_flip3)
        f2 = open(new_label_path + image_list[num][:-4] + "_flip_3" + ".txt", 'w')
        for saveline in save_annot_list:
            f2.write(saveline)
        save_annot_list = list()
        temp_sentence = ''
        # cv2.imshow("img_flip3", img_flip3)
        # cv2.waitKey(0)
        coord_list = temp_list

        if crop_rate != 0:
            img_crop1 = copy.deepcopy(img[0:int(col * (1 - crop_rate)), 0:int(row * (1 - crop_rate)), :])
            img_crop2 = copy.deepcopy(img[0:int(col * (1 - crop_rate)), int(row * crop_rate):row, :])
            img_crop3 = copy.deepcopy(img[int(col * crop_rate):col, 0:int(row * (1 - crop_rate)), :])
            img_crop4 = copy.deepcopy(img[int(col * crop_rate):col, int(row * crop_rate):row, :])

            # Crop 1
            temp_list = coord_list
            for count in temp_list:
                coord_list = list()
                coord_list.append(convert_coord(count, name='crop_1', original_image=img, converted_image=img_crop1))
                print("Crop 1 coordinate:", coord_list[0])
                # img_crop1 = draw_bbox(img_crop1, coord_list[0])
                # img_crop1 = draw_center(img_crop1, coord_list[0])
                for coindex in range(0, len(coord_list[0])):
                    if coindex != 4:
                        temp_sentence = temp_sentence + str(coord_list[0][coindex]) + ' '
                    else:
                        temp_sentence = temp_sentence + str(coord_list[0][coindex]) + '\n'
                save_annot_list.append(temp_sentence)
                temp_sentence = ''
            img_crop1 = filtering(img_crop1, random.randint(0, 5))
            cv2.imwrite(new_image_path + image_list[num][:-4] + "_crop_1" + ".jpg", img_crop1)
            f2 = open(new_label_path + image_list[num][:-4] + "_crop_1" + ".txt", 'w')
            for saveline in save_annot_list:
                f2.write(saveline)
            save_annot_list = list()
            temp_sentence = ''
            # cv2.imshow("img_crop1", img_crop1)
            # cv2.waitKey(10)
            coord_list = temp_list

            # Crop 2
            temp_list = coord_list
            for count in temp_list:
                coord_list = list()
                coord_list.append(convert_coord(count, name='crop_2', original_image=img, converted_image=img_crop2))
                print("Crop 2 coordinate:", coord_list[0])
                # img_crop2 = draw_bbox(img_crop2, coord_list[0])
                # img_crop2 = draw_center(img_crop2, coord_list[0])
                for coindex in range(0, len(coord_list[0])):
                    if coindex != 4:
                        temp_sentence = temp_sentence + str(coord_list[0][coindex]) + ' '
                    else:
                        temp_sentence = temp_sentence + str(coord_list[0][coindex]) + '\n'
                save_annot_list.append(temp_sentence)
                temp_sentence = ''
            img_crop2 = filtering(img_crop2, random.randint(0, 5))
            cv2.imwrite(new_image_path + image_list[num][:-4] + "_crop_2" + ".jpg", img_crop2)
            f2 = open(new_label_path + image_list[num][:-4] + "_crop_2" + ".txt", 'w')
            for saveline in save_annot_list:
                f2.write(saveline)
            save_annot_list = list()
            temp_sentence = ''
            # cv2.imshow("img_crop2", img_crop2)
            # cv2.waitKey(0)
            coord_list = temp_list

            # Crop 3
            temp_list = coord_list
            for count in temp_list:
                coord_list = list()
                coord_list.append(convert_coord(count, name='crop_3', original_image=img, converted_image=img_crop3))
                print("Crop 3 coordinate:", coord_list[0])
                # img_crop3 = draw_bbox(img_crop3, coord_list[0])
                # img_crop3 = draw_center(img_crop3, coord_list[0])
                for coindex in range(0, len(coord_list[0])):
                    if coindex != 4:
                        temp_sentence = temp_sentence + str(coord_list[0][coindex]) + ' '
                    else:
                        temp_sentence = temp_sentence + str(coord_list[0][coindex]) + '\n'
                save_annot_list.append(temp_sentence)
                temp_sentence = ''
            img_crop3 = filtering(img_crop3, random.randint(0, 5))
            cv2.imwrite(new_image_path + image_list[num][:-4] + "_crop_3" + ".jpg", img_crop3)
            f2 = open(new_label_path + image_list[num][:-4] + "_crop_3" + ".txt", 'w')
            for saveline in save_annot_list:
                f2.write(saveline)
            save_annot_list = list()
            temp_sentence = ''
            # cv2.imshow("img_crop3", img_crop3)
            # cv2.waitKey(0)
            coord_list = temp_list

            # Crop 4
            temp_list = coord_list
            for count in temp_list:
                coord_list = list()
                coord_list.append(convert_coord(count, name='crop_4', original_image=img, converted_image=img_crop4))
                print("Crop 4 coordinate:", coord_list[0])
                # img_crop4 = draw_bbox(img_crop4, coord_list[0])
                # img_crop4 = draw_center(img_crop4, coord_list[0])
                for coindex in range(0, len(coord_list[0])):
                    if coindex != 4:
                        temp_sentence = temp_sentence + str(coord_list[0][coindex]) + ' '
                    else:
                        temp_sentence = temp_sentence + str(coord_list[0][coindex]) + '\n'
                save_annot_list.append(temp_sentence)
                temp_sentence = ''
            img_crop4 = filtering(img_crop4, random.randint(0, 5))
            cv2.imwrite(new_image_path + image_list[num][:-4] + "_crop_4" + ".jpg", img_crop4)
            f2 = open(new_label_path + image_list[num][:-4] + "_crop_4" + ".txt", 'w')
            for saveline in save_annot_list:
                f2.write(saveline)
            save_annot_list = list()
            temp_sentence = ''
            # cv2.imshow("img_crop4", img_crop4)
            # cv2.waitKey(30)
            coord_list = temp_list

    exit(0)


if __name__ == "__main__":
    image_path = "C:\\dataset\\MyDataset\\polyps_kvasir7000_yolo\\original_train_images\\"
    label_path = "C:\\dataset\\MyDataset\\polyps_kvasir7000_yolo\\original_train_labels\\"
    new_image_path = "C:\\dataset\\MyDataset\\polyps_kvasir7000_yolo\\new_aug_train_image\\"
    new_label_path = "C:\\dataset\\MyDataset\\polyps_kvasir7000_yolo\\new_aug_train_label\\"

    # if you don't want cropping -> crop_rate = 0
    crop_rate = 0
    # crop_rate = 0.15

    augment_for_detection(image_path=image_path,
                          new_image_path=new_image_path,
                          label_path=label_path,
                          new_label_path=new_label_path,
                          crop_rate=crop_rate,
                          save_coord='ccwh',
                          save_type='relat')
