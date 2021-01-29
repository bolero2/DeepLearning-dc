import cv2
import os
import glob
import numpy as np


def coord2xywh(img_shape, input_coord, input_type, coord):
    """
    :param img_shape: row and column
    :param input_coord: 1. ccwh / 2. xywh / 3. xyrb
    :param input_type: 1. relat / 2. abs
    :param coord: coord[0] ~ coord[3]

    :return: output(shape=[4])

    [output standard : xywh + abs]
     - output[0] = xmin
     - output[1] = ymin
     - output[2] = width
     - output[3] = height

    """

    row = img_shape[0]
    col = img_shape[1]
    output = np.zeros(shape=[4])
    if input_coord == 'ccwh':
        if input_type == 'relat':
            output[0] = (coord[0] * col) - (coord[2] * col / 2)
            output[1] = (coord[1] * row) - (coord[3] * row / 2)
            output[2] = coord[2] * col
            output[3] = coord[3] * row
        elif input_type == 'abs':
            output[2] = coord[2]
            output[3] = coord[3]
            output[0] = coord[0] - (coord[2] / 2)
            output[1] = coord[1] - (coord[3] / 2)

    elif input_coord == 'xywh':
        if input_type == 'relat':
            output[0] = coord[0] * col
            output[1] = coord[1] * row
            output[2] = coord[2] * col
            output[3] = coord[3] * row
        elif input_type == 'abs':
            output[0] = coord[0]
            output[1] = coord[1]
            output[2] = coord[2]
            output[3] = coord[3]

    elif input_coord == 'xyrb':
        if input_type == 'relat':
            output[0] = coord[0] * col
            output[1] = coord[1] * row
            output[2] = abs(coord[2] - coord[0]) * col
            output[3] = abs(coord[3] - coord[1]) * row
        elif input_type == 'abs':
            output[0] = coord[0]
            output[1] = coord[1]
            output[2] = abs(coord[2] - coord[0])
            output[3] = abs(coord[3] - coord[1])

    return output


def convert_coord_type(input_coord,
                       input_type,
                       output_coord,
                       output_type,
                       conf_score,
                       image_path,
                       label_path,
                       new_label_path=None):
    """
    :param input_coord: 1. ccwh / 2. xywh / 3. xyrb
    :param input_type: 1. relat / 2. abs
    :param output_coord: --- Same as input coord. ---
    :param output_type: --- Same as input type. ---

    :param conf_score: if confidence score information is in annotation text file, then True / or False ---

    :param image_path: Original Color Image path
    :param label_path: Annotation file path
    :param new_label_path: Annotation file path to be saved

    :return:Save Annotation text file(.txt) in one-step
    """

    if new_label_path is None:
        new_label_path = label_path

    os.chdir(image_path)
    print("os -> Image path : ", image_path)
    image_list = list()
    for file in glob.glob('*.jpg'):
        image_list.append(file)

    os.chdir(label_path)
    print("os -> Label path : ", label_path)
    label_list = list()
    for file in glob.glob('*.txt'):
        label_list.append(file)

    full_sentence = list()

    for i in label_list:
        real_name = i[:-4]

        row, col, ch = cv2.imread(image_path + real_name + ".jpg").shape
        confidence = 0
        lines = open(label_path + real_name + ".txt", 'r').readlines()

        for line in lines:
            temp_sentence = ''
            line_split = line.split(' ')  # line_split = [class_index, coord[0], coord[1], coord[2], coord[3]]
            if line_split[-1] == "\n":
                line_split[-2] = line_split[-2] + line_split[-1]
                del line_split[-1]  # remove character "\n"
            print(f"line_split:{line_split}")
            if conf_score:
                confidence = float(line_split[1])
                coord = line_split[2:]
            else:
                coord = line_split[1:]

            print(f"coord:{coord}")
            class_index = int(line_split[0])
            coord = list(map(float, coord))

            """
            convert type only
            1. 'relat' -> 'abs'
            2.   'abs' -> 'relat'
            """
            if input_coord == output_coord:
                if input_type == 'relat' and output_type == 'abs':
                    if conf_score:
                        temp_sentence = str(class_index) + " " + str(confidence) + " " + \
                                        str(int(coord[0] * col)) + " " + str(int(coord[1] * row)) + " " + \
                                        str(int(coord[2] * col)) + " " + str(int(coord[3] * row)) + "\n"
                    else:
                        temp_sentence = str(class_index) + " " + \
                                        str(int(coord[0] * col)) + " " + str(int(coord[1] * row)) + " " + \
                                        str(int(coord[2] * col)) + " " + str(int(coord[3] * row)) + "\n"
                elif input_type == 'abs' and output_type == 'relat':
                    if conf_score:
                        temp_sentence = str(class_index) + " " + str(confidence) + " " + \
                                        str(coord[0] / col) + " " + str(coord[1] / row) + " " + \
                                        str(coord[2] / col) + " " + str(coord[3] / row) + "\n"
                    else:
                        temp_sentence = str(class_index) + " " + \
                                        str(coord[0] / col) + " " + str(coord[1] / row) + " " + \
                                        str(coord[2] / col) + " " + str(coord[3] / row) + "\n"
                else:
                    print("What do you want...? There is Error! (\\  '0`/)")
                    exit(0)
                full_sentence.append(temp_sentence)

            else:
                xywh = coord2xywh(img_shape=[row, col], input_coord=input_coord, input_type=input_type, coord=coord)

                if output_coord == 'ccwh':
                    if output_type == 'relat':
                        width_relat = xywh[2] / col
                        height_relat = xywh[3] / row
                        center_x = float(xywh[0] / col + (width_relat / 2))
                        center_y = float(xywh[1] / row + (height_relat / 2))
                        if conf_score:
                            temp_sentence = str(class_index) + " " + str(confidence) + " " + \
                                            str(center_x) + " " + str(center_y) + " " + \
                                            str(width_relat) + " " + str(height_relat) + "\n"
                        else:
                            temp_sentence = str(class_index) + " " + \
                                            str(center_x) + " " + str(center_y) + " " + \
                                            str(width_relat) + " " + str(height_relat) + "\n"
                    elif output_type == 'abs':
                        width = xywh[2]
                        height = xywh[3]
                        center_x = int(xywh[0] + (width / 2))
                        center_y = int(xywh[1] + (height / 2))
                        if conf_score:
                            temp_sentence = str(class_index) + " " + str(confidence) + " " + \
                                            str(center_x) + " " + str(center_y) + " " + \
                                            str(int(width)) + " " + str(int(height)) + "\n"
                        else:
                            temp_sentence = str(class_index) + " " + \
                                            str(center_x) + " " + str(center_y) + " " + \
                                            str(int(width)) + " " + str(int(height)) + "\n"

                elif output_coord == 'xywh':
                    if output_type == 'relat':
                        if conf_score:
                            temp_sentence = str(class_index) + " " + str(confidence) + " " + \
                                            str(xywh[0] / col) + " " + str(xywh[1] / row) + " " + \
                                            str(xywh[2] / col) + " " + str(xywh[3] / row) + "\n"
                        else:
                            temp_sentence = str(class_index) + " " + \
                                            str(xywh[0] / col) + " " + str(xywh[1] / row) + " " + \
                                            str(xywh[2] / col) + " " + str(xywh[3] / row) + "\n"
                    elif output_type == 'abs':
                        if conf_score:
                            temp_sentence = str(class_index) + " " + str(confidence) + " " + \
                                            str(int(xywh[0])) + " " + str(int(xywh[1])) + " " + \
                                            str(int(xywh[2])) + " " + str(int(xywh[3])) + "\n"
                        else:
                            temp_sentence = str(class_index) + " " + \
                                            str(int(xywh[0])) + " " + str(int(xywh[1])) + " " + \
                                            str(int(xywh[2])) + " " + str(int(xywh[3])) + "\n"

                elif output_coord == 'xyrb':
                    if output_type == 'relat':
                        if conf_score:
                            temp_sentence = str(class_index) + " " + str(confidence) + " " + \
                                            str(float(xywh[0] / col)) + " " + \
                                            str(float(xywh[1] / row)) + " " + \
                                            str(float((xywh[0] + xywh[2]) / col)) + " " + \
                                            str(float((xywh[1] + xywh[3]) / row)) + "\n"
                        else:
                            temp_sentence = str(class_index) + " " + \
                                            str(float(xywh[0] / col)) + " " + \
                                            str(float(xywh[1] / row)) + " " + \
                                            str(float((xywh[0] + xywh[2]) / col)) + " " + \
                                            str(float((xywh[1] + xywh[3]) / row)) + "\n"
                    elif output_type == 'abs':
                        if conf_score:
                            temp_sentence = str(class_index) + " " + str(confidence) + " " + \
                                            str(int(xywh[0])) + " " + str(int(xywh[1])) + " " + \
                                            str(int(xywh[0] + xywh[2])) + " " + str(int(xywh[1] + xywh[3])) + "\n"
                        else:
                            temp_sentence = str(class_index) + " " + \
                                            str(int(xywh[0])) + " " + str(int(xywh[1])) + " " + \
                                            str(int(xywh[0] + xywh[2])) + " " + str(int(xywh[1] + xywh[3])) + "\n"

                full_sentence.append(temp_sentence)
        print(f"Annot file: {i} -> Convert [{lines}] to [{full_sentence}]")
        # print(i, full_sentence)
        new_txt = open(new_label_path + real_name + ".txt", 'w')
        new_txt.writelines(full_sentence)
        full_sentence = list()


if __name__ == "__main__":
    image_path = '/home/bolero/.dc/dl/dataset/detection/idc_cancer/new_endo/new_data/'
    label_path = '/home/bolero/.dc/dl/dataset/detection/idc_cancer/new_endo/annot_xyrb/'
    new_label_path = '/home/bolero/.dc/dl/dataset/detection/idc_cancer/new_endo/annot_yolo/'

    convert_coord_type(input_coord='xyrb',
                       input_type='abs',
                       output_coord='ccwh',
                       output_type='relat',
                       conf_score=False,
                       image_path=image_path,
                       label_path=label_path,
                       new_label_path=new_label_path)
    exit(0)
