import cv2
import glob
import os
import numpy as np
import copy
import argparse


def _draw_center(image, center, color=(0, 255, 0)):
    image = cv2.circle(image, center=(int(center[0]), int(center[1])), radius=1, color=color, thickness=5)

    return image


def _draw_bbox(image, xyrb, type, confidence=None):
    xyrb = list(map(int, xyrb))
    color = (0, 0, 0)
    if type == 1:
        color = (0, 0, 255)
    elif type == 2:
        color = (255, 0, 0)

    if confidence is not None:
        image = cv2.rectangle(image, pt1=(xyrb[0], xyrb[1]), pt2=(xyrb[2], xyrb[3]), color=color, thickness=3)
        center_x = xyrb[0] + (abs(xyrb[0] - xyrb[2]) / 2)
        center_y = xyrb[1] + (abs(xyrb[1] - xyrb[3]) / 2)
        # image = _draw_center(image, center=(center_x, center_y), color=(0, 255, 255))

        bg_color = (0, 0, 0)
        bg = np.full((image.shape), bg_color, dtype=np.uint8)
        bg = cv2.putText(bg, text=f"{str(round(confidence, 2))}", org=(xyrb[0], xyrb[1]),
                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        x, y, w, h = cv2.boundingRect(bg[:, :, 2])

        # copy bounding box region from bg to img
        # result = image.copy()
        image[y:y + h, x:x + w] = copy.deepcopy(bg[y:y + h, x:x + w])
        return image
    else:
        image = cv2.rectangle(image, pt1=(xyrb[0], xyrb[1]), pt2=(xyrb[2], xyrb[3]), color=color, thickness=3)
        center_x = xyrb[0] + (abs(xyrb[0] - xyrb[2]) / 2)
        center_y = xyrb[1] + (abs(xyrb[1] - xyrb[3]) / 2)
        # image = _draw_center(image, center=(center_x, center_y), color=(0, 255, 255))
        return image


def draw_bbox(image_path, num_type,
              path1, path1_coord, path1_coord_type,
              path1_is_confidence=False,
              path2=None, path2_coord=None, path2_coord_type=None,
              path2_is_confidence=False,
              is_save=False):
    """
    :param image_path: Original Image file path
    :param label_path: Annotation file path
    :param type:
    [type]
    1. Absolute Coordinate -> type='abs' (only xyrb coord)
    2. relative Coordinate -> type='relat' (only ccwh coord)

    !!!!! Use convert_coordinate.py !!!!!

    """
    original_pwd = os.getcwd()
    if is_save:
        os.chdir(original_pwd)
        if not os.path.exists(f"{original_pwd}/BndboxImage"):
            os.mkdir(f"{original_pwd}/BndboxImage")
        os.chdir(f'{original_pwd}/BndboxImage/')
        dircount = len(os.listdir())

    os.chdir(image_path)
    print(f"Image files directory: {image_path}")
    image_list = list()
    for file in glob.glob('*.jpg'):
        image_list.append(file)

    image_list = sorted(image_list)

    os.chdir(path1)
    print(f"Path1 directory: {path1}")
    path1_label_list = list()
    for file in glob.glob('*.txt'):
        path1_label_list.append(file)

    valid_param = None
    if num_type > 1:
        try:
            valid_param = len(path2 + path2_coord + path2_coord_type)
        except:
            valid_param = None
            print("'path2' parameter has None Value.")

        if valid_param is not None:
            print("\nPath2 parameter has value.")
            os.chdir(path2)
            print(f"Path2 directory: {path2}")
            path2_label_list = list()
            for file in glob.glob('*.txt'):
                path2_label_list.append(file)

    for image_name in image_list:
        path1_coordinate = list()
        path2_coordinate = list()

        img = cv2.imread(image_path + image_name)
        row, col, ch = img.shape

        try:
            path1_lines = open(path1 + image_name[:-4] + ".txt").readlines()
        except:
            print("no path1 annotation file!")
            path1_lines = ['0 0 0 0 0 0\n']
        for path1_line in path1_lines:
            path1_line = path1_line.split(' ')
            for i in range(0, len(path1_line)):
                if path1_line[-1] == '\n':
                    path1_line[-2] = path1_line[-2] + path1_line[-1]
                    del path1_line[-1]
            path1_coordinate.append(list(map(float, path1_line[1:])))
        print(f"{image_name} | coord: {path1_coordinate}")

        if valid_param is not None:
            try:
                path2_lines = open(path2 + image_name[:-4] + ".txt").readlines()
            except:
                print("no path2 annotation file!")
                path2_lines = ['0 0 0 0 0 0\n']

            for path2_line in path2_lines:
                path2_line = path2_line.split(' ')
                for i in range(0, len(path2_line)):
                    if path2_line[-1] == '\n':
                        path2_line[-2] = path2_line[-2] + path2_line[-1]
                        del path2_line[-1]
                path2_coordinate.append(list(map(float, path2_line[1:])))
            print(f"{image_name} | coord: {path2_coordinate}")

        # change coordinate system into [xyrb -> xmin, ymin, xmax, ymax] + [abs]
        for coord in path1_coordinate:
            confidence = 0
            if path1_is_confidence:
                confidence = coord[-1]
                new_coord = coord[0:4]
            else:
                new_coord = coord
            xmin = 0
            ymin = 0
            xmax = 0
            ymax = 0
            # print("i:", i)
            if path1_coord == 'ccwh' and path1_coord_type == 'relat':
                xmin = new_coord[0] * col - (new_coord[2] * col / 2)
                ymin = new_coord[1] * row - (new_coord[3] * row / 2)
                xmax = new_coord[0] * col + (new_coord[2] * col / 2)
                ymax = new_coord[1] * row + (new_coord[3] * row / 2)
            elif path1_coord == 'ccwh' and path1_coord_type == 'abs':
                xmin = new_coord[0] - new_coord[2] / 2
                ymin = new_coord[1] - new_coord[3] / 2
                xmax = new_coord[0] + new_coord[2] / 2
                ymax = new_coord[1] + new_coord[3] / 2
            elif path1_coord == 'xywh' and path1_coord_type == 'relat':
                xmin = new_coord[0] * col
                ymin = new_coord[1] * row
                xmax = new_coord[0] * col + new_coord[2] * col
                ymax = new_coord[1] * row + new_coord[3] * row
            elif path1_coord == 'xywh' and path1_coord_type == 'abs':
                xmin = new_coord[0]
                ymin = new_coord[1]
                xmax = new_coord[0] + new_coord[2]
                ymax = new_coord[1] + new_coord[3]
            elif path1_coord == 'xyrb' and path1_coord_type == 'relat':
                xmin = new_coord[0] * col
                ymin = new_coord[1] * row
                xmax = new_coord[2] * col
                ymax = new_coord[3] * row
            elif path1_coord == 'xyrb' and path1_coord_type == 'abs':
                xmin = new_coord[0]
                ymin = new_coord[1]
                xmax = new_coord[2]
                ymax = new_coord[3]
            if path1_is_confidence:
                img = _draw_bbox(img, [xmin, ymin, xmax, ymax], type=2, confidence=confidence)
            else:
                img = _draw_bbox(img, [xmin, ymin, xmax, ymax], type=2, confidence=None)

        if valid_param is not None:
            for coord in path2_coordinate:
                confidence = 0
                if path2_is_confidence:
                    confidence = coord[-1]
                    new_coord = coord[0:4]
                else:
                    new_coord = coord
                xmin = 0
                ymin = 0
                xmax = 0
                ymax = 0
                # print("i:", i)
                if path2_coord == 'ccwh' and path2_coord_type == 'relat':
                    xmin = new_coord[0] * col - (new_coord[2] * col / 2)
                    ymin = new_coord[1] * row - (new_coord[3] * row / 2)
                    xmax = new_coord[0] * col + (new_coord[2] * col / 2)
                    ymax = new_coord[1] * row + (new_coord[3] * row / 2)
                elif path2_coord == 'ccwh' and path2_coord_type == 'abs':
                    xmin = new_coord[0] - new_coord[2] / 2
                    ymin = new_coord[1] - new_coord[3] / 2
                    xmax = new_coord[0] + new_coord[2] / 2
                    ymax = new_coord[1] + new_coord[3] / 2
                elif path2_coord == 'xywh' and path2_coord_type == 'relat':
                    xmin = new_coord[0] * col
                    ymin = new_coord[1] * row
                    xmax = new_coord[0] * col + new_coord[2] * col
                    ymax = new_coord[1] * row + new_coord[3] * row
                elif path2_coord == 'xywh' and path2_coord_type == 'abs':
                    xmin = new_coord[0]
                    ymin = new_coord[1]
                    xmax = new_coord[0] + new_coord[2]
                    ymax = new_coord[1] + new_coord[3]
                elif path2_coord == 'xyrb' and path2_coord_type == 'relat':
                    xmin = new_coord[0] * col
                    ymin = new_coord[1] * row
                    xmax = new_coord[2] * col
                    ymax = new_coord[3] * row
                elif path2_coord == 'xyrb' and path2_coord_type == 'abs':
                    xmin = new_coord[0]
                    ymin = new_coord[1]
                    xmax = new_coord[2]
                    ymax = new_coord[3]
                if path2_is_confidence:
                    img = _draw_bbox(img, [xmin, ymin, xmax, ymax], type=1, confidence=confidence)
                else:
                    img = _draw_bbox(img, [xmin, ymin, xmax, ymax], type=1, confidence=None)
        # cv2.imshow(image_name, img)
        # key = cv2.waitKey(10)
        # if key == ord('q'):
        #     print("key Q is entered.")
        #     exit(0)
        if is_save:
            if not os.path.exists(f"{original_pwd}/BndboxImage/save{dircount + 1}"):
                os.mkdir(f"{original_pwd}/BndboxImage/save{dircount + 1}")

            print(f"Save image >>> {original_pwd}/BndboxImage/save{dircount + 1}/bbox_{image_name}")
            cv2.imwrite(f"{original_pwd}/BndboxImage/save{dircount + 1}/bbox_{image_name}", img)
        else:
            cv2.imshow(image_name, img)
            key = cv2.waitKey(10)
            if key == ord('q'):
                print("key Q is entered.")
                exit(0)
            cv2.destroyWindow(image_name)


if __name__ == "__main__":
    image_path = "/home/bolero/.dc/dl/yolov5-c16-rid/test_dataset/"
    num_type = 2

    path1 = "/home/bolero/.dc/dl/yolov5-c16-rid/test_dataset/"
    path1_coord = 'ccwh'
    path1_coord_type = 'relat'
    path1_is_confidence = False

    path2 = "/home/bolero/.dc/dl/yolov5-c16-rid/c16_rid_aug_img416/labels/"
    path2_coord = 'ccwh'
    path2_coord_type = 'relat'
    path2_is_confidence = True
    
    """
    path2 = None
    path2_coord = None
    parh2_coord_type = None
    path2_is_confidence = False
    """

    is_save = True 

    draw_bbox(image_path=image_path,
              num_type=num_type,
              path1=path1, path1_coord=path1_coord, path1_coord_type=path1_coord_type,
              path1_is_confidence=path1_is_confidence,
              path2=path2, path2_coord=path2_coord, path2_coord_type=path2_coord_type,
              path2_is_confidence=path2_is_confidence,
              is_save=is_save)
    exit(0)
