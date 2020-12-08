import cv2
import glob
import os


def draw_center(image, bbox, type=None):
    if type == 'relat':
        row, col, ch = image.shape
        pt = (int(col * bbox[1]), int(row * bbox[2]))
        image = cv2.circle(image, center=pt, radius=1, color=(255, 0, 0), thickness=5)

        text = str(round(pt[0], 0)) + "/" + str(round(pt[1], 0))
        image = cv2.putText(image, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                            thickness=3)
        return image

    if type == 'abs':
        print(bbox)
        pt = (int((bbox[1] + bbox[3]) / 2), int((bbox[2] + bbox[4]) / 2))
        image = cv2.circle(image, center=pt, radius=1, color=(255, 0, 0), thickness=5)

        # text = str(round(pt[0], 0)) + "/" + str(round(pt[1], 0))
        # image = cv2.putText(image, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
        #                     color=(255, 0, 0), thickness=3)
        return image


def draw_bbox(image, bbox, type=None):
    if type == 'relat':
        row, col, ch = image.shape
        pt1 = (int(col * bbox[1] - (col * bbox[3]) / 2), int(row * bbox[2] - (row * bbox[4]) / 2))
        pt2 = (int(col * bbox[1] + (col * bbox[3]) / 2), int(row * bbox[2] + (row * bbox[4]) / 2))

        return cv2.rectangle(image, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=5)

    if type == 'abs':
        print(bbox)
        return cv2.rectangle(image, pt1=(int(bbox[1]), int(bbox[2])), pt2=(int(bbox[3]), int(bbox[4])),
                             color=(0, 0, 255), thickness=5)


def show_bbox(image_path, label_path, type):
    """
    :param image_path: Original Image file path
    :param label_path: Annotation file path
    :param type:
    [type]
    1. Absolute Coordinate -> type='abs' (only xyrb coord)
    2. relative Coordinate -> type='relat' (only ccwh coord)

    !!!!! Use convert_coordinate.py !!!!!

    """
    os.chdir(image_path)
    print("os -> Change directory : ", image_path)
    image_list = list()
    for file in glob.glob('*.jpg'):
        image_list.append(file)

    os.chdir(label_path)
    print("os -> Change directory : ", label_path)
    label_list = list()
    for file in glob.glob('*.txt'):
        label_list.append(file)

    coord = list()
    for num in range(0, len(image_list)):
        img = cv2.imread(image_path + image_list[num])
        lines = open(label_path + image_list[num][:-4] + ".txt").readlines()
        for line in lines:
            line = line.split(' ')
            for i in range(0, len(line)):
                coord.append(line[i])

            img = draw_bbox(img, coord, type=type)
            # img = draw_center(img, coord, type=type)
            coord = list()
        cv2.imshow(image_list[num], img)
        cv2.waitKey(0)
        cv2.destroyWindow(image_list[num])


if __name__ == "__main__":
    image_path = "C:\\dataset\\MyDataset\\detectoRS_kvasir_ETIS\\image\\"
    label_path = "C:\\dataset\\MyDataset\\detectoRS_kvasir_ETIS\\label\\"

    show_bbox(image_path=image_path, label_path=label_path, type='abs')
    exit(0)

