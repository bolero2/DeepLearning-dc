import cv2
import os
import glob


def convert_poly2bbox(image_path, gt_path, annot_path=None, diff_index=0, annot_coord='ccwh', annot_type='relat'):
    """
    :param image_path: Original Color Image path
    :param gt_path: Ground Truth Image path
    :param annot_path: Annotation file path to be saved

    :param diff_index:
    ground truth index position in the image file where the name of the original image file begins
    example)
     - gt_image = gt_im1.jpg
     - original_image = im1.jpg

     ---> diff_index = 3
     [      3            ]
     (g t _ i m 1 . j p g [3:] = im1.jpg)
     If the ground-truth image file and the original image file have the same name,
        -> diff_index will be 0.

    :param annot_coord:
    1) ccwh = center_x, center_y, width, height
    2) xywh = xmin, ymin, width, height
    3) xyrb = xmin, ymin, xmax, ymax(Right, Bottom)

    :param annot_type:
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

    :return: Save Annotation text file(.txt) in one-step
    """

    if annot_path is None:
        annot_path = image_path

    sentence_list = list()
    os.chdir(gt_path)
    print("os -> Change directory : ", gt_path)
    file_list = list()

    for file in glob.glob('*.jpg'):
        file_list.append(file)

    for i in file_list:
        img_color = cv2.imread(image_path + i[diff_index:])
        row, col, ch = img_color.shape
        print(row, col, ch)

        img_gt = cv2.imread(gt_path + i)
        img_gray = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)

        # Contouring 1 -> Image Binary
        otsu_thr, imgBin = cv2.threshold(img_gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Contouring 2 -> Find Contours
        contours, hierarchy = cv2.findContours(image=imgBin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            boundRect = cv2.boundingRect(cnt)
            print("Image:", i, " | Bound Rectangle >> ", boundRect)
            if annot_coord == 'ccwh':
                if annot_type == 'relat':
                    xmin = float(boundRect[0] / col)
                    ymin = float(boundRect[1] / row)
                    width = float(boundRect[2] / col)
                    height = float(boundRect[3] / row)

                    # Remove useless annotation information
                    # (Maybe USE-LESS!)
                    if 0.01 < width <= 0.99 and 0.01 < height <= 0.99:
                        sentence = "0 " + str(float(xmin + width / 2)) \
                                   + " " + str(float(ymin + height / 2)) \
                                   + " " + str(width) + " " + str(height) + "\n"
                        sentence_list.append(sentence)
                    else:
                        pass
                elif annot_type == 'abs':
                    xmin = boundRect[0]
                    ymin = boundRect[1]
                    width = boundRect[2]
                    height = boundRect[3]

                    if 0.01 < width <= 0.99 and 0.01 < height <= 0.99:
                        sentence = "0 " + str(float(xmin + width / 2)) \
                                   + " " + str(float(ymin + height / 2)) \
                                   + " " + str(width) + " " + str(height) + "\n"
                        sentence_list.append(sentence)
                    else:
                        pass

            elif annot_coord == 'xywh':
                if annot_type == 'relat':
                    xmin = float(boundRect[0] / col)
                    ymin = float(boundRect[1] / row)
                    width = float(boundRect[2] / col)
                    height = float(boundRect[3] / row)

                    if xmin == 0.0:
                        xmin = 0.01
                    if ymin == 0.0:
                        ymin = 0.01

                    if (col * 0.01) < width <= (col * 0.99) and (row * 0.01) < height <= (row * 0.99):
                        sentence = "0 " + str(xmin) + " " + str(ymin) + " " + str(width) + " " + str(height) + "\n"
                        sentence_list.append(sentence)
                    else:
                        pass
                elif annot_type == 'abs':
                    xmin = boundRect[0]
                    ymin = boundRect[1]
                    width = boundRect[2]
                    height = boundRect[3]

                    if xmin == 0:
                        xmin = col * 0.01
                    if ymin == 0:
                        ymin = row * 0.01

                    if (col * 0.01) < width <= (col * 0.99) and (row * 0.01) < height <= (row * 0.99):
                        sentence = "0 " + str(xmin) + " " + str(ymin) + " " + str(width) + " " + str(height) + "\n"
                        sentence_list.append(sentence)
                    else:
                        pass

            elif annot_coord == 'xyrb':
                if annot_type == 'relat':
                    xmin = float(boundRect[0] / col)
                    ymin = float(boundRect[1] / row)
                    width = float(boundRect[2] / col)
                    height = float(boundRect[3] / row)

                    xmax = xmin + width
                    ymax = ymin + height
                    if xmin == 0.0:
                        xmin = 0.01
                    if ymin == 0.0:
                        ymin = 0.01

                    if (col * 0.01) < boundRect[2] <= (col * 0.99) and (row * 0.01) < boundRect[3] <= (row * 0.99):
                        sentence = "0 " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n"
                        sentence_list.append(sentence)
                    else:
                        pass
                elif annot_type == 'abs':
                    xmin = boundRect[0]
                    ymin = boundRect[1]
                    xmax = xmin + boundRect[2]
                    ymax = ymin + boundRect[3]

                    if xmin == 0:
                        xmin = col * 0.01
                    if ymin == 0:
                        ymin = row * 0.01

                    if (col * 0.01) < boundRect[2] <= (col * 0.99) and (row * 0.01) < boundRect[3] <= (row * 0.99):
                        sentence = "0 " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n"
                        sentence_list.append(sentence)
                    else:
                        pass

        f = open(annot_path + i[diff_index:-4] + ".txt", "w")
        f.writelines(sentence_list)
        f.close()

        sentence_list = list()


if __name__ == "__main__":
    image_path = "C:\\dataset\\MedicalDataset\\Kvasir-SEG\\images\\"
    gt_path = "C:\\dataset\\MedicalDataset\\Kvasir-SEG\\masks\\"
    annot_path = "C:\\dataset\\MedicalDataset\\Kvasir-SEG\\annot_relat_ccwh\\"

    convert_poly2bbox(image_path=image_path,
                      gt_path=gt_path,
                      annot_path=annot_path,
                      diff_index=0,
                      annot_coord='ccwh',
                      annot_type='relat')
    exit(0)
