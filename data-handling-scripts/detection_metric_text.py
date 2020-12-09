import cv2
import os
import glob

file = "C:\\dataset\\MyDataset\\polyps_kvasir7000_coco\\test_kvasir300_coco\\annotation.json"

image_path = "C:\\dataset\\MyDataset\\polyps_kvasir7000_coco\\test_kvasir300_coco\\"
label_path = "C:\\dataset\\MyDataset\\polyps_kvasir7000_yolo\\test_folder_integ\\"
gt_label_path = "C:\\dataset\\MyDataset\\polyps_kvasir7000_coco\\test_gt_annotation\\"

os.chdir(image_path)
image_list = list()

for file in glob.glob('*.jpg'):
    image_list.append(file)

sentence = list()
sentence_list = list()

for image in image_list:
    row, col, ch = cv2.imread(image_path + image).shape
    print(row, col, ch)
    lines = open(label_path + image[:-4] + ".txt").readlines()
    for line in lines:
        aa = line[:-1].split(' ')
        width = float(aa[3])
        height = float(aa[4])
        center_x = float(aa[1])
        center_y = float(aa[2])
        xmin = (center_x - width / 2) * col
        ymin = (center_y - height / 2) * row
        sentence.append(str(aa[0]) + ' ' + str(int(xmin)) + ' ' + str(int(ymin)) + ' ' + str(int(width * col)) + ' ' + str(int(height * row)) + '\n')
    print(sentence)
    text = open(gt_label_path + image[:-4] + ".txt", 'w')
    text.writelines(sentence)
    text.close()
    sentence = list()



    # text = open(gt_label_path + image[:-4] + ".txt")

