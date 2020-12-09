import cv2
import os
import matplotlib.pyplot as plt

image_path = "C:\\dataset\\OpenedDataset\\kitti_tiny\\training\\image_2\\"
label_path = "C:\\dataset\\OpenedDataset\\kitti_tiny\\training\\label_2\\"

if __name__ == "__main__":
    image_list = os.listdir(image_path)
    # print(image_list)

    for i in image_list:
        name = i[:-5]
        print("name :", i)
        img = cv2.imread(image_path + i)
        annot_file = label_path + name + ".txt"
        f = open(annot_file)
        lines = f.readlines()
        for line in lines:
            coord = line.split(' ')[4:8]
            print(line.split(' ')[4:8])
            cv2.rectangle(img,
                          pt1=(int(float(coord[0])), int(float(coord[1]))),
                          pt2=(int(float(coord[2])), int(float(coord[3]))), color=(0, 0, 255), thickness=2)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", img)
        cv2.waitKey(0)
