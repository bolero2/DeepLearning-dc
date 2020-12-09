import cv2
import os

path = "C:\\dataset\\MedicalDataset\\ETIS-LaribPolypDB\\groundtruth\\"
file_list = os.listdir(path)

for i in file_list:
    img = cv2.imread(path + i)
    # print(i[:-4] + ".jpg")
    cv2.imwrite(path + i[:-4] + ".jpg", img)
    cv2.waitKey(1)
