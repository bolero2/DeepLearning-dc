import os
import cv2

target = "C:\\Users\\bolero\\Desktop\\kvasir_test_100\\"
compare = "C:\\dataset\\MyDataset\\sample300-yolo\\image\\"

target_list = os.listdir(target)
compare_list = os.listdir(compare)

for i in target_list:
    original = cv2.imread(target + i)
    original_col, original_row, original_ch = original.shape
    for j in compare_list:
        compare_img = cv2.imread(compare + j)
        col, row, ch = compare_img.shape
        # output = np.zeros()
        if sum(original - compare_img) == 0:
            print("Found:", original, "<- Trash gogo!")