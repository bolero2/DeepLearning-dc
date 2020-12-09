path = "C:\\dataset\\MyDataset\\mmdet_kvasir7000_ETIS_text_xyrb\\training\\label\\"

import os

annot_list = os.listdir(path)
sentence_list = list()

for i in annot_list:
    lines = open(path + i, 'r').readlines()
    for line in lines:
        temp = line.split(' ')
        # print(len(temp))
        for t in range(0, len(temp)):
            if t == 1 and temp[1] == '0':
                temp[1] = '1'
                print("temp[1] = 0")
            elif t == 2 and temp[2] == '0':
                temp[2] = '1'
                print("temp[2] = 0")
        print(temp)
