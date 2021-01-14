import os
import glob


path = '/home/yb/dc/yolo_c16/train/'
os.chdir(path)
annot_list = [x for x in glob.glob('*.txt')]
sentence_list = list()

for i in annot_list:
    f = open(path + i, 'r')
    lines = f.readlines()
    sentence_list = ['0' + sentence[1:] for sentence in lines]
    print(sentence_list)

    f2 = open(path + i, 'w')
    """
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
        """
