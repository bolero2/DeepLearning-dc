import os
import glob


path = '/home/yb/dc/yolo_c16/eval/'
os.chdir(path)
annot_list = [x for x in glob.glob('*.txt')]
sentence_list = list()

for i in annot_list:
    f = open(path + i, 'r')
    lines = f.readlines()
    sentence_list = ['0' + sentence[1:] for sentence in lines]
    print(sentence_list)
    f.close()

    f2 = open(path + i, 'w')
    f2.writelines(sentence_list)
    f2.close()
