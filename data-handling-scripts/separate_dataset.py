import os
import shutil as sh
import glob


path1 = '/home/yb/dc/AI_C18/cancer/'
path2 = '/home/yb/dc/yolo_c18/'

annot_path = '/home/yb/dc/annotations_c18/'

val_rate = 0.1
test_rate = 0.1
train_rate = 1.0 - (val_rate + test_rate)

os.chdir(path1)
img_list = [x for x in glob.glob('*.jpg')]
img_count = len(img_list)

os.chdir(annot_path)
annot_list = [x for x in glob.glob('*.txt')]

count = 0
for i in img_list:
    if count < int(train_rate * img_count):
        sh.copy(path1 + i, path2 + "train/")
        sh.copy(annot_path + i[:-3] + "txt", path2 + "train/")
    elif int(train_rate * img_count) <= count < int((val_rate + train_rate) * img_count):
        sh.copy(path1 + i, path2 + "eval/")
        sh.copy(annot_path + i[:-3] + "txt", path2 + "eval/")
    else:
        sh.copy(path1 + i, path2 + "test/")
        sh.copy(annot_path + i[:-3] + "txt", path2 + "test/")
    count += 1
        
