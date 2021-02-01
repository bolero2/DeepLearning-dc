import os
import shutil as sh
import glob


path1 = '/home/bolero/.dc/dl/dataset/detection/instrument/tooldetection/m2cai16-tool-locations/JPEGImages/'
save_path = '/home/bolero/.dc/dl/dataset/detection/instrument/tooldetection/m2cai16-tool-locations/yolo_dataset/' 

annot_path = '/home/bolero/.dc/dl/dataset/detection/instrument/tooldetection/m2cai16-tool-locations/annot_ccwh/'  

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
        sh.copy(path1 + i, save_path + "train/")
        sh.copy(annot_path + i[:-3] + "txt", save_path + "train/")
    elif int(train_rate * img_count) <= count < int((val_rate + train_rate) * img_count):
        sh.copy(path1 + i, save_path + "eval/")
        sh.copy(annot_path + i[:-3] + "txt", save_path + "eval/")
    else:
        sh.copy(path1 + i, save_path + "test/")
        sh.copy(annot_path + i[:-3] + "txt", save_path + "test/")
    count += 1
        
