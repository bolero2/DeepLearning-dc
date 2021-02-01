import os
import glob


original_path = '/home/yb/dc/yolo_c16/new_endo/'

os.chdir(original_path)
imglist = [x for x in glob.glob("*.jpg")]

for imgname in imglist:
    annot_file = f'{original_path}{imgname[:-3]}txt'
    f = open(annot_file, 'r')
    lines = f.readlines()

    count = len(lines)
    if count == 0:
        os.remove(f'{original_path}{imgname}')
        os.remove(f'{annot_file}')
    f.close()
