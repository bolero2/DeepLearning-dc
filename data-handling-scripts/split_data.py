import glob
import shutil
import os
import random

path1 = '/home/yb/dc/c16_rid/train/'
path2 = '/home/yb/dc/c16_rid/eval/'
path3 = '/home/yb/dc/c16_rid/test/'
val_rate = 0.1
test_rate = 0.0
train_rate = 1.0 - (val_rate + test_rate)

os.chdir(path1)
for index in os.listdir('./'):
    os.chdir(path1 + index)
    total_count = len(os.listdir('./'))
    print(f'total file count: {total_count}')
    limit = int(total_count * val_rate) 
    count = 0
    filenames = os.listdir('./')
    random.shuffle(filenames)
    for filename in filenames:
        shutil.move(f'{path1}{index}/{filename}', f'{path2}{index}/{filename}')
        # print(f'Move this file: {path1}{index}/{filename}')
        count += 1
        if count == limit:
            print(f'total {count} files are copied.')
            break
