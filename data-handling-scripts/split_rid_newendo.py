import pandas as pd
import glob
import os
import random
import shutil as sh

df = pd.read_csv("RID_for_test_newENDO.csv")

rid = list(df.loc[df['GUBUN'] == "TRAIN"]['RID'])
print(f'RID= {rid}')

root = os.getcwd()
print(root)
os.chdir(f'{root}/yolo_dataset/')

for imagename in glob.glob('*.jpg'):
    three = imagename[0:3]
    if three[-1] == '내':
        idnum = three[0]
    elif three[-1] == '_':
        idnum = three[0:2]
    else:
        idnum = three

    if int(idnum) in rid:
        print(f'Target= {idnum}')
        sh.copy(f'{root}/yolo_dataset/{imagename}', f'{root}/rid/train/')

    """
    if len(idx) == 1:
        os.chdir(root + "/" + idx)
        full_path = f'{root}/{idx}/'
        # print(os.getcwd())
        filelist = os.listdir()
        for filename in filelist:
            file_rid = filename[0:8]
            if file_rid in rid8:
                shutil.copy(f'{full_path}{filename}', f'{root}/test/{idx}/')
            else:
                shutil.copy(f'{full_path}{filename}', f'{root}/train/{idx}/')
                """
