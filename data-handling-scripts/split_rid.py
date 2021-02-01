import pandas as pd
import glob
import os
import random
import shutil

df = pd.read_csv("RID_for_test.csv")

rid = list(df.loc[df['GUBUN'] == "TEST"]['RID'])
rid8 = [str(x).zfill(8) for x in rid]
print(rid8)

root = os.getcwd()
print(root)

for idx in os.listdir():
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
