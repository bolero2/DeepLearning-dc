import glob
import os
import shutil as sh


root_path = '/home/bolero/.dc/dl/dataset/detection/idc_cancer/new_endo/rid/'
annot_path = '/home/bolero/.dc/dl/dataset/detection/idc_cancer/new_endo/yolo_dataset/'
target = 'eval'

os.chdir(f'{root_path}/{target}')
imagelist = [x for x in glob.glob('*.jpg')]
for imagename in imagelist:
    annot_name = imagename[:-3] + 'txt'
    sh.copy(f'{annot_path}/{annot_name}', f'{root_path}/{target}')
# print(imagelist)
