import glob
import os
import shutil as sh


root_path = os.getcwd()
annot_path = '/home/yb/dc/yolo_c16/'
target = 'test'

os.chdir(f'{root_path}/{target}')
imagelist = [x for x in glob.glob('*.jpg')]
for imagename in imagelist:
    annot_name = imagename[:-3] + 'txt'
    sh.copy(f'{annot_path}total/{annot_name}', f'{root_path}/{target}')
# print(imagelist)
