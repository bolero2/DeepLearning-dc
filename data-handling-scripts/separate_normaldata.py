import os
import glob
import shutil as sh


pwd = os.getcwd()
os.chdir(f'{pwd}/annotations_c18/')
annot_list = [x for x in glob.glob('*.txt')]

os.chdir(f'{pwd}/AI_C18/')
rid_list = os.listdir()

for rid in rid_list:
    if len(rid) == 8:
        os.chdir(f'{pwd}/AI_C18/{rid}/')
        if "ENDO" in os.listdir():
            os.chdir(f'{pwd}/AI_C18/{rid}/ENDO/')
            img_list = os.listdir()
            for img in img_list:
                name = img[:-3] + "txt"
                if name in annot_list:
                    sh.copy(f'{pwd}/AI_C18/{rid}/ENDO/{img}', f'{pwd}/AI_C18/normal/{img}')
                else:
                    sh.copy(f'{pwd}/AI_C18/{rid}/ENDO/{img}', f'{pwd}/AI_C18/abnormal/{img}')
        else:
            pass
    else:
        pass
