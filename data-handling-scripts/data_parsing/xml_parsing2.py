import xml.etree.ElementTree as xml
import glob
import os

data_path = '/home/bolero/.dc/dl/dataset/detection/idc_cancer/new_endo/new_data/'
new_path = '/home/bolero/.dc/dl/dataset/detection/idc_cancer/new_endo/annot_yolo/' 

os.chdir(data_path)
files = [x for x in glob.glob('*.xml')]
print(files)
# classes = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']

for file in files:
    full_path = f'{data_path}{file}'
    print("FILE NAME : ", full_path)

    fname = new_path + file[:-3] + "txt"
    print(fname)
    f = open(fname, "w")
    tree = xml.ElementTree(file=full_path)
    root = tree.getroot()
    label_list = list()
    coord = list()
    for elem in tree.iter(tag='object'):
        for i in elem.iter(tag='bndbox'):
            print(i)
            coord = [f'0 {i.getchildren()[0].text} {i.getchildren()[1].text} {i.getchildren()[2].text} {i.getchildren()[3].text}\n']
            print(coord)
            f.write(coord[0])
    f.close()
