import xml.etree.ElementTree as xml
import os

data_path = '/home/bolero/.dc/dl/dataset/detection/instrument/tooldetection/m2cai16-tool-locations/Annotations/'
new_path = '/home/bolero/.dc/dl/dataset/detection/instrument/tooldetection/m2cai16-tool-locations/annot_dc/' 

files = os.listdir(data_path)
classes = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']

for file in files:
    full_path = f'{data_path}{file}'
    print("FILE NAME : ", full_path)

    f = open(new_path + file[:-3] + "txt", 'w')
    tree = xml.ElementTree(file=full_path)
    root = tree.getroot()
    label_list = list()
    coord = list()
    for elem in tree.iter(tag='object'):
        for label_name in elem.iter(tag='name'):
            label_num = classes.index(label_name.text)
        for i in elem.iter(tag='bndbox'):
            coord = [f'{label_num} {i.getchildren()[0].text} {i.getchildren()[1].text} {i.getchildren()[2].text} {i.getchildren()[3].text}\n']
            print(coord)
            f.write(coord[0])
    f.close()
