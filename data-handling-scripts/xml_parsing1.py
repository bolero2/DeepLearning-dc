import xml.etree.ElementTree as xml
import os

data_path = "C:\\dataset\\MedicalDataset\\sample300-coco\\label\\"
new_path = "C:\\dataset\\MedicalDataset\\sample300-yolo_text_minmax\\label\\"
files = os.listdir(data_path)
for file in files:
    # print(file[:-4] + ".txt")
    full_path = data_path + file
    print("FILE NAME : ", full_path)
    f = open(new_path + file[:-4] + ".txt", 'w')
    tree = xml.ElementTree(file=full_path)
    root = tree.getroot()
    for elem in tree.iter(tag='object'):
        # print(elem.tag, elem.attrib)
        for i in elem.iter(tag='bndbox'):
            coord = ['0 ' + i.getchildren()[0].text + ' ' + i.getchildren()[1].text + ' ' + i.getchildren()[2].text + ' ' + i.getchildren()[3].text + '\n']
            print(coord)
            f.write(coord[0])
            coord = []
            # print(i.getchildren()[0].text)
            # print(i.getchildren()[1].text)
            # print(i.getchildren()[2].text)
            # print(i.getchildren()[3].text)
        # print(f'label= {elem.getchildren()[0].text}')
        # print(f'xmin= {elem.getchildren()[4].getchildren()[0].text}')
        # print(f'ymin= {elem.getchildren()[4].getchildren()[1].text}')
        # print(f'xmax= {elem.getchildren()[4].getchildren()[2].text}')
        # print(f'ymax= {elem.getchildren()[4].getchildren()[3].text}\n')
    # lines = open(full_path).readlines()

