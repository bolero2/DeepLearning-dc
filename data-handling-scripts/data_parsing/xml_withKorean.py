import xml.etree.ElementTree as xml
import os
import glob

# data_path = "C:\\dataset\\MedicalDataset\\sample300-coco\\label\\"
# new_path = "C:\\dataset\\MedicalDataset\\sample300-yolo_text_minmax\\label\\"

data_path = "D:/Files/works/0+Development/Python/0+DNN/1+data_handling/data_parsing/xml_data/"
new_path = "C:/Users/bolero/Downloads/xmlparsingtemp/"

files = list()
os.chdir(data_path)
for file in glob.glob("*.xml"):
    # if file == "gastric_sample.xml":
    if file == "00003297.xml":
        files.append(file)

for file in files:
    # with Korean
    # f = open(data_path + file, 'r', encoding='euc-kr')
    f = open(data_path + file, 'r', encoding='utf-8')
    text = f.read()
    f.close()
    full_path = data_path + file
    print(f"XML File name: {full_path}")
    root = xml.fromstring(text)

    # without Korean
    # full_path = data_path + file
    # print(f"XML File name: {full_path}")
    # tree = xml.ElementTree(file=full_path)
    # root = tree.getroot()

    print(f"get root result: {root}")
    classes = list()

    ##########################
    # Case 1. Tag Type
    ##########################
    for p1 in root.iter(tag='Rows'):
        for p2 in p1.iter(tag='ST_STAGE'):
            print(p2.text)
        for p2 in p1.iter(tag='SN_STAGE'):
            print(p2.text)

    ##########################
    # Case 2. Attribute Type
    ##########################
    for p1 in root.iter(tag='annotations'):
        for p2 in p1.iterfind('image'):
            print(p2.attrib)
            for i in range(len(p2)):
                print(p2[i].attrib)

    # print(count)
        # for p2 in p1.iter(tag='image'):
            # print(p2.text)
