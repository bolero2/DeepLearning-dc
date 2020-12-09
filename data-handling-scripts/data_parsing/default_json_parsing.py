import json
import os
import glob

data_path = "D:/Files/works/0+Development/Python/0+DNN/1+data_handling/data_parsing/json_data/"

files = list()
os.chdir(data_path)
for file in glob.glob("*.json"):
    files.append(file)

for file in files:
    full_path = data_path + file
    print(f"JSON File name: {full_path}")

    with open(full_path) as json_file:
        json_data = json.load(json_file)
        for attrib in range(len(json_data['images'])):
            print(" =================================================================")
            print(f"Name [{json_data['images'][attrib]['name']}]")
            print(f"Width [{json_data['images'][attrib]['width']}]")
            print(f"Height [{json_data['images'][attrib]['height']}]")
            # print(f"objects {json_data['images'][attrib]['objects']}")
            for attrib2 in range(len(json_data['images'][attrib]['objects'])):
                print(f"└---Label: [{json_data['images'][attrib]['objects'][attrib2]['label']}]\n"
                      f"    type: [{json_data['images'][attrib]['objects'][attrib2]['type']}]\n"
                      f"    position: [{json_data['images'][attrib]['objects'][attrib2]['position'].split(';')}]")