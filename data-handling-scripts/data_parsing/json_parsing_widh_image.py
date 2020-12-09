import json
import os

file = "C:/dataset/MedicalDataset/Kvasir-SEG/kavsir_bboxes.json"

image_path = "C:\\dataset\\MedicalDataset\\Kvasir-SEG\\test_image\\"
label_path = "C:\\dataset\\MedicalDataset\\Kvasir-SEG\\test_label\\"

image_list = os.listdir(image_path)
print(image_list)
sentence = ''
sentence_list = list()
with open(file) as json_file:
    json_data = json.load(json_file)

    for f in image_list:
        col = json_data[f[:-4]]['width']
        row = json_data[f[:-4]]['height']

        bbox = json_data[f[:-4]]['bbox']
        print(f"{f} file has {len(bbox)} bbox.")
        for b in bbox:
            xmin = b['xmin']
            ymin = b['ymin']
            xmax = b['xmax']
            ymax = b['ymax']
            center_x = float((xmin + float(abs(xmax - xmin) / 2)) / col)
            center_y = float((ymin + float(abs(ymax - ymin) / 2)) / row)
            width = float(abs(xmax - xmin) / col)
            height = float(abs(ymax - ymin) / row)

            # Remove dummy annotation
            if 0.01 < width <= 0.99 and 0.01 < height <= 0.99:
                sentence = "0 " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height) + "\n"
                sentence_list.append(sentence)
            else:
                print(f"======================================== {f} has dummy annotation!")
        annot = open(label_path + f[:-4] + ".txt", "w")
        annot.writelines(sentence_list)
        sentence_list = list()


