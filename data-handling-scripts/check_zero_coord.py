path = "C:\\dataset\\MyDataset\\detectoRS_kvasir_ETIS\\label\\"
new_path = "C:\\dataset\\MyDataset\\detectoRS_kvasir_ETIS\\fixed_label\\"


import os


sentence_list = list()

files = os.listdir(path)
for ff in files:
    lines = open(path + ff, 'r').readlines()
    for line in lines:
        split_line = line.split(' ')
        # print(split_line)
        if split_line[1] == '0':
            split_line[1] = '1'
        if split_line[2] == '0':
            split_line[2] = '1'
        sentence_list.append(split_line[0] + ' ' + split_line[1] + ' ' + split_line[2] + ' ' + split_line[3] + ' ' + split_line[4])
    new_file = open(new_path + ff, 'w')
    new_file.writelines(sentence_list)
    new_file.close()
    sentence_list = list()
