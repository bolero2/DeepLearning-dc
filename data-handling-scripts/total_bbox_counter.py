import os

file_list = open("C:\\dataset\\MyDataset\\totaltest\\val.txt", 'r').readlines()
annot_path = "C:\\dataset\\MyDataset\\totaltest\\total_label-3000_relat_ccwh\\"
print(file_list)

target_count = 0

for file in file_list:
    f = open(annot_path + file[:-1] + ".txt", 'r')
    lines = f.readlines()
    target_count += len(lines)
    # print(lines)

print(target_count)