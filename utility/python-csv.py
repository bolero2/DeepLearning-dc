import csv

f = open("C:\\dataset\\MPII\\mpii_human_pose_v1_annot\\mpii_dataset.csv", 'r', encoding='utf-8')
rdr = csv.reader(f)
f2 = open("C:\\dataset\\MPII\\mpii_human_pose_v1_annot\\mpii_annot.txt", 'a')
f3 = open("C:\\dataset\\MPII\\mpii_human_pose_v1_annot\\class.names", 'a')
# label count = 16
label = ['r_ankle', 'r_knee', 'r_hip',
         'l_hip', 'l_knee', ' l_ankle',
         'pelvis', 'thorax', 'upper_neck', 'head_top',
         'r_wrist', 'r_elbow', 'r_shoulder',
         'l_shoulder', 'l_elbow', 'l_wrist']
size = 16
for line in rdr:
    print(f'file name = {line[1]}')
    label_count = 0
    f2.write("C:/dataset/MPII/images/" + line[1])
    f2.write(' ')
    for c in range(1, size + 1):
        f2.writelines(','.join([line[c * 2], line[c * 2 + 1], line[c * 2], line[c * 2 + 1], str(label_count)]))
        f2.write(" ")
        label_count += 1
    f2.write("\n")

for cc in range(0, 16):
    f3.write(label[cc])
    f3.write("\n")
f.close()
f2.close()
f3.close()