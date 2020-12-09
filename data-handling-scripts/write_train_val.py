import os
import random
import glob

path = "C:\\Users\\bolero\\Downloads\\polyps_test_kvasir300_relat_ccwh_integ\\"
text_path = "C:\\Users\\bolero\\Downloads\\polyps_test_kvasir300_relat_ccwh_integ\\"
ext = 'jpg'
val_rate = 1.0

os.chdir(path)
print("os -> Change directory : ", path)
file_list = list()

for file in glob.glob(f'*.{ext}'):
    file_list.append(file)

print("File number :", len(file_list))
random.shuffle(file_list)

sentence_train = list()
sentence_val = list()

for i in range(0, len(file_list)):
    if i < len(file_list) * (1 - val_rate):
        sentence_train.append(file_list[i][:-4] + '\n')
    else:
        sentence_val.append(file_list[i][:-4] + '\n')

f1 = open(text_path + "train.txt", 'w')
f2 = open(text_path + "val.txt", 'w')

for sent in sentence_train:
    # new_sent = '/content/drive/My Drive/DeepLearning/Dataset/Detection/no_polyps_test_kvasir300_relat_ccwh_integ' + sent[:-1] + ".jpg\n"
    new_sent = f'/home/clt_dc/dataset/detection/detectoRS_ct_deeplesion/{sent[:-1]}.{ext}\n'
    # new_sent = sent
    f1.write(new_sent)

for sent in sentence_val:
    # new_sent = '/content/drive/My Drive/DeepLearning/Dataset/Detection/no_polyps_test_kvasir300_relat_ccwh_integ' + sent[:-1] + ".jpg\n"
    new_sent = f'/home/clt_dc/dataset/detection/detectoRS_ct_deeplesion/{sent[:-1]}.{ext}\n'
    # new_sent = sent
    f2.write(new_sent)

f1.close()
f2.close()
