import os
import random

data_path = "C:\\dataset\\MedicalDataset\\sample300-pascal\\image\\"

if __name__ == "__main__":
    data_list = os.listdir(data_path)
    random.shuffle(data_list)

    train_list = list()
    val_list = list()
    count = 0
    val_rate = 0.1
    num_data = len(data_list)
    for i in data_list:
        if count >= num_data * (1 - val_rate):
            target = val_list
        else:
            target = train_list
        target.append(i[:-4] + "\n")
        count += 1

    print(len(train_list), train_list)
    print(len(val_list), val_list)

    train_f = open("C:\\dataset\\MedicalDataset\\sample300-pascal\\" + "train.txt", "w")
    val_f = open("C:\\dataset\\MedicalDataset\\sample300-pascal\\" + "val.txt", "w")

    for i in train_list:
        train_f.write(i)
    for i in val_list:
        val_f.write(i)

    train_f.close()
    val_f.close()