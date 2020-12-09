import os
import glob


def rename_dataset(ext, path, new_path=None, front_tag=''):
    os.chdir(path)
    print("os -> Change directory : ", path)
    file_list = list()
    for file in glob.glob('*.' + ext):
        file_list.append(file)
    num = len(file_list)
    print("The number of files :", num)
    os.chdir(path)
    for i in range(len(file_list)):
        number = str(i + 1).zfill(len(str(num)))
        if new_path is not None:
            os.rename(path + file_list[i], new_path + front_tag + str(number) + '.' + ext)
        else:
            os.rename(path + file_list[i], path + front_tag + str(number) + '.' + ext)


if __name__ == "__main__":
    """
    # You should think about !!! BACK UP !!! of dataset.
    """
    path = "C:\\dataset\\MedicalDataset\\ETIS-LaribPolypDB\\groundtruth\\"
    # new_path = "C:\\dataset\\MedicalDataset\\ETIS-LaribPolypDB\\aug_image\\"
    front_tag = "gt_ETIS_"

    rename_dataset(ext='jpg', path=path, front_tag=front_tag)
    exit(0)
