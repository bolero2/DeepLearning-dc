import os
import glob


def rename_dataset(exts, path, new_path=None, front_tag=''):
    os.chdir(path)
    print("OS :  Change directory : ", path)

    if type(exts).__name__ == "str":    # entered only one extension type.
        exts = list(exts)

    for ext in exts: 
        file_list = list()
        for filename in glob.glob('*.' + ext):
            file_list.append(filename)

        num = len(file_list)
        print("The number of files :", num)

        for i in range(len(sorted(file_list))):
            number = str(i + 1).zfill(len(str(num)))
            if new_path is not None:
                os.rename(path + file_list[i], new_path + front_tag + str(number) + '.' + ext)
            else:
                os.rename(path + file_list[i], path + front_tag + str(number) + '.' + ext)


if __name__ == "__main__":
    """
    # You should think about !!! BACK UP !!! of dataset.
    """
    path = '/home/bolero/Downloads/export/'
    front_tag = "racoon_"

    rename_dataset(exts=['jpg', 'xml'], path=path, front_tag=front_tag)
    exit(0)
