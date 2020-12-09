import cv2
import os
import numpy as np
import glob

image_list = list()


def lut(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def preprocess_ct_image(path, ext='jpg', is_save_path=""):
    os.chdir(path)
    for image in glob.glob(f'*.{ext}'):
        image_list.append(image)

    for i in image_list:
        print(f"Target image: {path + i}")
        img = cv2.imread(path + i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # step 1. Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(img, -1, kernel)

        # step 2. Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(filtered)
        cl1 = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)

        # step 3. Show or Save image file
        if len(is_save_path) > 0:    # save mode
            print(f"---> Saving image: {is_save_path + i}")
            cv2.imwrite(is_save_path + i, cl1)
        else:     # show mode
            cv2.imshow("New Image", cl1)
            cv2.imshow("Original Image", img)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                print("Script is terminated(ESCape)")
                break


if __name__ == "__main__":
    image_path = "C:/dataset/MedicalDataset/Colon_CT_annotaion/images/side/"
    ext = 'png'
    # is_save_path = "D:/"
    is_save_path = ''
    preprocess_ct_image(path=image_path, ext=ext, is_save_path=is_save_path)

    exit(0)
