import cv2
import numpy as np
import matplotlib.pyplot as plt
import config as cfg

image_size = cfg.image_Width

file = "C:\\dataset\\VOC2012\\JPEGImages\\2011_006205.jpg"


def image_read(imname, flipped=False):  # 读取图片
    image = cv2.imread(imname)
    original_h = image.shape[0]  # height = row
    original_w = image.shape[1]  # width = column(=col)
    print(original_w)
    image = cv2.resize(image, (image_size, image_size))  # resize大小
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = (image / 255.0) * 2.0 - 1.0
    print(f"image name = {imname}")
    plt.imshow(image)
    plt.show()
    if flipped:
        image = image[:, ::-1, :]

    return image


if __name__ == "__main__":
    image = image_read(file)
