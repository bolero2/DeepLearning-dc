import logging
import math

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(self, datapack, imgsz=(224, 224, 3)):
        self.th, self.tw, self.tc = imgsz  # target-height, target-width, target-channel

        self.imgs = datapack[0]
        self.annots = datapack[1]

    def __getitem__(self, index):
        img = self.imgs[index]
        # print(img.shape)
        # if img.shape != (self.th, self.tw, self.tc):
            # img = cv2.reshape

        if isinstance(img, str):
            img = cv2.imread(img)

        _img = torch.from_numpy(img).permute(2, 0, 1)
        _label = torch.Tensor(self.annots[index])
        return _img, _label

    def __len__(self):
        return len(self.imgs)