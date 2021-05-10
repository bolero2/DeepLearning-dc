import logging
import math

import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


"""
[Torch Data Class]
    -> torch.utils.data.Dataset 
        + torch.utils.data.DataLoader 두개로 구성됨.

1) Dataset 필수 구현 함수
    * __getitem__(self, index)
    * __len(self)

2) DataLoader 구현
    * 필요없음.
"""
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

        _img = torch.from_numpy(self.imgs[index]).permute(2, 0, 1)
        _label = torch.Tensor(self.annots[index])
        return _img, _label

    def __len__(self):
        return len(self.imgs)