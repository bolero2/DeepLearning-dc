import argparse
import logging
import math
import sys
import os
import random
import yaml
import time
import cv2

import numpy as np
import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
from collections import OrderedDict


img_shape = (3, 28, 28)
#pytorch image shape : (batch_size, channel, height, width)

"""
Generator : 작은 Array에서 de-conv2d를 통해 원본 수준의 이미지를 생성해냄.
Discriminator : Generator가 생성해낸 이미지(라고 부르는 것)가 거짓인지 진짜인지 판별해냄.
    이 때, real-loss와 fake-loss가 존재함. 
"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)

        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()


if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()
    network = Model()

    print(generator)
    print(discriminator)
    print(network)
    # print(out)