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


class Generator(nn.Module):
    def __init__(self, input_size=int(28 * 28 * 3), image_size=[28, 28, 3], batch_size=4):
        super(Generator, self).__init__()
        # if len(input_size) == 3:
        #     self.ih, self.iw, self.ic = input_size
        # elif len(input_size) == 2:
        #     self.ih, self.iw = input_size
        #     self.ic = 1 
        self.batch_size = batch_size
        self.input_size = input_size
        self.image_size = image_size

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, self.input_size, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.image_size))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.image_size)
        return img

    # def forward(self, x):

    #     return output


class Discriminator(nn.Module):
    def __init__(self, image_size=[28, 28, 3], batch_size=4):
        super(Discriminator, self).__init__()
        if len(image_size) == 3:
            self.ih, self.iw, self.ic = image_size
        elif len(image_size) == 2:
            self.ih, self.iw = image_size
            self.ic = 1
        self.batch_size = batch_size
        # self.input_size = input_size
        self.image_size = image_size

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.image_size)), 512),
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
        
    # def forward(self, x):

    #     return output


class Model(nn.Module):
    def __init__(self, input_size=[7, 7, 128], image_size=[28, 28, 3], batch_size=4):
        super(Model, self).__init__()
        self.input_size = input_size
        self.image_size = image_size
        self.batch_size = batch_size

        print(self.input_size, self.image_size)
        self.generator = Generator(input_size, image_size, batch_size)
        self.discriminator = Discriminator(image_size, batch_size)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.generator = self.generator.to(self.device)
            self.discriminator = self.discriminator.to(self.device)
        
        print(self.generator, "\n", self.discriminator)

    def forward(self, x):
        out1 = self.generator(x)
        out2 = self.discriminator(out1)

        return out1, out2

if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()
    network = Model()

    print(generator)
    print(discriminator)
    print(network)
    # print(out)