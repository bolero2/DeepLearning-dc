import argparse
import logging
import math
import sys
import os
import random
import yaml
import time

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

from datasets import CustomDataset


class SRNet(nn.Module):
    def __init__(self, name, ch, num_classes, setting=None, job=None):
        super(SRNet, self).__init__()
        # =========================== Setting ============================
        self.job = job

        self.yaml = setting

        if self.job is not None:
            self.num_classes = self.job.get_num_of_class()
            self.category_names = self.job.get_name_of_class()
        else:       # default setting -> else if ... "job is None"
            self.num_classes = num_classes
            self.category_names = [str(x) for x in range(0, self.num_classes)]

        self.conv_layers = list()
        self.flatten = list()
        self.fc_layers = list()

        # ======================== get layer info ========================
        self.name = name
        layerset = self.yaml[self.name]
        fcset = self.yaml['fc_layer']
        self.ch = ch

        # ======================= Model Definition =======================
        for block in layerset:
            for layer_output in block:
                self.conv_layers += [nn.Conv2d(self.ch, layer_output, kernel_size=3, padding=1), 
                                      nn.BatchNorm2d(layer_output), 
                                      nn.ReLU(inplace=True)]
                self.ch = layer_output
            self.conv_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.conv_layers += [nn.AdaptiveAvgPool2d((7, 7))]

        self.flatten += [nn.Flatten()]

        last_block = 0
        for block in range(0, len(fcset)):
            if block == 0:
                self.fc_layers += [nn.Linear(512 * 7 * 7, fcset[block])]
            else:
                self.fc_layers += [nn.Linear(fcset[block - 1], fcset[block])]
            last_block = fcset[block]

        self.fc_layers += [nn.Linear(last_block, self.num_classes)]

        self.total_layers = self.conv_layers + self.flatten + self.fc_layers

        self.model = nn.Sequential(*self.total_layers)
        # ================================================================
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        for layer in self.model:
            x = layer(x)

        return x

    def fit(self, x, y, validation_data, epochs=30, batch_size=4, callbacks=[]):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        imgsz = self.job.model.get_target_size()

        my_getmetric = callbacks[0]
        my_savebestweight = callbacks[1]

        trainpack = (x, y)
        validpack = validation_data

        train_dataset = CustomDataset(trainpack, imgsz=imgsz)
        valid_dataset = CustomDataset(validpack, imgsz=imgsz)

        total_train_iter = math.ceil(len(x) / batch_size)
        total_valid_iter = math.ceil(len(validation_data[0]) / batch_size)
        # print(train_dataset)

        trainloader = DataLoader(train_dataset,
                                 batch_size=batch_size, 
                                 num_workers=4, 
                                 shuffle=False,
                                 pin_memory=True)

        validloader = DataLoader(valid_dataset, 
                                 # batch_size=int(batch_size / 2), 
                                 batch_size=batch_size,
                                 num_workers=4, 
                                 shuffle=False,
                                 pin_memory=True)

        optimizer = optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)
        self = self.to(device)
        
        for epoch in range(0, epochs):
            start = time.time()
            train_loss, train_acc = 0, 0
            self.train()
            # Training Part
            print(f"[Epoch {epoch + 1}/{epochs}] Start")
            for i, (img, label) in enumerate(trainloader):
                optimizer.zero_grad()
                img = Variable(img.to(device))
                label = Variable(label.to(device, dtype=torch.int64))

                out = self(img)
                acc = (torch.max(out, 1)[1].cpu().numpy() == torch.max(label, 1)[1].cpu().numpy())
                acc = float(np.count_nonzero(acc) / batch_size)
                loss = self.criterion(out, torch.max(label, 1)[1])
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += acc
                print("[train %s/%3s] Epoch: %3s | Time: %6.2fs | loss: %6.4f | Acc: %g" % (
                        i + 1, total_train_iter, epoch + 1, time.time() - start, round(loss.item(), 4), float(acc)))

            train_loss = train_loss / total_train_iter
            train_acc = train_acc / total_train_iter
            print("[Epoch {} training Ended] > Time: {:.2}s/epoch | Loss: {:.4f} | Acc: {:g}\n".format(
                epoch + 1, time.time() - start, np.mean(train_loss), train_acc))

            val_loss, val_acc = self.evaluate(model=self, dataloader=validloader, valid_iter=total_valid_iter, batch_size=batch_size);

            # (dc) GetMetricCallback ==================================================================================
            my_getmetric.save_status(epoch + 1, 
                                     metrics=[[round(train_loss, 4), round(train_acc, 4)],   # train metric
                                              [round(val_loss, 4), round(val_acc, 4)]],      # validation metric
                                     metrics_name=[['loss', 'acc'],                 # train metric name
                                                   ['loss', 'acc']]                 # validation metric name
                                    )   
            # =========================================================================================================

            # (dc) SaveBestWeightCallback =============================================================================
            my_savebestweight.save_best_weight(self, model_metric=val_loss, compared='less')
            # =========================================================================================================

            # print(train_loss, train_acc, val_loss, val_acc)

    def evaluate(self, model, dataloader, valid_iter=1, batch_size=1):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(device)

        model.eval()
        start = time.time()
        total_loss, total_acc = 0, 0
        for i, (img, label) in enumerate(dataloader):
            img = Variable(img.to(device))
            label = Variable(label.to(device, dtype=torch.int64))

            out = model(img)

            loss = self.criterion(out, torch.max(label, 1)[1])
            acc = (torch.max(out, 1)[1].cpu().numpy() == torch.max(label, 1)[1].cpu().numpy())
            acc = float(np.count_nonzero(acc) / batch_size)

            total_loss += loss
            total_acc += acc

            print("[valid {}/{}] Time: {:.2}s | loss: {} | Acc: {}".format(
                i + 1, valid_iter, time.time() - start, round(loss.item(), 4), float(acc)))
        total_loss = total_loss / valid_iter
        total_acc = total_acc / valid_iter

        return total_loss.item(), total_acc
        
    def predict(self, test_images):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self = self.to(device)
        self.eval()
    
        count = test_images.shape[0]
        result_np = []
        
        for idx in range(0, count):
            # print(idx)
            img = test_images[idx, :, :, :]
            img = np.expand_dims(img, axis=0)
            img = torch.Tensor(img).permute(0, 3, 1, 2).to(device)
            # print(img.shape)
            pred = self(img)
            pred_np = pred.cpu().detach().numpy()
            for elem in pred_np:
                result_np.append(elem)
        return result_np
            

def num(s):
    """ 3.0 -> 3, 3.001000 -> 3.001 otherwise return s """
    s = str(s)
    try:
        int(float(s))
        return s.rstrip('0').rstrip('.')
    except ValueError as e:
        return s