import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

TrainDir = "/home/clt_dc/dataset/classification/cifar10/train/"
EvalDir = "/home/clt_dc/dataset/classification/cifar10/eval/"

classes = os.listdir(TrainDir)

batch_size = 80
total_train = 50000
total_eval = 10000
index_train = 0
index_eval = 0

image_size = 224
label_size = len(classes)
channel = 3


def printProgress(iteration, total, prefix='', suffix='', bar_length=100):
    percent = str((iteration / total) * 100)[0:4]
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '■' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| [{percent}%] iteration {iteration}/{total}    {suffix}'),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def read_path():
    print("Reading path of dataset ... Start")
    train_buffer = list()
    eval_buffer = list()
    for class_name in tqdm(classes, desc="Reading path of training dataset", ncols=100):
        filelist = os.listdir(TrainDir + str(class_name))
        for j in range(len(filelist)):
            train_buffer.append([TrainDir + str(class_name) + '/' + filelist[j], classes.index(class_name)])
    random.shuffle(train_buffer)

    for class_name in tqdm(classes, desc="Reading path of evaluation dataset", ncols=100):
        filelist = os.listdir(EvalDir + str(class_name))
        for j in range(len(filelist)):
            eval_buffer.append([EvalDir + str(class_name) + '/' + filelist[j], classes.index(class_name)])
    random.shuffle(eval_buffer)
    print("Reading path of dataset ... End\n")

    return np.array(train_buffer), np.array(eval_buffer)


def load_image(filenames, type):
    print("Loading dataset in Array ... Start")
    images = filenames[:, 0]
    labels = filenames[:, 1]

    total_size = 0
    tqdm_sentence = ''

    if type == 'train':
        total_size = total_train
        tqdm_sentence = "training"
    elif type == 'eval':
        total_size = total_eval
        tqdm_sentence = "evaluation"

    image_buffer = np.zeros(shape=(total_size, image_size, image_size, channel), dtype=np.float32)
    label_buffer = np.zeros(shape=(total_size, label_size), dtype=np.int32)
    # images / 255.0
    # labels.astype('float32') or ('uint8'), don't care about label type

    for i in tqdm(range(filenames.shape[0]), desc=f'Loading {tqdm_sentence} dataset', ncols=100):
        image_buffer[i, :, :, :] = cv2.resize(cv2.imread(images[i]), (image_size, image_size))
        label_buffer[i, int(labels[i])] = 1.0     # one-hot encoding
    image_buffer = np.transpose(image_buffer, (0, 3, 1, 2))
    print("Loading dataset in Array ... End\n")

    return image_buffer, label_buffer


class CustomDataset(Dataset):
    def __init__(self, load_type):
        filenames_train, filenames_eval = read_path()
        if load_type == 'train':
            images, labels = load_image(filenames_train, type=load_type)
        elif load_type == 'eval':
            images, labels = load_image(filenames_eval, type=load_type)

        self.images = images
        self.labels = labels

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.images[idx])
        y = torch.LongTensor(self.labels[idx])

        return x, y


class _block_residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(_block_residual, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, padding_mode='zeros')
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if int(out.shape[-1]) != int(residual.shape[-1]):
            residual = self.conv1x1(residual)

        out = out + residual
        out = self.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode)
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=1, padding_mode='zeros')
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, padding=1)
        self.layer1 = torch.nn.Sequential(
            _block_residual(64, 64, kernel_size=3, stride=1),
            _block_residual(64, 64, kernel_size=3, stride=1),
            _block_residual(64, 64, kernel_size=3, stride=1),
        )
        self.layer2 = torch.nn.Sequential(
            _block_residual(64, 128, kernel_size=3, stride=2),
            _block_residual(128, 128, kernel_size=3, stride=1),
            _block_residual(128, 128, kernel_size=3, stride=1),
            _block_residual(128, 128, kernel_size=3, stride=1),
        )
        self.layer3 = torch.nn.Sequential(
            _block_residual(128, 256, kernel_size=3, stride=2),
            _block_residual(256, 256, kernel_size=3, stride=1),
            _block_residual(256, 256, kernel_size=3, stride=1),
            _block_residual(256, 256, kernel_size=3, stride=1),
            _block_residual(256, 256, kernel_size=3, stride=1),
            _block_residual(256, 256, kernel_size=3, stride=1),
        )
        self.layer4 = torch.nn.Sequential(
            _block_residual(256, 512, kernel_size=3, stride=2),
            _block_residual(512, 512, kernel_size=3, stride=1),
            _block_residual(512, 512, kernel_size=3, stride=1),
        )
        self.gap = nn.AvgPool2d(7)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, len(classes))
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nb_epochs = 10
    print(f"Device: {device}\n"
          f"Total Epoch: {nb_epochs}")

    dataset = CustomDataset(load_type='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"dataloader: {dataloader}")
    print(f"dataloader + id func.: {id(dataloader)}")

    network = ResNet34().to(device)

    loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)

    for epoch in range(nb_epochs + 1):
        iter = -1
        for batch_index, sample in enumerate(dataloader):
            x_train, y_train = sample

            # Upload Dataset in GPU device
            X = x_train.to(device, dtype=torch.float)
            Y = y_train.to(device, dtype=torch.long)
            optimizer.zero_grad()
            output = network(X)
            # print(output.size())
            cost = loss(output, torch.max(Y, 1)[1])
            cost.backward()
            optimizer.step()
            iter = iter + 1

            if iter % 100 == 99 or iter == dataloader.__len__():
                print(f"Epoch {epoch + 1}/{nb_epochs}   "
                      f"Iteration {iter + 1}/{dataloader.__len__()}     "
                      f"Loss {str(float(cost))[0:7]}")

        PATH = f'./trained/epoch_{epoch}.pth'
        torch.save(network.state_dict(), PATH)

    print('Finished Training')


