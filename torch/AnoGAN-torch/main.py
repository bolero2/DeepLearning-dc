import os
from glob import glob
import cv2
import numpy as np
from dataset import CustomDataset
from model import Model

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable


if __name__ == "__main__":
    batch_size = 4
    epochs = 20
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_images = glob('/home/neuralworks/dataset/mnist/0/*.jpg')
    valid_images = glob('/home/neuralworks/dataset/mnist/5/*.jpg')

    train_data = [[], []]
    valid_data = [[], []]

    for i in train_images:
        train_data[0].append(os.path.abspath(i))
        train_data[1].append(0)

    for i in valid_images:
        valid_data[0].append(os.path.abspath(i))
        valid_data[1].append(5)

    # print(train_data)
    imgsz = cv2.imread(train_data[0][0]).shape
    input_dim = imgsz[0] * imgsz[1] * imgsz[2] if len(imgsz) == 3 else imgsz[0] * imgsz[1]
    # print(imgsz)

    trainpack = tuple(train_data)
    validpack = tuple(valid_data)

    train_dataset = CustomDataset(trainpack, imgsz=imgsz)
    valid_dataset = CustomDataset(validpack, imgsz=imgsz)

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

    network = Model(input_size=input_dim, image_size=imgsz, batch_size=batch_size)

    if torch.cuda.is_available():
        network = network.to('cuda')

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    # Optimizers
    optimizer_G = torch.optim.Adam(network.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(network.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for epoch in range(epochs):
        for i, (img, label) in enumerate(trainloader):
            # Adversarial ground truths
            valid = Variable(Tensor(img.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(img.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_img = Variable(img.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (img.shape[0], 100))))

            # Generate a batch of images
            gen_img = network.generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(network.discriminator(gen_img), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(network.discriminator(real_img), valid)
            fake_loss = adversarial_loss(network.discriminator(gen_img.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            # if batches_done % opt.sample_interval == 0:
            #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            # if torch.cuda.is_available():
            #     img = img.to(device)
            #     label = label.to(device)

            # noise = torch.rand(size=(batch_size, input_dim))
            # noise = (-1 - 1) * noise
            # generated_image = network.generator(noise)
            # output = network(noise)

            # print(output)
            # print(generated_image)
            # exit(0)
            


