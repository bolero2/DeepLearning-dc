#!/bin/bash

# Reference from here:
# https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130

sudo apt update
sudo add-apt-repository ppa:graphics-drivers


sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

sudo apt update
sudo apt install cuda-10-1
sudo apt install libcudnn7


# Add this lines in ~/.bashrc
# ==============================================================================================
# set PATH for cuda 10.1 installation
# if [ -d "/usr/local/cuda-10.1/bin/" ]; then
#     export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
#     export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# fi
# ==============================================================================================

# reboot system.
