FROM ubuntu:20.04

MAINTAINER DC Noh <sheocjf1025@gmail.com>
MAINTAINER bolero2 <https://github.com/bolero2>

LABEL bolero-ubuntu and hadoop

SHELL ["/bin/bash", "-c"]

RUN apt-get -y update \
&& apt-get install -y --no-install-recommends git \
net-tools \
vim \
htop \
apt-utils \
sudo \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get -y upgrade

RUN apt-get clean

RUN adduser --disabled-password --gecos '' bolero \
&& echo 'bolero:bolero' | chpasswd \
&& adduser bolero sudo \
&& echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER bolero
WORKDIR /home/bolero

COPY ./* /home/bolero/.hadoop_install/

WORKDIR /home/bolero/.hadoop_install
RUN sudo chmod 777 ./*.sh
RUN ./java_install-dc.sh
RUN ./hadoop_install-dc.sh
RUN ./add_path.sh

WORKDIR /home/bolero

CMD ["/bin/bash"]
