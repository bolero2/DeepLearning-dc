# DETR - End-to-End Object Detection with Transformer

이번에 소개할 논문은 Facebook AI 팀에서 공개한  
Transformer 방식을 Computer Vision의 Object Detection 분야에 적용시킨 **DETR**입니다.

DETR은 **DE**tection + **TR**ansformer 의 줄임말로, 이름에서부터 Transformer가 Detection 방식에 사용됨을 유추할 수 있습니다.  

논문 제목에서 __End-to-End__ 라는 말의 의미는,  
(뒤에 등장하지만)기존 Detection Network가 가지고 있는 초매개변수(**Hyper-Parameter**, ex. NMS, threshold, anchor-box etc.)를  
Transformer의 End-to-End 방식의 학습을 통해 없앴다고 볼 수 있습니다.

--------

## 1. Abstract

논문에서 크게 주장하는 핵심은 다음과 같습니다:  
```
1. 사용자가 설정해야 하는 것(Hand-designed Components) 을 제거  
2. Simple한 Network 구성  
3. 이분법적 매칭(Bipartite Matching)과 Transformer의 Encoder-Decoder 구조 사용  
```
추가적으로, Object Detection 분야 뿐 만 아니라  
**Panoptic Segmentation(a.k.a Instance Segmentation)** 분야에서도 좋은 성능을 보여준다고 합니다.  

--------

## 2. Model Architecture

네트워크의 전체적인 구성은 다음과 같습니다:  

![model1](https://user-images.githubusercontent.com/41134624/105303525-e6fb9400-5bfe-11eb-947c-ef4939938df6.jpg)

해당 네트워크는 크게 본다면  
> 1. **C**onvolution **N**eural **N**etwork(ResNet)
> 2. **Transformer** Encoder
> 3. **Transformer** Decoder
> 4. **F**eed-**F**oward **N**etwork(FFN)

이렇게 4단계로 구분할 수 있습니다.  

### 1) Convolution Neural Network

CNN의 주 목적은 입력 영상 데이터의 _**특징 추출**_ 입니다.  
논문에서 사용한 CNN(=Backbone)은 ResNet으로,  
**3ch * W * H** 영상 데이터가 입력으로 들어온 후 > 최종 **2048ch * W/32 * H/32** 크기의 Feature Map을 생성합니다. 

![resnet](https://user-images.githubusercontent.com/41134624/105318677-262de300-5c07-11eb-983c-c26c68abe782.jpg)

저자는 Backbone CNN으로 ResNet50 모델을 사용하였는데,  
해당 모델의 맨 마지막 channel 깊이는 **2048**임을 알 수 있습니다.

### 2) Transformer Encoder

CNN을 거쳐 생성된 Feature Map은 1x1 convolution을 통해 **d 차원**(=d 채널)으로 축소됩니다.  
> * _**Encoder는 Sequence Data를 입력으로 받기 때문에, Vectorizing함을 알 수 있습니다.**_  
> * _**또한, 축소 된 d 채널은 Spatial하게 분리하여 H*W 크기로 구성된 d 개의 조각으로 분리할 수 있습니다.**_

각각의 d개 조각은 Encoder Layer의 입력으로 Sequencial하게 들어가며, Encoder Layer는 기본적인 구조로 구성되어 있습니다.  
> * _**Encoder Layer는 Multi-head Self-attention module로 구성되어 있습니다.**_

Encoder에서 살펴 볼 것은 다음과 같습니다:
```
* 원래 Transformer는 입력 데이터의 순서가 출력 데이터에 영향을 주지 않습니다.
* 하지만 Vision 문제에서는 분리 된 d개의 조각에 대한 순서가 중요하기 때문에 각각의 Attention Layer마다 Position Embedding을 실시니다.
```

### 3) Transformer Decoder

Decoder 역시 Encoder와 동일하게 Standard한 구조를 따릅니다.  
Encoder의 출력으로 d size의 N Embedding이 나오고, 이는 그대로 Decoder의 입력으로 들어갑니다.  

Decoder에서 살펴 볼 것은 다음과 같습니다:
```
* 원래의 Decoder는 분리 된 d 개의 조각을 하나의 Sequence로 보고, 통째로 입력 데이터로 들어갑니다.
* 하지만 DETR에서는 각각의 Decoder Layer마다 N 개의 Embedding 객체를 Parallel하게 Decoding합니다.
* 또한, Encoder처럼 각각의 Attention Layer에 Object Query를 추가하여 Position Embedding과 유사한 작업을 합니다.
```
