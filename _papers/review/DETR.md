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
> 2. **Transformer** Encoder + Decoder
> 3. **F**eed-**F**oward **N**etwork(FFN)

이렇게 3단계로 구분할 수 있습니다.  

### 1) Convolution Neural Network

CNN의 주 목적은 입력 영상 데이터의 _**특징 추출**_ 입니다.  
논문에서 사용한 CNN(=Backbone)은 ResNet으로,  
**3ch * W * H** 영상 데이터가 입력으로 들어온 후 > 최종 **2048ch * W/32 * H/32** 크기의 Feature Map을 생성합니다.  

### 2) Transformer Encoder + Decoder

CNN을 거쳐 생성된 Feature Map은 1x1 convolution을 통해 1차원 채널로 축소됩니다.  
> _** Encoder는 Sequence Data를 입력으로 받기 때문에, Vectorizing함을 알 수 있습니다.**_

