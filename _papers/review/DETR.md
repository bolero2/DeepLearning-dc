# DETR - End-to-End Object Detection with Transformer

이번에 소개할 논문은 Facebook AI 팀에서 공개한  
Transformer 방식을 Computer Vision의 Object Detection 분야에 적용시킨 **DETR**입니다.

DETR은 **DE**tection + **TR**ansformer 의 줄임말로, 이름에서부터 Transformer가 Detection 방식에 사용됨을 유추할 수 있습니다.  

논문 제목에서 __End-to-End__ 라는 말의 의미는,  
(뒤에 등장하지만)기존 Detection Network가 가지고 있는 초매개변수(**Hyper-Parameter**, ex. NMS, threshold, anchor-box etc.)를  
Transformer의 End-to-End 방식의 학습을 통해 없앴다고 볼 수 있습니다.

--------

## 1. Abstract

논문에서 크게 주장하는 바는 다음과 같습니다:  

```
1. 사용자가 설정해야 하는 것(Hand-designed Components) 을 제거  
2. Simple한 Network 구성  
3. 이분법적 매칭(Bipartite Matching)과 Transformer의 Encoder-Decoder 구조 사용  
```
추가적으로, Object Detection 분야 뿐 만 아니라  
**Panoptic Segmentation(a.k.a Instance Segmentation)** 분야에서도 좋은 성능을 보여준다고 합니다.  
