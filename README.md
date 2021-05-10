# DC Deep Learning(dcdl)
***Author** : Dae-Cheol, Noh*

***License** : MIT License ([Link](https://github.com/bolero2/DeepLearning-dc/blob/master/LICENSE))*

## Classification Model (*Updated 2020. 12. 09*)
**All of source codes are written by myself using python language.**
|TensorFlow2 (+ Keras)|PyTorch|
|:-----------:|:-----------:|
|[ResNet-34-tf2](https://github.com/bolero2/DeepLearning-dc/tree/master/tf2/ResNet-34-tf2)|[ResNet-34-torch](https://github.com/bolero2/DeepLearning-dc/tree/master/torch/ResNet-34-torch)|
|[ResNet-101-tf2](https://github.com/bolero2/DeepLearning-dc/tree/master/tf2/ResNet-101-tf2)|[VGGNet-torch](https://github.com/bolero2/DeepLearning-dc/tree/master/torch/VGGNet-torch)|
|[ResNet-152-tf2](https://github.com/bolero2/DeepLearning-dc/tree/master/tf2/ResNet-152-tf2)||
|[EfficientNet-tf2-keras](https://github.com/bolero2/DeepLearning-dc/tree/master/tf2/EfficientNet-tf2-keras)||
|[Cancer-stage-classification](https://github.com/bolero2/DeepLearning-dc/tree/master/tf2/stage_classification)||

## _papers
Folder link is [Here](https://github.com/bolero2/DeepLearning-dc/tree/master/_papers)
|Date|Name|Link|
|:-----------:|:-----------|:-----------:|
|2018. 12|이중흐름 3차원 합성곱 신경망 구조를 이용한 효율적인 손 제스처 인식 방법|[Link](https://github.com/bolero2/DeepLearning-dc/blob/master/_papers/%5B2018.12%5D%20%EC%9D%B4%EC%A4%91%ED%9D%90%EB%A6%84%203%EC%B0%A8%EC%9B%90%20%ED%95%A9%EC%84%B1%EA%B3%B1%20%EC%8B%A0%EA%B2%BD%EB%A7%9D%20%EA%B5%AC%EC%A1%B0%EB%A5%BC%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%ED%9A%A8%EC%9C%A8%EC%A0%81%EC%9D%B8%20%EC%86%90%20%EC%A0%9C%EC%8A%A4%EC%B2%98%20%EC%9D%B8%EC%8B%9D%20%EB%B0%A9%EB%B2%95.pdf)|
|2018. 12|[KSC2018] 실시간 손 제스처 인식을 위한 덴스넷 기반 이중흐름 3차원 합성곱 신경망 구조|[Link](https://github.com/bolero2/DeepLearning-dc/blob/master/_papers/%5BKSC2018%5D%20%EC%8B%A4%EC%8B%9C%EA%B0%84%20%EC%86%90%20%EC%A0%9C%EC%8A%A4%EC%B2%98%20%EC%9D%B8%EC%8B%9D%EC%9D%84%20%EC%9C%84%ED%95%9C%20%EB%8D%B4%EC%8A%A4%EB%84%B7%20%EA%B8%B0%EB%B0%98%20%EC%9D%B4%EC%A4%91%ED%9D%90%EB%A6%84%203%EC%B0%A8%EC%9B%90%20%ED%95%A9%EC%84%B1%EA%B3%B1%20%EC%8B%A0%EA%B2%BD%EB%A7%9D%20%EA%B5%AC%EC%A1%B0.pdf)|
|2018. 12|**[KSC2018] Award Confirmation(수상 확인서)**|[Link](https://github.com/bolero2/DeepLearning-dc/blob/master/_papers/Award_confirmation_KSC2018_20190215.pdf)|
|2019. 12|Atrous Convolution과 Grad-CAM을 통한 손 끝 탐지|[Link](https://github.com/bolero2/DeepLearning-dc/blob/master/_papers/%5B2019.12%5D%20Atrous%20Convolution%EA%B3%BC%20Grad-CAM%EC%9D%84%20%ED%86%B5%ED%95%9C%20%EC%86%90%20%EB%81%9D%20%ED%83%90%EC%A7%80.pdf)|
|2020. 06|실시간 손끝 탐지를 위한 VGGNet 기반 객체 탐지 네트워크|[Link](https://github.com/bolero2/DeepLearning-dc/blob/master/_papers/%5B2020.06%5D%20%EC%8B%A4%EC%8B%9C%EA%B0%84%20%EC%86%90%EB%81%9D%20%ED%83%90%EC%A7%80%EB%A5%BC%20%EC%9C%84%ED%95%9C%20VGGNet%20%EA%B8%B0%EB%B0%98%20%EA%B0%9D%EC%B2%B4%20%ED%83%90%EC%A7%80%20%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC.pdf)|

## _docker (*for hadoop*)
**docker section was opened!!!**  
* *Github* foler Link is [Here](https://github.com/bolero2/DeepLearning-dc/tree/master/_docker)  
* *Docker* hub Link is [Here](https://hub.docker.com/repository/docker/sheocjf1025/bolero-hadoop)


## _old version
### MobileNet-v2-tensorflow.py
Date : 2019. 01. 15



### ShuffleNet-v1-tensorflow.py
Date : 2019. 01. 22



### Image_Augmentation.py
Date : 2019. 02. 25

--- Types of dataset ---
1. Original + flip
2. left-top crop
3. left-top crop + flip
4. left-bottom crop
5. left-bottom crop + flip
6. right-bottom crop
7. right-bottom crop + flip
8. right-top crop
9. right-top crop + flip
10. center crop
11. center crop + flip



### 3D_VGG16.py
Date : 2019. 04. 16



### 3D_NUMBER_Demo_1
Date : 2019. 04. 16

Version : version. 1

Used Network : VGG16

Dataset : 3D Hand Gesture, Drawing Number (0~9, 10EA)

