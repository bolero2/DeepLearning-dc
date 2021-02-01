import cv2
import random


images = ['bird.jpg', 'cat.jpg', 'frog.jpg', 'dog.jpg', 'deer.jpg', 'horse.jpg']
# images = ['bird.jpg', 'cat.jpg']

total_frame = 4800
fps = 20
width = 224
height = 224

writer = cv2.VideoWriter()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('static.mp4', fourcc, fps, (width, height))

count = 0
while count < total_frame:
    random.seed()
    target = random.randint(0, 5)
    print(f'count= {count}, now target= {images[target]}')
    for i in range(0, fps * 10):
        img = cv2.imread(images[target])
        img = cv2.resize(img, (width, height))
        writer.write(img)
        count += 1

writer.release()
