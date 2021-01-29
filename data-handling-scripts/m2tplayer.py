import cv2
import numpy


target = "test.mp4"

cap = cv2.VideoCapture(target)

ret, frame = cap.read()

total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
running_time = total_frame / fps

print("\n===== Video Information =====\n",
      f"Total Frame Count: {total_frame}\n",
      f"FPS: {fps}\n",
      f"Video running time: (about) {round(running_time, 2)}s\n"
      f"Width: {width} | Height: {height}\n")
    
while ret:
    cv2.imshow("Test", frame)
    key = cv2.waitKey(10)

    if key == ord('q'):
        break
    ret, frame = cap.read()
