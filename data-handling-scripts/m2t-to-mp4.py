import cv2
import numpy

target = "videos/azure_test1.m2t"
convert_name = "videos/azure_test1.mp4"

cap = cv2.VideoCapture(target)
ret, frame = cap.read()
    
width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
running_time = total_frame / fps

print("\n===== Video Information =====\n",
      f"Total Frame Count: {total_frame}\n",
      f"FPS: {fps}\n",
      f"Video running time: (about) {round(running_time, 2)}s\n"
      f"Width: {width} | Height: {height}\n")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(convert_name, fourcc, fps, (width, height))

while ret:
    writer.write(frame)
    cv2.imshow("frame window", frame)
    key = cv2.waitKey(10)

    if key == ord('q'):
        break
    ret, frame = cap.read()

cap.release()
writer.release()
cv2.destroyAllWindows()
