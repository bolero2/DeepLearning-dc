import cv2
import os
import glob


pwd = os.getcwd()
os.chdir(pwd)

files = [x for x in glob.glob('*.png')]
# print(files)

count = 0
for f in files:
    img = cv2.imread(f)
    print(f'target image = {f} ------> ', end='')
    cv2.imwrite(f[:-3] + "jpg", img)
    print(f'{f[:-3]}jpg')
    count += 1

print(f'total image count: {len(files)}, converted image count: {count}')
