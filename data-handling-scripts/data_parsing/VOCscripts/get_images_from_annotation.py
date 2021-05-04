import os
import glob
import shutil as sh


imagepath = "/home/neuralworks/dataset/VOCdevkit/VOC2012/JPEGImages/"
savepath = "/home/neuralworks/dataset/VOCdevkit/VOCdetection/"
annot_ext = "png"

os.chdir(imagepath)
imagefiles = glob.glob("*.jpg")
os.chdir(savepath + "annotations/")
annotfiles = glob.glob(f"*.{annot_ext}")

for imagefile in imagefiles:
    realname = os.path.basename(imagefile)[:-4]
    if realname + f".{annot_ext}" in annotfiles:
        sh.copy(imagepath + imagefile, savepath + "images/")
    else:
        pass
