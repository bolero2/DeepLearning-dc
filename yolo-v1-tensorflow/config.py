image_Width = 448
image_Height = 448
channel = 3

label_size = 20     # pascal VOC 2012 Dataset

grid = 7
batchsize = 1
Learning_Rate = 0.00001

box_per_cell = 2        # one cell have 2 box

boundary1 = grid * grid * label_size  # 7 * 7 * 20

boundary2 = boundary1 + grid * grid * box_per_cell  # 7 * 7 * 20 + 7 * 7 *2

TrainDir = "C:\\dataset\\VOC2012\\JPEGImages\\"  # 300,000 images
TrainDir_Annot = "C:\\dataset\\VOC2012\\Annotations\\"
# TrainDir = "/content/drive/My Drive/voc2012/VOC2012/JPEGImages/"  # 300,000 images
# TrainDir_Annot = "/content/drive/My Drive/voc2012/VOC2012/Annotations/"

# The names of this variables(=ModelDir, ModelName) must come from the script name.
ModelName = "4lab_detection1"
ModelDir = "D:\\0+2020ML\\1+Saver\\" + ModelName + "\\"
# ModelDir = "/content/drive/My Drive/" + ModelName + "/"

Filenames_Eval = []
Filenames_Train = []

index_train = 0
index_eval = 0

ForEpoch = 40

label_full = ['person', 'bird', 'cat', 'cow', 'dog',
              'horse', 'sheep', 'aeroplane', 'bicycle', 'boat',
              'bus', 'car', 'motorbike', 'train', 'bottle',
              'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

label_size = 20     # => len(label_full)
Total_Train = 17125

relu_alpha = 0.5

object_scale = 1.0
noobject_scale = 1.0
class_scale = 2.0
coord_scale = 5.0

keep_prob = 0.5
