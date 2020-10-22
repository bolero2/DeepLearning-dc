image_Width = 448
image_Height = 448
channel = 3

label_size = 20     # => len(label_full), pascal VOC 2012 Dataset

grid = 7
batchsize = 1
Learning_Rate = 0.00001

box_per_cell = 2        # one cell have 2 box
boundary1 = grid * grid * label_size  # 7 * 7 * 20
boundary2 = boundary1 + grid * grid * box_per_cell  # 7 * 7 * 20 + 7 * 7 *2

"""
About Training variables 
"""
Filenames_Eval = []
Filenames_Train = []

index_train = 0
index_eval = 0

ForEpoch = 40

TrainDir = "C:\\dataset\\VOC2012\\JPEGImages\\"  # 300,000 images
TrainDir_Annot = "C:\\dataset\\VOC2012\\Annotations\\"
# TrainDir = "/content/drive/My Drive/voc2012/VOC2012/JPEGImages/"  # 300,000 images
# TrainDir_Annot = "/content/drive/My Drive/voc2012/VOC2012/Annotations/"

# The names of this variables(=ModelDir, ModelName) must come from the script name.
ModelName = "4lab_detection1"
ModelDir = "C:\\1+works\\2+Python\\1+Saver\\" + ModelName + "\\"
# ModelDir = "D:\\0+2020ML\\1+Saver\\" + ModelName + "\\"
# ModelDir = "/content/drive/My Drive/" + ModelName + "/"

Total_Train = 17125

"""
if you want to load pretrained-weights, then you should input ckpt file's path
"""
weight_file = None
# weight_file = "C:\\1+works\\2+Python\\1+Saver\\4lab_detection1\\4lab_detection1_Epoch_1.ckpt"

label_full = ['person', 'bird', 'cat', 'cow', 'dog',
              'horse', 'sheep', 'aeroplane', 'bicycle', 'boat',
              'bus', 'car', 'motorbike', 'train', 'bottle',
              'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

relu_alpha = 0.5

object_scale = 1.0
noobject_scale = 1.0
class_scale = 2.0
coord_scale = 5.0

keep_prob = 0.5

threshold = 0.2
iou_threshold = 0.5
