## 1. preprocess_ct_image.py  
Modify parameters:  
```  
image_path = "C:/dataset/MedicalDataset/Colon_CT_annotaion/images/side/"  
ext = 'png'  
# is_save_path = "D:/"  # if you want to save converted image  
is_save_path = ''       # if you don't want to save converted image  
```  


## 2. draw_bbox.py
Modify parameters:  
```  
image_path = "C:/Users/bolero/Desktop/temp/gt_images/"  
gt_label_path = "C:/Users/bolero/Desktop/temp/gt_label_abs_xyrb/"  
dt_label_path = "D:/Files/works/1+AICenter/result/detectoRS/inference_xyrb_abs/epoch9/"  
gt_coord = 'xyrb'         # 1. ccwh   2. xywh   3. xyrb  
gt_coord_type = 'abs'     # 1. relat  2. abs  
dt_coord = 'xyrb'         # 1. ccwh   2. xywh   3. xyrb  
dt_coord_type = 'abs'     # 1. relat  2. abs  
is_confidence = True      # if there is confidence score in detection result text files  
```  

