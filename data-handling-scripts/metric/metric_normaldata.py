"""
Materials :
    1) Test Images
    2) Ground Truth Annotation files
    3) Detection Result Annotation files
"""
import os
import cv2
import pandas as pd
import glob
from decimal import Decimal as dec
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
import datetime


def extract_file_name(image_path, gt_path, dt_path):
    image_file_list = list()
    gt_file_list = list()
    detect_file_list = list()

    os.chdir(image_path)
    for image in glob.glob('*.jpg'):
        image_file_list.append(image)

    os.chdir(gt_path)
    for gt_label in glob.glob('*.txt'):
        gt_file_list.append(gt_label)

    os.chdir(dt_path)
    for detect_label in glob.glob('*.txt'):
        detect_file_list.append(detect_label)

    return image_file_list, gt_file_list, detect_file_list


def make_csv(image_path, gt_path, dt_path, conf_threshold=0.1,
             csv_save_path=None, csv_save_name=None, sorting_index=0):
    count_TN = 0
    dt_count = 0

    csv_list1 = list()
    csv_list2 = list()

    image_file_list, gt_file_list, detect_file_list = extract_file_name(image_path, gt_path, dt_path)

    ###########################################
    # get Total Object count
    ###########################################
    for dt in detect_file_list:
        try:
            dt_file = open(dt_path + dt, 'r')
            dt_lines = dt_file.readlines()
            dt_count += len(dt_lines)
            dt_file.close()
        except:
            print("Can't read detection result file!")
    print("   ============== START ==============")
    print(" TITLE : C18 YOLOv5 Model -> Normal Dataset inference\n\n")
    print(f"Now Date and Time : {datetime.datetime.now()}\n\n"
          f"Number of total normal images : {len(image_file_list)}\n"
          f"Number of detected Objects Count : {dt_count}\n"
          f"Confidence score Threshold : {conf_threshold}\n")

    # Reading Detection(Inference) file(.txt)
    for image_index in image_file_list:
        real_name = image_index[:-4]

        ###########################################
        # Read annotation file
        ###########################################
        try:
            dt_file = open(f'{dt_path}{real_name}.txt', 'r')
        except:
            # if there isn't detection result file -> TN +1 (because of normal data)
            csv_list1.append([f'{real_name}.txt', -1, 1])
            continue
        dt_line_list = dt_file.readlines()
        # Reading 1-line in one Detection(Inference) file(.txt)

        is_passed = True
        conf_score = 0

        for dt_line in dt_line_list:
            dt_line = dt_line.split(' ')
            if dt_line[-1] == '\n':
                dt_line[-2] = dt_line[-2] + dt_line[-1]
                del dt_line[-1]
            class_dt = dt_line[0]
            conf_score = float(dt_line[1])
            if conf_score > conf_threshold:     # TN -> lower than conf_threshold
                is_passed = False
                break
            else:
                is_passed = True

        if is_passed is True:
            csv_list1.append([f'{real_name}.txt', conf_score, 1])
        elif is_passed is False:
            csv_list1.append([f'{real_name}.txt', conf_score, 0])

    if sorting_index == 0:
        csv_list1 = csv_list1
    else:
        csv_list1 = sorted(csv_list1, key=lambda x: x[sorting_index])

    for rows in range(len(csv_list1)):
        val_TN = csv_list1[rows][2]
        count_TN += val_TN

        csv_list2.append([count_TN])

    df1 = pd.DataFrame(csv_list1, columns=['filename', 'confidence', 'TN'])
    df2 = pd.DataFrame(csv_list2, columns=['sum TN'])
    df_total = pd.concat([df1, df2], axis=1)  # column bind

    specificity = round(count_TN / len(image_file_list), 2)

    print(f"Total TN Count: {count_TN}\n")
    print(f"[Specificity] (expression: TN={count_TN} / normal data count={len(image_file_list)}): {specificity}\n")

    if csv_save_path is not None:
        if csv_save_name is not None:
            print(f"Save file : {csv_save_path + csv_save_name}.csv")
            df_total.to_csv(csv_save_path + csv_save_name + ".csv", index=False)
        else:
            print(f"Save file : {csv_save_path}result.csv")
            df_total.to_csv(csv_save_path + "result.csv", index=False)
    else:
        if csv_save_name is not None:
            print(f"Save file : ./{csv_save_path}.csv")
            df_total.to_csv(csv_save_name + ".csv", index=False)
        else:
            print(f"Save file : ./result.csv")
            df_total.to_csv("result.csv", index=False)

    print("   ============== END ==============\n")


if __name__ == "__main__":
    ###########################################
    # Single experiment result
    ###########################################
    # image_path = "C:/Users/bolero/Desktop/metric_dc/idc_c16_cancer/detectoRS/test_images_c16_normal/"
    # gt_path = "C:/Users/bolero/Desktop/metric_dc/idc_c16_cancer/detectoRS/test_labels_c16_normal/"
    # # dt_path = f"D:\\Files\\works\\1+AICenter\\result\\detectoRS\\inference_xyrb_abs\\epoch{e}\\"
    # dt_path = "C:/Users/bolero/Desktop/metric_dc/idc_c16_cancer/detectoRS/results_idc_c16_mmdet_normal/"
    # gt_coord = 'xyrb'
    # gt_coord_type = 'abs'
    # dt_coord = 'xyrb'
    # dt_coord_type = 'abs'
    # csv_save_path = "C:/Users/bolero/Desktop/metric_dc/idc_c16_cancer/detectoRS/metric_idc_c16_mmdet_normal/"
    # sorting_index = 1
    #
    # # sorting index
    # # 0 = list not sorted
    # # 1 = confidence score
    # # 2 = IoU
    # conf_threshold = 0.3
    # csv_save_name = f'best_conf{conf_threshold}'
    # prc_save_name = f'best_conf{conf_threshold}'
    #
    # # csv_save_name = f"csv_epoch{e}_IoU{iou_threshold}"
    # # prc_save_name = f"prc_epoch{e}_IoU{iou_threshold}"
    #
    # make_csv(image_path=image_path,
    #          gt_path=gt_path,
    #          dt_path=dt_path,
    #          gt_coord=gt_coord,
    #          gt_coord_type=gt_coord_type,
    #          dt_coord=dt_coord,
    #          dt_coord_type=dt_coord_type,
    #          iou_threshold=iou_threshold,
    #          csv_save_path=csv_save_path,
    #          csv_save_name=csv_save_name,
    #          sorting_index=sorting_index)

    ###########################################
    # List of experiment results
    ###########################################
    conf_threshold = 0.25
    image_path = '/home/bolero/.dc/dl/yolov5-c18-rid/test_normal_dataset/'
    gt_path = '/home/bolero/.dc/dl/yolov5-c18-rid/test_normal_dataset/'
    dt_path = '/home/bolero/.dc/dl/yolov5-c18-rid/c18_rid_aug_inf_normal/labels/'
    csv_save_path = '/home/bolero/.dc/dl/yolov5-c18-rid/'
    sorting_index = 1

    # sorting index
    # 0 = list not sorted
    # 1 = confidence score
    # 2 = IoU

    csv_save_name = f"best_csv_normaldata_c18_rid_aug_inf_normal"

    make_csv(image_path=image_path,
             gt_path=gt_path,
             dt_path=dt_path,
             conf_threshold=conf_threshold,
             csv_save_path=csv_save_path,
             csv_save_name=csv_save_name,
             sorting_index=sorting_index)
    exit(0)
