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


def floor(x, index):
    number = str(x)
    return float(number[:index + 2])


def find_in_list(list_x, elem):
    try:
        ret = list_x.index(elem)
    except:
        ret = -1

    return ret


def _is_box_intersect(box1, box2):
    if abs(box1[0] - box2[0]) < box1[2] + box2[2] and abs(box1[1] - box2[1]) < box1[3] + box2[3]:
        return True
    else:
        return False


def _get_area(box):
    return box[2] * box[3]


def _get_intersection_area(box1, box2):
    # intersection area
    return abs(max(box1[0], box2[0]) -
               min(box1[0] + box1[2], box2[0] + box2[2])) \
           * abs(max(box1[1], box2[1]) -
                 min(box1[1] + box1[3], box2[1] + box2[3]))


def _get_union_area(box1, box2, inter_area=None):
    area_a = _get_area(box1)
    area_b = _get_area(box2)
    if inter_area is None:
        inter_area = _get_intersection_area(box1, box2)

    return float(area_a + area_b - inter_area)


def iou(box1, box2):
    # if boxes dont intersect
    if _is_box_intersect(box1, box2) is False:
        # print("zero")
        return 0
    inter_area = _get_intersection_area(box1, box2)
    union = _get_union_area(box1, box2, inter_area=inter_area)
    # intersection over union
    iou = inter_area / union
    # print(f"iou: {iou}")
    assert iou >= 0

    return iou


def extract_file_name(image_path, gt_path, detect_path):
    image_file_list = list()
    gt_file_list = list()
    detect_file_list = list()

    os.chdir(image_path)
    for image in glob.glob('*.jpg'):
        image_file_list.append(image)

    os.chdir(gt_path)
    for gt_label in glob.glob('*.txt'):
        gt_file_list.append(gt_label)

    os.chdir(detect_path)
    for detect_label in glob.glob('*.txt'):
        detect_file_list.append(detect_label)

    return image_file_list, gt_file_list, detect_file_list


def convert_coordinate(coord, coord_type, shape, bbox):
    row = shape[0]
    col = shape[1]
    output = list()
    if coord == 'ccwh' and coord_type == 'relat':
        output = [bbox[0] * col - (bbox[2] * col / 2), bbox[1] * row - (bbox[3] * row / 2),
                  bbox[2] * col, bbox[3] * row]

    elif coord == 'ccwh' and coord_type == 'abs':
        output = [bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[2], bbox[3]]

    elif coord == 'xyrb' and coord_type == 'relat':
        output = [bbox[0] * col, bbox[1] * row, abs(bbox[0] * col - bbox[2] * col), abs(bbox[1] * row - bbox[3] * row)]

    elif coord == 'xyrb' and coord_type == 'abs':
        output = [bbox[0], bbox[1], abs(bbox[0] - bbox[2]), abs(bbox[1] - bbox[3])]

    elif coord == 'xywh' and coord_type == 'relat':
        output = [bbox[0] * col, bbox[1] * row, bbox[2] * col, bbox[3] * row]

    elif coord == 'xywh' and coord_type == 'abs':
        output = [bbox[0], bbox[1], bbox[2], bbox[3]]

    return output


def AP(precision_list, recall_list):
    start_index = 0
    total_area = 0
    for recall_index in range(len(recall_list) - 1):
        if recall_list[recall_index] == recall_list[recall_index + 1]:
            if start_index == 0:
                width = recall_list[recall_index]
            else:
                width = dec(str(recall_list[recall_index])) - dec(str(recall_list[start_index]))
            start_index = recall_index
            height = precision_list[recall_index]
            total_area = total_area + dec(str(width)) * dec(str(height))

    return total_area


def make_csv(image_path, gt_path, dt_path, gt_coord, gt_coord_type, dt_coord, dt_coord_type, iou_threshold=0.5,
             csv_save_path=None, csv_save_name=None, sorting_index=0):
    count_TP = 0
    count_FP = 0
    gt_count = 0
    dt_count = 0
    iou_val = 0

    csv_list1 = list()
    csv_list2 = list()

    graph_precision_list = list()
    graph_recall_list = list()
    ap_precision_list = list()
    ap_recall_list = list()

    image_file_list, gt_file_list, detect_file_list = extract_file_name(image_path, gt_path, dt_path)

    ###########################################
    # get Total Object count
    ###########################################
    for gt in gt_file_list:
        try:
            gt_file = open(gt_path + gt, 'r')
            gt_lines = gt_file.readlines()
            gt_count += len(gt_lines)
            gt_file.close()
        except:
            print("Can't read ground-truth file!")
    for dt in detect_file_list:
        try:
            dt_file = open(dt_path + dt, 'r')
            dt_lines = dt_file.readlines()
            dt_count += len(dt_lines)
            dt_file.close()
        except:
            print("Can't read detection result file!")
    print("   ============== START ==============")
    print(f"Number of total Objects Count : {gt_count}\n"
          f"Number of detected Objects Count : {dt_count}\n"
          f"IoU Threshold : {iou_threshold}\n")

    # Reading Detection(Inference) file(.txt)
    for detect_index in detect_file_list:
        real_name = detect_index[:-4]
        img = cv2.imread(image_path + real_name + ".jpg")
        row, col, ch = img.shape

        ###########################################
        # Read annotation file
        ###########################################
        dt_file = open(dt_path + detect_index, 'r')
        gt_file = open(gt_path + detect_index, 'r')
        dt_line_list = dt_file.readlines()
        gt_line_list = gt_file.readlines()
        # Reading 1-line in one Detection(Inference) file(.txt)

        number_TP = 0           # TP count in 1 detection result file
        tp_gt_same = False      # TP count == ground-truth boxes count?

        for dt_line in dt_line_list:
            toggle = False
            dt_line = dt_line.split(' ')
            if dt_line[-1] == '\n':
                dt_line[-2] = dt_line[-2] + dt_line[-1]
                del dt_line[-1]
            class_dt = dt_line[0]
            conf_score = float(dt_line[1])
            coord = list(map(float, dt_line[2:]))
            # print(coord)
            dt_coordinate = convert_coordinate(dt_coord, dt_coord_type, [row, col], coord)

            for gt_line in gt_line_list:
                if tp_gt_same is True:
                    break
                gt_line = gt_line.split(' ')
                if gt_line[-1] == '\n':
                    gt_line[-2] = gt_line[-2] + gt_line[-1]
                    del gt_line[-1]
                class_gt = gt_line[0]
                coord = list(map(float, gt_line[1:]))

                # if class_dt != class_gt:


                # change coordinate system into [xywh -> xmin, ymin, width, height] + [abs]
                gt_coordinate = convert_coordinate(gt_coord, gt_coord_type, [row, col], coord)

                # get IoU Value
                iou_val = float(iou(dt_coordinate, gt_coordinate))

                if iou_val > iou_threshold:
                    val_TP = 1
                    val_FP = 0
                    csv_list1.append([detect_index, class_gt, class_dt, conf_score, iou_val, val_TP, val_FP])
                    number_TP += 1
                    toggle = True
                    break

            if toggle is False:
                val_TP = 0
                val_FP = 1
                csv_list1.append([detect_index, str(-1), class_dt, conf_score, iou_val, val_TP, val_FP])

            if number_TP == len(gt_line_list):
                iou_val = -1
                tp_gt_same = True

    if sorting_index == 0:
        csv_list1 = csv_list1
    else:
        csv_list1 = sorted(csv_list1, key=lambda x: x[sorting_index], reverse=True)

    for rows in range(len(csv_list1)):
        val_TP = csv_list1[rows][5]
        val_FP = csv_list1[rows][6]
        count_TP += val_TP
        count_FP += val_FP
        precision = count_TP / (count_TP + count_FP)
        recall = count_TP / gt_count

        graph_precision_list.append(precision)
        graph_recall_list.append(recall)

        ap_precision_list.append(float((str(precision) + "00")[0:4]))
        ap_recall_list.append(float((str(recall) + "00")[0:4]))

        csv_list2.append([count_TP, count_FP, precision, recall])

    df1 = pd.DataFrame(csv_list1, columns=['filename', 'ground-truth class', 'detected class', 'confidence', 'IoU', 'TP', 'FP'])
    df2 = pd.DataFrame(csv_list2, columns=['sum TP', 'sum FP', 'precision', 'recall'])
    df_total = pd.concat([df1, df2], axis=1)  # column bind

    ap = AP(ap_precision_list, ap_recall_list)

    print(f"Total TP Count: {count_TP}\nTotal FP Count: {count_FP}\n\n"
          f"False Positive Rate per Image: {round(int(csv_list2[-1][1]) / len(gt_file_list), 3)}\n"
          f"Recall(=Sensitivity, True Positive Rate): {float(graph_recall_list[-1]):.2f}\n"
          f"Precision: {float(graph_precision_list[-1]):.2f}\n"
          f"AP: {str(ap * 100)[:5]}%\n")

    plt.plot(graph_recall_list, graph_precision_list, label='Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision x Recall Curve\nmAP: {str(ap * 100)[:5]}%')
    plt.ylim([float(str(graph_precision_list[-1])[0:3]) - 0.0225, 1.0225])
    plt.xlim([-0.05, 1.05])
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.show()

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

    if prc_save_name is None:
        plt.savefig(csv_save_path + 'precision-recall-curve.jpg')
    else:
        plt.savefig(csv_save_path + f'{prc_save_name}.jpg')

    print("   ============== END ==============\n")


if __name__ == "__main__":
    image_path = "C:/Users/bolero/Desktop/metric_dc/gt_images/"
    gt_path = "C:/Users/bolero/Desktop/metric_dc/gt_label_abs_xyrb/"
    # dt_path = f"D:\\Files\\works\\1+AICenter\\result\\detectoRS\\inference_xyrb_abs\\epoch{e}\\"
    dt_path = "C:/Users/bolero/Desktop/metric_dc/result_detectors_epoch4_conf0.001_label_abs_xyrb/"
    gt_coord = 'xyrb'
    gt_coord_type = 'abs'
    dt_coord = 'xyrb'
    dt_coord_type = 'abs'
    csv_save_path = "C:\\Users\\bolero\\Desktop\\metric_dc\\"
    sorting_index = 3

    # sorting index
    # 0 = list not sorted
    # 1 = confidence score
    # 2 = IoU
    iou_threshold = 0.3
    csv_save_name = f'epoch4_iou0.3'
    prc_save_name = f'epoch4_iou0.3'

    # csv_save_name = f"csv_epoch{e}_IoU{iou_threshold}"
    # prc_save_name = f"prc_epoch{e}_IoU{iou_threshold}"

    make_csv(image_path=image_path,
             gt_path=gt_path,
             dt_path=dt_path,
             gt_coord=gt_coord,
             gt_coord_type=gt_coord_type,
             dt_coord=dt_coord,
             dt_coord_type=dt_coord_type,
             iou_threshold=iou_threshold,
             csv_save_path=csv_save_path,
             csv_save_name=csv_save_name,
             sorting_index=sorting_index)
    # for e in range(1, 13):
    #     for i in range(3, 6):
    #         iou_threshold = round(float(i / 10), 1)
    #         image_path = "C:/Users/bolero/Desktop/metric_dc/hyuk_dt/final_test/"
    #         gt_path = "C:/Users/bolero/Desktop/metric_dc/hyuk_dt/final_g/"
    #         # dt_path = f"D:\\Files\\works\\1+AICenter\\result\\detectoRS\\inference_xyrb_abs\\epoch{e}\\"
    #         dt_path = "C:/Users/bolero/Desktop/metric_dc/hyuk_dt/final_d/"
    #         gt_coord = 'xywh'
    #         gt_coord_type = 'relat'
    #         dt_coord = 'xywh'
    #         dt_coord_type = 'relat'
    #         csv_save_path = "C:\\Users\\bolero\\Desktop\\temp\\detectoRS-results\\"
    #         sorting_index = 1
    #
    #         # sorting index
    #         # 0 = list not sorted
    #         # 1 = confidence score
    #         # 2 = IoU
    #
    #         csv_save_name = f"csv_epoch{e}_IoU{iou_threshold}"
    #         prc_save_name = f"prc_epoch{e}_IoU{iou_threshold}"
    #
    #         make_csv(image_path=image_path,
    #                  gt_path=gt_path,
    #                  dt_path=dt_path,
    #                  gt_coord=gt_coord,
    #                  gt_coord_type=gt_coord_type,
    #                  dt_coord=dt_coord,
    #                  dt_coord_type=dt_coord_type,
    #                  iou_threshold=iou_threshold,
    #                  csv_save_path=csv_save_path,
    #                  csv_save_name=csv_save_name,
    #                  sorting_index=sorting_index)
    exit(0)
