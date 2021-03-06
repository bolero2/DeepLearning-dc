# -*- coding: utf-8 -*-

import os
import glob
import shutil
import cv2
import xmltodict
import json


CLASSES = ("aeroplane", "bicycle", "bird", 
           "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", 
           "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", 
           "sheep", "sofa", "train", "tvmonitor")


class convert_coordinate(object):
    def __init__(self, coordinate, img_shape, input_coord, input_type, output_coord, output_type):
        # img_shape = (height, width) 이미지의 세로/가로
        """
        Arguments:
        1) coordinate : bbox coordinate
        2) img_shape = (height, width)
        3) input_coord = ['ccwh', 'xyrb', 'xywh']
        4) input_type = ['relat', 'abs']
        5) output_coord = ['ccwh', 'xyrb', 'xywh']
        6) output_type = ['relat', 'abs']
        return:
        converted coordinate, self.result
        example:
        answer = convert_coordinate(bbox, img_shape=(480, 640), 
                                    input_coord='ccwh', input_type='relat',
                                    output_coord='xyrb', output_type='abs')
        """

        self.coord = [float(coordinate[0]), 
                      float(coordinate[1]), 
                      float(coordinate[2]), 
                      float(coordinate[3])]
        self.ih = img_shape[0]
        self.iw = img_shape[1]
        self.input_coord = input_coord
        self.input_type = input_type
        self.output_coord = output_coord
        self.output_type = output_type

        self.converted_type = self.convert_type(self.coord, self.input_type, self.output_type)
        self.converted_coord = self.convert_coord(self.converted_type, self.input_coord, self.output_coord)
        self.result = self.converted_coord
        
    def convert_type(self, coord, input_type, output_type):
        """
        coordinate type만 변경.
        1) relat(상대 좌표) to abs(절대 좌표)
        2) abs(절대 좌표) to relat(상대 좌표)
        """
        result = []

        if input_type != output_type:
            if input_type == 'relat' and output_type == 'abs':
                result = [coord[0] * self.iw, coord[1] * self.ih, coord[2] * self.iw, coord[3] * self.ih]
            elif input_type == 'abs' and output_type == 'relat':
                result = [coord[0] / self.iw, coord[1] / self.ih, coord[2] / self.iw, coord[3] / self.ih]

        elif input_type == output_type:
            result = coord

        return result

    def convert_coord(self, coord, input_coord, output_coord):
        """
        coordinate system을 변경.
        [대상 인자]
        1) ccwh : center_x, center_y, width, height
        2) xyrb : xmin, ymin, xmax, ymax
        3) xywh : xmin, ymin, width, height
        """
        result = []

        if input_coord != output_coord:
            if input_coord == 'ccwh':
                center_x = coord[0]
                center_y = coord[1]
                width = coord[2]
                height = coord[3]

                if output_coord == 'xywh':
                    result = [center_x - (width / 2), center_y - (height / 2), width, height]
                elif output_coord == 'xyrb':
                    result = [center_x - (width / 2),
                              center_y - (height / 2),
                              center_x + (width / 2),
                              center_y + (height / 2)]
            elif input_coord == 'xywh':
                xmin = coord[0]
                ymin = coord[1]
                width = coord[2]
                height = coord[3]

                if output_coord == 'ccwh':
                    result = [xmin + (width / 2), ymin + (height / 2), width, height]
                elif output_coord == 'xyrb':
                    result = [xmin, ymin, xmin + width, ymin + height]

            elif input_coord == 'xyrb':
                xmin = coord[0]
                ymin = coord[1]
                xmax = coord[2]
                ymax = coord[3]

                width = xmax - xmin
                height = ymax - ymin
                
                if output_coord == 'ccwh':
                    result = [xmin + (width / 2), ymin + (height / 2), width, height]
                elif output_coord == 'xywh':
                    result = [xmin, ymin, width, height]

        elif input_coord == output_coord:
            result = coord

        return result


if __name__ == "__main__":
    imgpath = "/home/neuralworks/dataset/VOCdevkit/VOC2012/JPEGImages/"
    annotpath = "/home/neuralworks/dataset/VOCdevkit/VOC2012/Annotations/"
    save_path = "/home/neuralworks/dataset/VOCdevkit/VOCdetection/annotations/"

    imglist = [x.split('/')[-1] for x in glob.glob(imgpath + '*.jpg')]
    annotlist = glob.glob(annotpath + "*.xml")
    print(f"image count: {len(imglist)}")
    print(f"annotation count: {len(annotlist)}")

    for i in annotlist:
        file = open(i, 'r').read()
        xmlfile = xmltodict.parse(file)
        data = json.loads(json.dumps(xmlfile))
        
        filename = data['annotation']['filename']
        iw = int(data['annotation']['size']['width'])
        ih = int(data['annotation']['size']['height'])
        print(f"Filename: {filename} | shape=({ih}, {iw})")

        object = data['annotation']['object']
        if not isinstance(object, list):
            object = [object]
        sentence = []
        for index in range(0, len(object)):
            classname = object[index]['name']
            label = str(CLASSES.index(object[index]['name']))
            xmin = float(object[index]['bndbox']['xmin'])
            ymin = float(object[index]['bndbox']['ymin'])
            xmax = float(object[index]['bndbox']['xmax'])
            ymax = float(object[index]['bndbox']['ymax'])
            bbox = [xmin, ymin, xmax, ymax]
            _bbox = convert_coordinate(bbox, (ih, iw), 'xyrb', 'abs', 'ccwh', 'relat')
            bbox = _bbox.result
            sentence.append(f"{label} {round(bbox[0], 5)} {round(bbox[1], 5)} {round(bbox[2], 5)} {round(bbox[3], 5)}\n")
        print(sentence)

        file = open(save_path + filename[:-3] + "txt", "w")
        file.writelines(sentence)
        file.close()
