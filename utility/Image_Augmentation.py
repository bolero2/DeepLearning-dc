import cv2
import os
import copy

srcDir = "D:\\Tiny_ImageNet\\Tiny_ImageNet\\Train\\"       # Location of Original Data
dstDir = "D:\\Tiny_ImageNet\\Tiny_ImageNet\\Train\\"

crop_percent = 0.8

"""
1. Original + flip
2. left-top crop
3. left-top crop + flip
4. left-bottom crop
5. left-bottom crop + flip
6. right-bottom crop
7. right-bottom crop + flip
8. right-top crop
9. right-top crop + flip
10. center crop
11. center crop + flip
"""

setN = 0

dirList = os.listdir(srcDir)    # Reading label directory name(ex. '1', '2', '3', '4', ..., '98', '99', '100')
print(dirList)

cropnum = 0

for dir_num in dirList:
    srcList = os.listdir(srcDir + dir_num)

    for i in srcList:
        value = cv2.imread(srcDir + dir_num + "\\" + i)
        value2 = copy.deepcopy(value)
        row, col, ch = value.shape   # row = height / col = width
        new_row = round(row * crop_percent)
        new_col = round(col * crop_percent)
        filename, ext = i.split('.')

        # Original + Flip
        savename = filename + "+flip" + '.' + ext
        print(savename)
        cv2.imwrite(dstDir + dir_num + "\\" + savename, cv2.flip(value2, flipCode=1))

        cropnum = cropnum + 1

        # left-top crop
        value2 = copy.deepcopy(value)
        savename = filename + "+crop" + str(cropnum) + '.' + ext
        cv2.imwrite(dstDir + dir_num + "\\" + savename, value2[0:new_row, 0:new_col, :])

        # left-top crop + flip
        savename = filename + "+crop" + str(cropnum) + "+flip" + '.' + ext
        cv2.imwrite(dstDir + dir_num + "\\" + savename, cv2.flip(value2[0:new_row, 0:new_col, :], flipCode=1))

        cropnum = cropnum + 1

        # left-bottom crop
        value2 = copy.deepcopy(value)
        savename = filename + "+crop" + str(cropnum) + '.' + ext
        cv2.imwrite(dstDir + dir_num + "\\" + savename, value2[row - new_row:row, 0:new_col, :])

        # left-bottom crop + Flip
        savename = filename + "+crop" + str(cropnum) + "+flip" + '.' + ext
        cv2.imwrite(dstDir + dir_num + "\\" + savename, cv2.flip(value2[row - new_row:row, 0:new_col, :], flipCode=1))

        cropnum = cropnum + 1

        # right-bottom crop
        value2 = copy.deepcopy(value)
        savename = filename + "+crop" + str(cropnum) + '.' + ext
        cv2.imwrite(dstDir + dir_num + "\\" + savename, value2[row - new_row:row, col - new_col:col, :])

        # right-bottom crop + Flip
        savename = filename + "+crop" + str(cropnum) + "+flip" + '.' + ext
        cv2.imwrite(dstDir + dir_num + "\\" + savename,
                    cv2.flip(value2[row - new_row:row, col - new_col:col, :], flipCode=1))

        cropnum = cropnum + 1

        # right-top crop
        value2 = copy.deepcopy(value)
        savename = filename + "+crop" + str(cropnum) + '.' + ext
        cv2.imwrite(dstDir + dir_num + "\\" + savename, value2[0:new_row, col - new_col:col, :])

        # right-top crop + Flip
        savename = filename + "+crop" + str(cropnum) + "+flip" + '.' + ext
        cv2.imwrite(dstDir + dir_num + "\\" + savename,
                    cv2.flip(value2[0:new_row, col - new_col:col, :], flipCode=1))

        cropnum = cropnum + 1

        # center crop
        value2 = copy.deepcopy(value)
        savename = filename + "+crop" + str(cropnum) + '.' + ext
        cv2.imwrite(dstDir + dir_num + "\\" + savename, value2[row - new_row:new_row, col - new_col:new_col, :])

        # center crop + Flip
        savename = filename + "+crop" + str(cropnum) + "+flip" + '.' + ext
        cv2.imwrite(dstDir + dir_num + "\\" + savename,
                    cv2.flip(value2[row - new_row:new_row, col - new_col:new_col, :], flipCode=1))

        cropnum = 0
