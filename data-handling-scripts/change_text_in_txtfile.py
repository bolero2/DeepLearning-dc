import os

path = "C:\\dataset\\dataset+annotation_ndc\\"
new_path = "C:\\dataset\\annot_temp\\"
# num = 100

file_list = os.listdir(path)
print(f"{len(file_list)} -> file_list:{file_list}")

new_filelist = list()
new_lines = list()

# 텍스트 파일만 추출
for i in range(0, len(file_list)):
    if str(path + file_list[i])[-3:] == 'txt':
        new_filelist.append(str(path + file_list[i]))

# 텍스트 파일만 모인 리스트 변수
print(f"{len(new_filelist)} -> new_filelist:{new_filelist}\n\n")

# 리스트 변수에서 하나씩 선택 -> 텍스트 파일 하나를 읽는 동작
for i in range(0, len(new_filelist)):
    f1 = open(new_filelist[i], 'r')
    lines = f1.readlines()  # 텍스트 파일의 내용 읽어들임 -> n줄(인덱스로 구분가능)
    print(f"file name = {new_filelist[i]}")
    print(f"lines={lines}")
    for index in range(0, len(lines)):
        new_line = str(0) + lines[index][1:]
        print(f"index[{index}] -> ", new_line, end='')
        new_lines.append(new_line)
    print("newfile:", new_path + new_filelist[i][-11:-8] + ".txt")
    f2 = open(new_path + new_filelist[i][-11:-8] + ".txt", "w")
    for c in range(0, len(new_lines)):
        f2.write(new_lines[c])
    new_lines = list()
    # f2 = open(new_path + new_filelist[i][0:4] + new_filelist[i][-3:])
