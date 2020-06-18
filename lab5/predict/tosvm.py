import numpy as np
from task2_util import *
import cv2
import math
import random

train_path = "./train/train50.txt"
test_path = "./test/test50.txt"

n_cluster = 51

# read file in train
train_his = []
with open(train_path, "r") as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行讀取資料
        if not lines:
            break
        row = lines.split()
        # normalize
        x = list()
        for i in range(1, n_cluster):
            colon = row[i].split(":")
            x.append(float(colon[1]))
            train_his.append((x, int(row[0])))
#print(train_his)
feature = open("filetrain50.txt",'w+')
class_file = open("fileclass.txt",'w+')
for (row, Class_) in train_his:
    s = ""
    for j in range(len(row)):
        s = s + " " + str(row[j])
    print(s, file = feature)
    print(Class_, file = class_file)
feature.close()
class_file.close()






    