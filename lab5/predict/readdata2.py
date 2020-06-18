import numpy as np
from task2_util import *
import cv2
import math
import random

train_path = "./train50.txt"
test_path = "./test50.txt"

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
            #t1 = (float(a[1][2:]) - data_min) / (data_max - data_min)
            #t2 = (float(a[2][2:]) - data_min) / (data_max - data_min)
            #t3 = (float(a[3][2:]) - data_min) / (data_max - data_min)
        train_his.append((x, int(row[0])))
#print(train_his)


# read file in test
test_his = []
with open(test_path, "r") as file_to_read:
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
            #t1 = (float(a[1][2:]) - data_min) / (data_max - data_min)
            #t2 = (float(a[2][2:]) - data_min) / (data_max - data_min)
            #t3 = (float(a[3][2:]) - data_min) / (data_max - data_min)
            test_his.append((x, int(row[0])))

#print(test_his)
# K-means cluster for sift

# Vector Quantization

for k in range(1,11):   #testing on different k
    iter_ = 0           #counter of image
    right = 0           #counter of right prediction
    Result = list()
    for (test_row, Class_) in test_his:
        iter_ += 1
        neighbors = GetNeighbors(train = train_his, test_row = test_row, num_neighbors = k)
        output_values = [row for row in neighbors]
        prediction = max(set(output_values), key = output_values.count)
        if prediction == Class_:
            right += 1
        else:
            pass
            #print("wrong, prediction = ", prediction)
            #print("class = ", Class_)

    Result.append((k, right / (float(iter_))))
    print(k, iter_, right, right / float(iter_))





    