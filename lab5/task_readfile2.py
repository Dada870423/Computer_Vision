import numpy as np
#from task1 import ReadFile, GetNeighbors
import cv2
import math
import random
def GetNeighbors(train, test_row, num_neighbors):
    distances = list()
    for (train_row, Class_) in train:
        dist = euclidean_distance(row1 = test_row, row2 = train_row)
        distances.append((Class_, dist))
    distances.sort(key = lambda y: y[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def euclidean_distance(row1, row2):
    distance = 0.0
    if len(row1) != len(row2):
        print("not equal", len(row1), len(row2))
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)


filename = 'train617.txt'


train_histogram = []

with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行讀取資料
        if not lines:
            break
        a = lines.split()
        # normalize
        a_sum = (float(a[1][2:]) + (float(a[2][2:])) + (float(a[3][2:])))
        t1 = (float(a[1][2:]) / a_sum)
        t2 = (float(a[2][2:]) / a_sum)
        t3 = (float(a[3][2:]) / a_sum)
        train_histogram.append(([t1, t2, t3], int(a[0])))


filename = 'test617.txt'

            # 讀第二次存成tr_histogram
test_histogram = []

with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行讀取資料
        if not lines:
            break
        a = lines.split()
        # normalize
        a_sum = (float(a[1][2:]) + (float(a[2][2:])) + (float(a[3][2:])))
        t1 = (float(a[1][2:]) / a_sum)
        t2 = (float(a[2][2:]) / a_sum)
        t3 = (float(a[3][2:]) / a_sum)
        test_histogram.append(([t1, t2, t3], int(a[0])))



print("do knn")

right = 0
iter = 0
for (test_row, Class_) in test_histogram:
    iter += 1
    neighbors = GetNeighbors(train = train_histogram, test_row = test_row, num_neighbors = 3)
    output_values = [row for row in neighbors]
    prediction = max(set(output_values), key = output_values.count)
    if prediction == Class_:
        right += 1
print("accuracy", right / (float(iter)))
    