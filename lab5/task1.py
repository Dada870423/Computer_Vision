## Tiny images representation + nearest neighbor classifier

import numpy as np
import math
import cv2
import os, sys
from matplotlib import pyplot as plt

def euclidean_distance(row1, row2):
    distance = 0.0
    if len(row1) != len(row2):
    print("not equal", len(row1), len(row2))
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)


# Locate the most similar neighbors
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


def ReadFile(Path):
    ImgClass = []
    Dirs = [ d for d in os.listdir(Path) if not d.startswith(".") ]
    ImgNames = [ os.listdir(Path + d) for d in Dirs if not d.startswith(".") ]
    for Class_ in range( len(ImgNames) ): 
        for name in ImgNames[Class_]:
            if not name.startswith("."):
                img = cv2.imread(Path + Dirs[Class_] + "/" + name, cv2.IMREAD_GRAYSCALE)
                ImgClass.append( (img, Class_) )
    return ImgClass

def Resize_Normal(Img_list):
    Result = []
    for ImgClass in Img_list:
        img, Class_ = ImgClass

        Rsz = cv2.resize( img, dsize=(16, 16) )
        temp = [ pixel for row in Rsz for pixel in row ]
        #normalize
        Nmlz = [ float(pixel) / sum(temp) for pixel in temp ]
        
        Result.append( (Nmlz, Class_) )
    return Result



train_path = "./hw5_data/train/"
test_path  = "./hw5_data/test/"
## read file
train_list = ReadFile(Path = train_path)
test_list  = ReadFile(Path = test_path)

## resize & normal
RN_train_list = Resize_Normal(Img_list = train_list)
RN_test_list  = Resize_Normal(Img_list = test_list)

Result = []
## NN
for k in range(1,11):   #testing on different k
    iter_ = 0           #counter of image
    right = 0           #counter of right prediction

    for (test_row, Class_) in RN_test_list:
        iter_ += 1
        neighbors = GetNeighbors(train = RN_train_list, test_row = test_row, num_neighbors = k)
        output_values = [row for row in neighbors]
        prediction = max(set(output_values), key = output_values.count)
        if prediction == Class_:
            right += 1

    Result.append((k, right / (float(iter_))))
    print(k, iter_, right, right / float(iter_))

## plot
x = [p[0] for p in Result]
y = [p[1] for p in Result]
plt.plot(x, y, 'r+', label = "hit rate")
plt.ylim(0, 0.5)
plt.title("task1")
plt.show()
