import numpy as np
from task1 import ReadFile
import cv2
import math
import random

train_path = "./hw5_data/train/"
train_list = ReadFile(Path = train_path)
test_path = "./hw5_data/test/"
test_list = ReadFile(Path = test_path)

mode = "debug"
'''
debug: only do clustering on first 1000 kp, generate histogram on 1st picture
'''

def distance(point1, point2):
    dimension = len(point1)
    dist = 0.0
    for i in range(dimension):
        dist += (point1[i] - point2[i]) ** 2
    
    #print(dist)
    return math.sqrt(dist)

def k_means(data, k, threas):
    '''
    input data, k number, error threashold
    output k centers
    '''
    data_dimention = data.shape[1]
    center = np.random.choice(data.shape[0], k, replace=False)
    print(center)
    center = data[center]
    error = float("inf")
    abs_error = float("inf")
    while(abs_error > threas):
        # cluster 為有k個空list的list
        cluster = []
        for i in range(k):
            cluster.append([])   
        # classify pts
        for pt in data:
            classIdx = -1
            min_distance = float("inf")
            for idx, c in enumerate(center):
                #print(pt)
                #print(c)
                new_distance = distance(pt, c)
                if new_distance < min_distance:
                    classIdx = idx
                    min_distance = new_distance
            cluster[classIdx].append(pt)
        print('k_cluster')
        for C in cluster:
            print(len(C))
        cluster = np.array(cluster)
        
        # calculate new center & error
        new_error = 0
        for idx, C in enumerate(cluster):
            center[idx] = np.mean(C, axis=0)
            #print(idx, center[idx].shape)
            #print(center[idx])
            for point in C:
                new_error += distance(center[idx], point)          
        abs_error = abs(error - new_error)
        error = new_error
        print(abs_error)

    return center

def build_histogram(descriptor, center):
    '''
    input: descriptor of a picture, centers of k-cluster
    output: histogram of the picture
    '''
    histogram = np.zeros(center.shape[0])
    for kp in descriptor:
        label = -1
        min_distance = float("inf")
        for idx, c in enumerate(center):
            dis = distance(kp, c)
            if dis < min_distance:
                lable = idx
                min_distance = dis
            histogram[lable] += 1    
    return histogram

# find kp
print("finding keypoints")
train_kp =[]
for i in train_list:
    img, img_class = i
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(img, None)
    train_kp.append(descriptors)
train_kp = np.concatenate(train_kp, axis=0)    


# K-means cluster for sift
print('K-means cluster')
if mode == "debug":
    center = k_means(train_kp[:1000], 3, 50)
else:
    center = k_means(train_kp, 3, 1000)

# Vector Quantization
print("generating histogram")
if mode == "debug":
    img, img_class = train_list[0]
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(img, None)
    histogram = build_histogram(descriptors, center)
    print(histogram)
else:
    for i in train_list:
        img, img_class = i
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(img, None)
        build_histogram(descriptors, center)
    
    