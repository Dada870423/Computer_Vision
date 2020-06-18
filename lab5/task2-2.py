import numpy as np
from task_function import *
import cv2
import math
import random
from sklearn.cluster import KMeans

train_path = "./hw5_data/train/"
train_list = ReadFile(Path = train_path)
test_path = "./hw5_data/test/"
test_list = ReadFile(Path = test_path)

# Normalize picture
print("Normalize")
train_list = Normal(Img_list = train_list)
test_list  = Normal(Img_list = test_list)

mode = "all"
'''
debug: only do clustering on first 1000 kp, generate histogram on 1st picture
'''
# find kp
print("finding keypoints of training data")
train_kp = []
for i in train_list:
    img, img_class = i
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(img, None)
    train_kp.append(descriptors)
train_kp = np.concatenate(train_kp, axis=0)    

# K-means cluster for sift
print('K-means cluster')
if mode == "debug":
    skl_kmeans = KMeans(n_clusters=3, random_state=0).fit(train_kp)
    center = skl_kmeans.cluster_centers_
    print(center)
else:
    center = k_means(train_kp, 3, 1000)

# Vector Quantization
print("generating histogram of train data")
if mode == "debug":
    #img, img_class = train_list[0]
    #sift = cv2.xfeatures2d.SIFT_create()
    #kp, descriptors = sift.detectAndCompute(img, None)
    #histogram = build_histogram(descriptors, center)
    #print(histogram, img_class)
    train_histogram = []
    for i in train_list:
        img, img_class = i
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(img, None)
        train_histogram.append((build_histogram(descriptors, center), img_class))
else:
    train_histogram = []
    for i in train_list:
        img, img_class = i
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(img, None)
        train_histogram.append((build_histogram(descriptors, center), img_class))
        

# write train data
data = open("train2.txt",'w+')
for i in range(0, len(train_histogram)):
    s = str(train_histogram[i][1])+' 1:'+str(train_histogram[i][0][0])+' 2:'+str(train_histogram[i][0][1])+' 3:'+str(train_histogram[i][0][2])
    print(s, file=data)
data.close()

# testing
print("finding keypoints of test data")
test_kp = []
for i in test_list:
    img, img_class = i
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(img, None)
    test_kp.append(descriptors)
test_kp = np.concatenate(test_kp, axis=0)

print("generating histogram of testing data")
test_histogram = []
for i in test_list:
    img, img_class = i
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(img, None)
    test_histogram.append((build_histogram(descriptors, center), img_class))


# write test data
data = open("test2.txt",'w+')
for i in range(0, len(test_histogram)):
    s = str(test_histogram[i][1])+' 1:'+str(test_histogram[i][0][0])+' 2:'+str(test_histogram[i][0][1])+' 3:'+str(test_histogram[i][0][2])
    print(s, file=data)
data.close()

print("do knn")
if mode == "debug":
    '''
    test 1st test picture
    '''
    (test_row, Class_) = test_histogram[0]
    neighbors = GetNeighbors(train = histogram, test_row = test_row, num_neighbors = 3)
    output_values = [row for row in neighbors]
    prediction = max(set(output_values), key = output_values.count)
    if prediction == Class_:
        print("right, class = ", Class_)
    else:
        print("wrong, prediction = ", prediction)
        print("class = ", Class_)
else:
    right = 0
    iter = 0
    for (test_row, Class_) in test_histogram:
        iter += 1
        neighbors = GetNeighbors(train = train_histogram, test_row = test_row, num_neighbors = 3)
        output_values = [row for row in neighbors]
        prediction = max(set(output_values), key = output_values.count)
        if prediction == Class_:
            right += 1
    print("accuracy", right/float(iter))
