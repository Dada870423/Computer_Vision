import numpy as np
import cv2
import math
import random
from task2_util import *
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

train_path = "./hw5_data/train/"
train_list = ReadFile(Path = train_path)
test_path = "./hw5_data/test/"
test_list = ReadFile(Path = test_path)

train_list = Resize_Normal(Img_list = train_list)
test_list  = Resize_Normal(Img_list = test_list)

n_clusters = 20

mode = "debug"
'''
debug: try sklearn kmeans
'''
print("mode: ", mode)
print("vocabulary size:",  n_clusters)
# find kp
print("finding keypoints of training data")
train_kp = []
for i in train_list:
    img, img_class = i
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(img, None)
    train_kp.append(descriptors)
random.shuffle(train_kp)
train_kp = np.concatenate(train_kp, axis=0)    

# K-means cluster for sift
print('K-means cluster')
if mode == "debug": #use sklearn
    print("try sklearn kmeans")
    #skl_kmeans = KMeans(n_clusters=200, random_state=0).fit(train_kp)
    skl_kmeans = MiniBatchKMeans(n_clusters = n_clusters , random_state = 0, batch_size = 7000).fit(train_kp)
    #skl_kmeans = skl_kmeans.partial_fit(train_kp)
    center = skl_kmeans.cluster_centers_ #cluster_centers_
    print(center)

else:
    center = k_means(data=train_kp, k=200, threas=1000)

# Vector Quantization
print("generating histogram of train data")
if mode == "debug":
    train_histogram = []
    for i in train_list:
        img, img_class = i
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(img, None)
        train_histogram.append((predict_histogram(descriptors, skl_kmeans, center), img_class))
else:
    train_histogram = []
    for i in train_list:
        img, img_class = i
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(img, None)
        train_histogram.append((build_histogram(descriptors, center), img_class))
random.shuffle(train_histogram)        

# write train data
data = open("train20.txt",'w+')
for i in range(0, len(train_histogram)):
    s = str(train_histogram[i][1])
    for j in range(n_clusters):
        s = s +" " + str(j+1) + ":" + str(train_histogram[i][0][j])
    print(s, file=data)
data.close()

# testing
print("generating histogram of test data")
test_histogram = []
for i in test_list:
    img, img_class = i
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(img, None)
    if mode == "debug":
        test_histogram.append((predict_histogram(descriptors, skl_kmeans, center), img_class))
    else:
        test_histogram.append((build_histogram(descriptors, center), img_class))
random.shuffle(test_histogram) 

# write test data
data = open("test20.txt",'w+')
for i in range(0, len(test_histogram)):
    s = str(test_histogram[i][1])
    for j in range(n_clusters):
        s = s +" " + str(j+1) + ":" + str(test_histogram[i][0][j])
    print(s, file=data)
data.close()

print("do knn")
if mode == "debug":
    '''
    for each test print prediction and answer
    '''
    right = 0
    iter = 0
    for (test_row, Class_) in test_histogram:
        iter += 1
        neighbors = GetNeighbors(train = train_histogram, test_row = test_row, num_neighbors = 3)
        output_values = [row for row in neighbors]
        prediction = max(set(output_values), key = output_values.count)
        if prediction == Class_:
            right += 1
            print("right, class = ", Class_)
        else:
            print("wrong, prediction = ", prediction)
            print("class = ", Class_)
    print("accuracy", right / (float(iter)))
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
    print("accuracy", right / (float(iter)))