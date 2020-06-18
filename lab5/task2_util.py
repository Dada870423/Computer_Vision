import numpy as np
import math
import cv2
import os, sys
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
    print("normalizing")
    for ImgClass in Img_list:
        img, Class_ = ImgClass
        temp = [ pixel for row in img for pixel in row ]
        #normalize
        norm_img = np.zeros(img.size)
        Nmlz = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
        Result.append( (Nmlz, Class_) )
    return Result
def Euclidean_distance(row1, row2):
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
        dist = Euclidean_distance(row1 = test_row, row2 = train_row)
        distances.append((Class_, dist))
    distances.sort(key = lambda y: y[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors



def k_means(data, k, threas):
    '''
    input data, k number, error threashold
    output k centers
    '''
    data_dimention = data.shape[1]
    center = np.random.choice(data.shape[0], k, replace=False)
    #print(center)
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
                new_distance = Euclidean_distance(pt, c)
                if new_distance < min_distance:
                    classIdx = idx
                    min_distance = new_distance
            cluster[classIdx].append(pt)
        #print('k_cluster')
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
                new_error += Euclidean_distance(center[idx], point)          
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
            dis = Euclidean_distance(kp, c)
            if dis < min_distance:
                lable = idx
                min_distance = dis
            histogram[lable] += 1    
    return histogram

def predict_histogram(descriptor, model, center):
    '''
    input: descriptor of a picture, k-cluster model
    output: histogram of the picture
    '''
    histogram = np.zeros(center.shape[0])
    for kp in descriptor:
        label = model.predict(kp.reshape(1, -1))  
        histogram[label] += 1
    #print(histogram)
    return histogram
