import numpy as np
from numpy.linalg import inv
import cv2
import random

class Stitcher:

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        (imageB, imageA) = images
        # Doing SIFT, DetectAndCompute
        print("Doing SIFT on imgA")
        (keypointsA, featuresA) = self.detectAndDescribe(imageA)
        print("Doing SIFT on imgB")
        (keypointsB, featuresB) = self.detectAndDescribe(imageB)
        print("Done")

        # Match keypoints from two images, return Homography matrix
        Homo, match_pic = self.matchKeypoints(keypointsA, keypointsB, featuresA, featuresB, ratio, reprojThresh, imageA, imageB)


        result_pic = warp(imageA, Homo, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        
        # Find the leftest position of overlapped area
        print("Linear(alpha) Blending...")
        leftest_overlap = imageB.shape[1]
        for i in range(0,imageB.shape[0]):
            for j in range(0,imageB.shape[1]):
                if any(v != 0 for v in result_pic[i][j]):
                    leftest_overlap = min(leftest_overlap, j)
        # 將圖片B傳入左邊
        for i in range(0,imageB.shape[0]):
            for j in range(0,imageB.shape[1]):
                if any(v != 0 for v in result_pic[i][j]): # overlapped pixel
                    # Linear(alpha) Blending
                    alpha = float(imageB.shape[1]-j)/(imageB.shape[1]-leftest_overlap)
                    result_pic[i][j] = (result_pic[i][j] * (1-alpha) +  imageB[i][j] * alpha).astype(int)
                else:
                    result_pic[i][j] = imageB[i][j]
        return result_pic, match_pic

    def detectAndDescribe(self, image):
        # Change colorful to gray
        #gray = (0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]).astype(np.uint8) # Y = 0.299R + 0.587G + 0.114B
        
        # SIFT create
        descriptor = cv2.xfeatures2d.SIFT_create()

        # 檢測SIFT特徵點並計算
        keypoints, features = descriptor.detectAndCompute(image, None)

        return keypoints, features

    def matchKeypoints(self, keypointsA, keypointsB, featuresA, featuresB, ratio, reprojThresh, imageA, imageB):
        '''
        match_list,dist_list,ratio_list = find_ratio(keypointsA,featuresA,keypointsB,featuresB)
        find_match(match_list,dist_list,ratio_list) # call by reference
        t1,t2,t3 = remove_trash(match_list,dist_list,ratio_list)    #match_list / dist_list / ratio_list
        match_obj_list = create_match_obj(t1,t2)
        matches = sorted(match_obj_list, key = lambda x:x.distance) 
        '''
        matches = bfmatch(featuresA, featuresB)
        matches = sorted(matches, key=lambda x: x.distance)
        keypoints = [keypointsA, keypointsB]
        correspondenceList = []
        for match in matches:
            (x1, y1) = keypoints[0][match.queryIdx].pt
            (x2, y2) = keypoints[1][match.trainIdx].pt
            correspondenceList.append([x1, y1, x2, y2])

        corrs = np.array(correspondenceList)

        # run ransac algorithm
        Homo, inliers = ransac(corrs, 10.0)
        match_pic = 0
        # Draw Matches by cv2.drawMatches, which is allowed by TA : tiny.cc/kim24y (HW3 Question List)
        match_pic = cv2.drawMatches(imageA,keypointsA,imageB,keypointsB,matches[:30], match_pic,flags=2)
        
        #Homo = RANSAC(matches,keypointsA,keypointsB)

        return Homo, match_pic


#
# Computers a homography from 4-correspondences
#
def calculateHomography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.array([corr.item(0), corr.item(1), 1])
        p2 = np.array([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h


#
#Calculate the geometric distance between estimated points and original points
#
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.array([correspondence[0], correspondence[1], 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.array([correspondence[2], correspondence[3], 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


#
#Runs through ransac algorithm, creating homographies from random correspondences
#
def ransac(corr, thresh):
    maxInliers = []
    finalH = None
    for i in range(3500):
        #find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        #call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        #print("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers


def bfmatch(input1, input2):
    print(input1.shape)
    print(input2.shape)
    res = []
    for ii in range(input1.shape[0]):
        min_dist = float('inf')
        min_trainIdx = 0
        min_queryIdx = 0
        for jj in range(input2.shape[0]):
            tmp_dist = l2_distance(input1[ii], input2[jj])
            if tmp_dist < min_dist:
                min_dist = tmp_dist
                min_trainIdx = ii
                min_queryIdx = jj
        tmp = cv2.DMatch(_distance=min_dist, _trainIdx=min_queryIdx, _queryIdx=min_trainIdx, _imgIdx=0)
        res.append(tmp)
    return res


def cartesian_product_transpose_pp(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T

def _warp(image, transformation, outputShape):
    mapping = cartesian_product_transpose_pp(
        np.arange(outputShape[0], dtype=np.float32),
        np.arange(outputShape[1], dtype=np.float32)
    )
    mapping = mapping.reshape(outputShape[0], outputShape[1], 2)
    mapping = perspectiveTransform(mapping, inv(np.array(transformation)))
    return np.swapaxes(remapping(image,mapping), 0, 1)


def perspectiveTransform(mapping, M):
    t1 = ((M[0, 0] * mapping[:, :, 0]) + (M[0, 1] * mapping[:, :, 1]) + M[0, 2]) / (
                (M[2, 0] * mapping[:, :, 0]) + (M[2, 1] * mapping[:, :, 1]) + M[2, 2])
    t2 = ((M[1, 0] * mapping[:, :, 0]) + (M[1, 1] * mapping[:, :, 1]) + M[1, 2]) / (
                (M[2, 0] * mapping[:, :, 0]) + (M[2, 1] * mapping[:, :, 1]) + M[2, 2])
    return np.stack((t1, t2), axis=2)


def warpBorder(image, transformation):
    height = image.shape[0]
    width = image.shape[1]
    border = np.float32([(0, 0), (width, 0), (width, height), (0, height)]).reshape(-1, 1, 2)
    return perspectiveTransform(border, transformation).reshape(-1, 2)


def warp(image, transformation, outputShape = None):
    outputShape = tuple(outputShape)

    newImage = _warp(image, transformation, outputShape)
    return newImage

def remapping(src, mapping):
    ret = np.zeros((mapping.shape[0],mapping.shape[1],src.shape[2]),dtype=np.uint8)
    height = mapping.shape[0]
    width = mapping.shape[1]
    print("Doing mapping and bilinear interpolation...\n")
    for i in range(height):
        for j in range(width):
                x = mapping[i][j][1]
                y = mapping[i][j][0]
                x1 = int(x)
                x2 = x1+1
                y1 = int(y)
                y2 = y1+1
                if x1 >= 0 and y1 >= 0 and x2 < src.shape[0] and y2 < src.shape[1]:
                    # Bilinear interpolation
                    ret[i][j] = (src[x1][y1]*(x2-x)*(y2-y) + src[x1][y2]*(x2-x)*(y-y1) + src[x2][y1]*(x-x1)*(y2-y) + src[x2][y2]*(x-x1)*(y-y1)).astype(np.uint8)
    return ret


def l2_distance(a,b):
    return np.linalg.norm(a-b)

def find_ratio(kp1,ft1,kp2,ft2):  #match kp1 to kp2 (kp1 is smaller than kp2)
    match_list=[]
    dist_list=[]
    ratio_list=[]
    
    for i in range(0,len(kp1)): #kp1 is smaller
        cloest_dist,closest_point_index = 10000000,0
        second_dist,second_point_index = 10000000,0

        for j in range(0,len(kp2)):
            dist = l2_distance(ft1[i],ft2[j])
            if(dist < cloest_dist): #update is needed
                second_dist,second_point_index = cloest_dist,closest_point_index
                cloest_dist,closest_point_index = dist,j
            elif(dist < second_dist):
                second_dist,second_point_index = dist,j

        match_list.append((i,closest_point_index))
        dist_list.append(cloest_dist)
        ratio_list.append(cloest_dist/second_dist)

    return match_list,dist_list,ratio_list

def find_match(match_list,dist_list,ratio_list):    #call by ref
    for i in range(0,len(match_list)):
        p1,p2 = match_list[i]   #get the match
        for j in range(i+1,len(match_list)):    #search if there exist a duplicate match
            check1,check2 = match_list[j]
            if(p2 == check2):   #duplicate element exist(one to two)
                if(ratio_list[i]>ratio_list[j]):
                    match_list[j] = (-1,-1)
                    dist_list[j] = -1
                    ratio_list[j]=-1
                else:
                    match_list[i]= (-1,-1)
                    dist_list[i]=-1
                    ratio_list[i]=-1
    return 0
def remove_trash(l1,l2,l3):
    new1=[]
    new2=[]
    new3=[]
    i=0
    for i in range(0,len(l1)):
        a,b=l1[i]
        if(b!=-1):
            new1.append(l1[i])
    for i in range(0,len(l2)):
        if(l2[i]!=-1):
            new2.append(l2[i])
    for i in range(0,len(l3)):
        if(l3[i]!=-1):
            new3.append(l3[i])
    return new1,new2,new3

def create_match_obj(match_list,dist):
    res = []
    for i in range(0,len(dist)):
        idx_1 , idx_2 = match_list[i]
        # Simply use struct by cv2.DMatch, which is allowed by TA : tiny.cc/kim24y (HW3 Question List)
        tmp = cv2.DMatch(_queryIdx=idx_1,_trainIdx=idx_2,_distance=dist[i])
        res.append(tmp)
    return res

def find_homography(src,target):
    res=[]
    for i in range(0,len(src)):
        x,y = src[i]
        u,v = target[i]
        res.append([x,y,1,0,0,0,-1*u*x,-1*y*u,-1*u])
        res.append([0,0,0,x,y,1,-1*v*x,-1*v*y,-1*v])
    res = np.asarray(res)
    U,S,VH = np.linalg.svd(res)
    L = VH[-1,:]/VH[-1,-1]
    return L.reshape(3,3)

def check_inline(match,kp1,kp2,homo,threshold=50):
    inliner=0
    for i in range(0,len(match)):
        a,b = kp1[match[i].queryIdx].pt
        temp1 = [a,b,1]
        a,b = kp2[match[i].trainIdx].pt
        temp2 = [a,b,1]
        transfer = np.matmul(np.asarray(homo) , np.asarray(temp1))
        dist=l2_distance(transfer , np.asarray(temp2))
        if(dist<threshold):
           inliner=inliner+1     
    return inliner

def RANSAC(match,kp1,kp2,times=3500): #return a homography which has the most inliners in RANSAC test
    best_homo=[]
    best_inliner=-1
    for i in range(0,times):
        select = random.sample(range(0, len(match)), 4) #select 4 points from the matching list
        src=[]
        tar=[]
        for k in range(0,len(select)):  #append the points to list
            src.append(kp1[match[select[k]].queryIdx].pt)
            tar.append(kp2[match[select[k]].trainIdx].pt)
        curr_homo = find_homography(src,tar)    #calculate the homography matrix of four datapoints
        curr_inliner = check_inline(match,kp1,kp2,curr_homo)    #check howmany inliners it has
        #print('running %d times, with its inliner %d and best inliner %d'%(i,curr_inliner,best_inliner))
        if(curr_inliner > best_inliner):    #we find a better homography matrix 
            print('find a better homography matrix with inliner %d'%(curr_inliner))
            best_homo = curr_homo
            best_inliner = curr_inliner
    print('best inliner: %d'%(best_inliner))
    print('best Homo')
    print(best_homo)
    return best_homo