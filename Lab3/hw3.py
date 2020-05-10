import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from RANSAC import RANSAC

class myMatch():
    def __init__(self, id, distance):
        self.id = id
        self.distance = distance

def BFmatch (des1, des2, k):
    match = []
    idx1 = 0
    for p1 in des1:
        best_m = []
        temp0 = cv2.DMatch(idx1, 0, math.sqrt((p1 - des2[0]).T.dot(p1 - des2[0])))
        temp1 = cv2.DMatch(idx1, 1, math.sqrt((p1 - des2[1]).T.dot(p1 - des2[1])))
        if temp0.distance < temp1.distance:
            best_m.append(temp0)
            best_m.append(temp1)       
        else:
            best_m.append(temp1)
            best_m.append(temp0)
            
        idx2 = 0
        for p2 in des2:
            dis = math.sqrt((p1-p2).T.dot((p1-p2)))
            if dis < best_m[0].distance:
                best_m[0].trainIdx = idx2
                best_m[0].distance = dis
            elif dis < best_m[1].distance:
                best_m[1].trainIdx = idx2
                best_m[1].distance = dis
            idx2 = idx2 + 1
        idx1 = idx1 + 1
        match.append(best_m)
    return match


imname1 = 'data/S1.jpg'
imname2 = 'data/S2.jpg'

# part1 
img1 = cv2.imread(imname1)
Gimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(imname2)
Gimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(Gimg1, None)
kp2, des2 = sift.detectAndCompute(Gimg2, None)

# part2

#找match不確定可不可以用cv2 function 好像 不行(?)
#bf = cv2.BFMatcher()
#matches = bf.knnMatch(des1, des2, k=2)
Mymatches = BFmatch(des1, des2, 2)

# Apply ratio test
'''good = []
for m in matches:
    if m[0].distance < 0.8*m[1].distance:
        good.append((m[0].trainIdx, m[0].queryIdx))
matches = np.asarray(good)
'''
temp = []
for m in Mymatches:
    if m[0].distance < 0.8*m[1].distance:
        temp.append((m[0].trainIdx, m[0].queryIdx))
Mymatches = np.asarray(temp)
  
# 畫圖
(hA, wA) = img1.shape[:2]
(hB, wB) = img2.shape[:2]
vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
vis[0:hA, 0:wA] = img1
vis[0:hB, wA:] = img2
 
for (trainIdx, queryIdx) in Mymatches:
    temp = np.random.randint(0, high=255, size=(3,))
    color = (np.asscalar(temp[0]), np.asscalar(temp[1]), np.asscalar(temp[2]))
    #print(color)
    ptA = (int(kp1[queryIdx].pt[0]), int(kp1[queryIdx].pt[1]))
    ptB = (int(kp2[trainIdx].pt[0] + wA), int(kp2[trainIdx].pt[1]))
    cv2.line(vis, ptA, ptB, color, 1)
plt.imshow(vis)
plt.show()


# part3 
#do RANSAC
RSC = RANSAC(thresh = 10.0, n_times = 100, points = 4)
RSC.ransac()
    