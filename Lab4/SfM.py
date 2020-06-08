import matplotlib.pyplot as plt
import sys
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from RANSAC import *
from WARP import *
from BFMATCH import *
from draw import *
from find_essential import *
from triangular import *

def normalize(points, imgsize):
    ''' 
    new_pt =
     [[x0, x1, x2 ...],
      [y0, y1, y2 ...],
      [1,  1,  1  ...]]
    '''
    T = np.array([[2/imgsize[1], 0, -1], 
                  [0, 2/imgsize[0], -1], 
                  [0, 0, 1]])
    new_pt = T.dot(np.array(points))
    return new_pt, T
#case = input("Choose case, 1=Mesona, 2=Statue, 3=nctu: ")
case = sys.argv[1] 
if case == "1":
    InputFile1="./Mesona1.JPG"
    InputFile2="./Mesona2.JPG"
elif case == "2":
    InputFile1="./Statue1.bmp"
    InputFile2="./Statue2.bmp"
elif case == "3":
    InputFile1="./test1.jpg"
    InputFile2="./test2.jpg"

img1 = cv2.imread(InputFile1,0)
img2 = cv2.imread(InputFile2,0)

## Step1 : find out correspondence across images
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

BFmatch = BFMATCH(thresh = 0.8, des1 = des1, des2 = des2, kp1 = kp1, kp2 = kp2)
Mymatches, thirty_match = BFmatch.B2M_30()

x, xp = BFmatch.CorresspondenceAcrossImages()

h_x = np.ones( (x.shape[0], 3), dtype=float)
h_xp = np.ones( (xp.shape[0], 3), dtype=float)
h_x[:, :2] = x
h_xp[:, :2] = xp
h_x = h_x.T
h_xp = h_xp.T

CorList = BFmatch.CORLIST(Mymatches)
## Step2 : estimate the fundamental matrix across images (normalized 8 points)

# normalize
norm_x, T1 = normalize(h_x, img1.shape)
norm_xp, T2 = normalize(h_xp, img2.shape)

RSC8pt = RANSAC(thresh = 0.1, n_times = 1000, points = 10)
F, idx = RSC8pt.ransac_8points(h_x, h_xp, T1, T2)
#print("idx ", idx)

## Step3 : draw the interest points on you found in step.1 in one image and the corresponding epipolar lines in another
inliers_x = h_x[:, idx]
inliers_xp = h_xp[:, idx]

lines_on_img1 = np.dot(F.T, inliers_xp).T
lines_on_img2 = np.dot(F, inliers_x).T

#print(lines_on_img1)
#print(lines_on_img2)

draw(lines_on_img1, lines_on_img2, inliers_x, inliers_xp, img1, img2)

## Step4 : get 4 possible solutions of essential matrix from fundamental matrix
if case == "1":
    K = np.array([[1.4219, 0.0005, 0.5092],
                  [0, 1.4219, 0.3802],
                  [0, 0, 0.0010]], dtype=float)
elif case == "2":
    K = np.array([[5426.566895, 0.678017, 330.096680],
                  [0.000000, 5423.133301, 648.950012],
                  [0.000000,    0.000000,   1.000000]], dtype=float)
else:
    K = np.array([[30519.1445, 11.1330655, 2008.34471],
                  [0, 3056.12848, 1501.53437],
                  [0, 0, 1]], dtype=float)
m1, m2, m3, m4 = find_E(K, F)
## Step5 : find out the most appropriate solution of essential matrix
## Step6 : apply triangulation to get 3D points
true_E, points3D = find_true_E(m1, m2, m3, m4, inliers_x.T, inliers_xp.T)
## Step7 : find out correspondence across images
# print( np.size(points3D))
(xxx, yyy) = points3D.shape
for iii in range(xxx):
    for jjj in range(yyy):
        print(points3D[iii][jjj], end = " ")
    print("", end = "\n")
##print(points3D, end="")
