import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from RANSAC import *
from WARP import *
from BFMATCH import *


InputFile1="./Mesona1.JPG"
InputFile2="./Mesona2.JPG"

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
RSC = RANSAC(thresh = 10.0, n_times = 1000, points = 4)
H, Lines = RSC.ransac(CorList = CorList)
print("H: ", H)


## Step3 : draw the interest points on you found in step.1 in one image and the corresponding epipolar lines in another
## Step4 : get 4 possible solutions of essential matrix from fundamental matrix
## Step5 : find out the most appropriate solution of essential matrix
## Step6 : apply triangulation to get 3D points
## Step7 : find out correspondence across images