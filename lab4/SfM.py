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

BFmatch = BFMATCH(0.8)
Mymatches = BFmatch.Best2Matches(des1, des2)

x, xp = BFmatch.CorresspondenceAcrossImages(kp1, kp2)



print(x)
print(xp)
#print(x)
## Step2 : estimate the fundamental matrix across images (normalized 8 points)
## Step3 : draw the interest points on you found in step.1 in one image and the corresponding epipolar lines in another
## Step4 : get 4 possible solutions of essential matrix from fundamental matrix
## Step5 : find out the most appropriate solution of essential matrix
## Step6 : apply triangulation to get 3D points
## Step7 : find out correspondence across images