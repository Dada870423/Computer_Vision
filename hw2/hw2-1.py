import cv2
import math
import numpy as np 
from matplotlib import pyplot as plt

############## To do list
#  1. Multiply the input image by (-1)x+y 
#  2. center the transform
#  3. Compute Fourier transformation 
#  4. Multiply F(u,v) by a ﬁlter 
#  5. Compute the inverse Fourier transformation 
#  6. Obtain the real part 
#  7. Multiply the result in (5) by (-1)x+y.

##############
def My_resize(Img1, Img2, Diff_x, Diff_y):
	FirstDim_x, FirstDim_y = Img1.shape
	SecondDim_x, SecondDim_y = Img2.shape

	if FirstDim_x > SecondDim_x:
		if (Diff_x % 2) == 0:
			bound1 = Diff_x / 2
			bound2 = FirstDim_x - bound1
			Img1 = Img1[bound1:bound2,:]
		else:
			bound1 = int(Diff_x / 2)
			bound2 = FirstDim_x - bound1 + 1
			Img1 = Img1[bound1:bound2,:]
	elif SecondDim_x > FirstDim_x:
		if (Diff_x % 2) == 0:
			bound1 = Diff_x / 2
			bound2 = SecondDim_x - bound1
			Img2 = Img2[bound1:bound2,:]
		else:
			bound1 = int(Diff_x / 2)
			bound2 = SecondDim_x - bound1 + 1
			Img2 = Img2[bound1:bound2,:]

	if FirstDim_y > SecondDim_y:
		if (Diff_y % 2) == 0:
			bound1 = Diff_y / 2
			bound2 = FirstDim_y - bound1
			Img1 = Img1[:,bound1:bound2]
		else:
			bound1 = int(Diff_y / 2)
			bound2 = FirstDim_y - bound1 + 1
			Img1 = Img1[:,bound1:bound2]
	elif SecondDim_y > FirstDim_y:
		if (Diff_y % 2) == 0:
			bound1 = Diff_y / 2
			bound2 = SecondDim_y - bound1
			Img2 = Img2[:,bound1:bound2]
		else:
			bound1 = int(Diff_y / 2)
			bound2 = SecondDim_y - bound1 + 1
			Img2 = Img2[:,bound1:bound2]
	return Img1,Img2










## read amd get the useful information
InputFile1="./cat.bmp"
InputFile2="./dog.bmp"

FirstThreeColor = cv2.imread(InputFile1,3)
SecondThreeColor = cv2.imread(InputFile2,3)

First_B, First_G,First_R = cv2.split(FirstThreeColor)
Second_B, Second_G,Second_R = cv2.split(SecondThreeColor)

FirstDim_x, FirstDim_y = First_B.shape
SecondDim_x, SecondDim_y = Second_B.shape

Dim_x = max(FirstDim_x, SecondDim_x)
Dim_y = max(FirstDim_y, SecondDim_y)

Diff_x = max(FirstDim_x, SecondDim_x) - min(FirstDim_x, SecondDim_x)
Diff_y = max(FirstDim_y, SecondDim_y) - min(FirstDim_y, SecondDim_y)


## our resize
First_B, Second_B = My_resize(First_B, Second_B, Diff_x, Diff_y)

## our resize done









