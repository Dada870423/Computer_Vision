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

## read amd get the useful information
InputFile1="./cat.bmp"
InputFile2="./dog.bmp"

FirstThreeColor = cv2.imread(InputFile1,3)
SecondThreeColor = cv2.imread(InputFile2,3)

First_B,First_G,First_R = cv2.split(FirstThreeColor)
Second_B,Second_G,Second_R = cv2.split(SecondThreeColor)

FirstDim_x,FirstDim_y = img1_B.shape