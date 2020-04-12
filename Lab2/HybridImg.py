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
def UnderBound(InputImg):
    Dim_x, Dim_y, Dim_z = InputImg.shape
    for x in range(Dim_x):
        for y in range(Dim_y):
            for z in range(3):
                if InputImg[x][y][z] > 255:
                    InputImg[x][y][z] = 255
    return InputImg




def InverseFourier(B, G, R):
    BB = np.fft.ifft2(np.fft.ifftshift(B))
    GG = np.fft.ifft2(np.fft.ifftshift(G))
    RR = np.fft.ifft2(np.fft.ifftshift(R))
    return BB, GG, RR


def HFilter(InputImg, D0, H1L0):
    process_img = InputImg
    Dim_x, Dim_y = InputImg.shape
    for x in range(Dim_x):
        for y in range(Dim_y):
            if H1L0 == 1:
                if InputImg[x][y] < D0:
                    InputImg[x][y] = 0
                else:
                    InputImg[x][y] = InputImg[x][y] * (1 - GaussianValue(InputImg[x][y], D0))
            else:
                if InputImg[x][y] > D0:
                    InputImg[x][y] = 0
                else:
                    InputImg[x][y] = InputImg[x][y] * (GaussianValue(InputImg[x][y], D0))
    return InputImg

def GaussianValue(D, D0):
    ExpValue = (abs((D * D) / (2 * D0 * D0))) * (-1)
    return math.exp(ExpValue)



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
## Function done
## read amd get the useful information
InputFile1="./1_bicycle.bmp"
InputFile2="./1_motorcycle.bmp"

FirstThreeColors = cv2.imread(InputFile1,3)
SecondThreeColors = cv2.imread(InputFile2,3)

First_B, First_G, First_R = cv2.split(FirstThreeColors)
Second_B, Second_G, Second_R = cv2.split(SecondThreeColors)

FirstDim_x, FirstDim_y = First_B.shape
SecondDim_x, SecondDim_y = Second_B.shape

Dim_x = max(FirstDim_x, SecondDim_x)
Dim_y = max(FirstDim_y, SecondDim_y)

Diff_x = max(FirstDim_x, SecondDim_x) - min(FirstDim_x, SecondDim_x)
Diff_y = max(FirstDim_y, SecondDim_y) - min(FirstDim_y, SecondDim_y)


## our resize
First_B, Second_B = My_resize(Img1 = First_B, Img2 = Second_B, Diff_x = Diff_x, Diff_y = Diff_y)
First_G, Second_G = My_resize(Img1 = First_G, Img2 = Second_G, Diff_x = Diff_x, Diff_y = Diff_y)
First_R, Second_R = My_resize(Img1 = First_R, Img2 = Second_R, Diff_x = Diff_x, Diff_y = Diff_y)
## our resize done

## Compute Fourier transformation 
Fourier_1_B = np.fft.fft2(First_B, (Dim_x, Dim_y))
Fourier_1_G = np.fft.fft2(First_G, (Dim_x, Dim_y))
Fourier_1_R = np.fft.fft2(First_R, (Dim_x, Dim_y))

Fourier_2_B = np.fft.fft2(Second_B, (Dim_x, Dim_y))
Fourier_2_G = np.fft.fft2(Second_G, (Dim_x, Dim_y))
Fourier_2_R = np.fft.fft2(Second_R, (Dim_x, Dim_y))

### shift
FourierShift_1_B = np.fft.fftshift(Fourier_1_B)
FourierShift_1_G = np.fft.fftshift(Fourier_1_G)
FourierShift_1_R = np.fft.fftshift(Fourier_1_R)

FourierShift_2_B = np.fft.fftshift(Fourier_2_B)
FourierShift_2_G = np.fft.fftshift(Fourier_2_G)
FourierShift_2_R = np.fft.fftshift(Fourier_2_R)

## Multiply F(u,v) by a ﬁlter 
### High
H_B = HFilter(FourierShift_1_B, D0 = 6, H1L0 = 1)
H_G = HFilter(FourierShift_1_G, D0 = 6, H1L0 = 1)
H_R = HFilter(FourierShift_1_R, D0 = 6, H1L0 = 1)

### Low
L_B = HFilter(FourierShift_2_B, D0 = 99996, H1L0 = 0)
L_G = HFilter(FourierShift_2_G, D0 = 99996, H1L0 = 0)
L_R = HFilter(FourierShift_2_R, D0 = 99996, H1L0 = 0)
### filter done

## Compute the inverse Fourier transformation 
Inv_1_B, Inv_1_G, Inv_1_R = InverseFourier(B = H_B, G = H_G, R = H_R)
Inv_2_B, Inv_2_G, Inv_2_R = InverseFourier(B = L_B, G = L_G, R = L_R)

FilterImg1 = np.absolute(np.dstack((Inv_1_R, Inv_1_G, Inv_1_B)))
FilterImg2 = np.absolute(np.dstack((Inv_2_R, Inv_2_G, Inv_2_B)))
## inverse done


## Obtain the real part
### mix
MixB = np.fft.ifftshift(H_B + L_B)
MixG = np.fft.ifftshift(H_G + L_G)
MixR = np.fft.ifftshift(H_R + L_R)

### inverse
RealPart_B = np.fft.ifft2(MixB)
RealPart_G = np.fft.ifft2(MixG)
RealPart_R = np.fft.ifft2(MixR)

### REAL PART??????
Pre_FinalImg = np.dstack((RealPart_R, RealPart_G, RealPart_B))
Pre_FinalImg = UnderBound(InputImg = Pre_FinalImg)
FinalImg = np.absolute(Pre_FinalImg)

## Obtain done

## show
OriginImg1 = cv2.merge([First_R, First_G, First_B])
OriginImg2 = cv2.merge([Second_R, Second_G, Second_B])


plt.subplot(151), plt.imshow(OriginImg1)
plt.title("Input(H)"), plt.xticks([]), plt.yticks([])

plt.subplot(152), plt.imshow(OriginImg2)
plt.title("Input(L)"), plt.xticks([]), plt.yticks([])

plt.subplot(153), plt.imshow(FilterImg1.astype(int))
plt.title("High"), plt.xticks([]), plt.yticks([])

plt.subplot(154), plt.imshow(FilterImg2.astype(int))
plt.title("Low"), plt.xticks([]), plt.yticks([])


plt.subplot(155), plt.imshow(FinalImg.astype(int))
plt.title("Hybrid"), plt.xticks([]), plt.yticks([])
plt.show()
