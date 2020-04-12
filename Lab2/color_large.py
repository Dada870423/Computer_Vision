import matplotlib.pyplot as plt
import glob
import numpy as np
from PIL import Image
import tifffile
import math
import cv2

def ncc(a,b):
    a=a-a.mean(axis=0)
    b=b-b.mean(axis=0)
    return np.sum(((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))))
def nccAlign(a, b, t):
    min_ncc = -1
    ivalue=np.linspace(-t,t,2*t,dtype=int)
    jvalue=np.linspace(-t,t,2*t,dtype=int)
    for i in ivalue:
        for j in jvalue:
            nccDiff = ncc(a,np.roll(b,[i,j],axis=(0,1)))
            if nccDiff > min_ncc:
                min_ncc = nccDiff
                output = [i,j]
    return output

imname = 'hw2_data/task3_colorizing/icon.tif'
img = Image.open(imname)
w, h = img.size
new_w = math.floor(w/10)
new_h = math.floor(h/10)
smallimg = img.resize((new_w, new_h), Image.NEAREST)
img = np.asarray(img)
smallimg = np.asarray(smallimg)
plt.imshow(img)
print(img.shape)
print('w',w,'h',h)
#split RGB
height = int(h/3)
print(height)
blue_ = img[0 : height, :]
green_ = img[height : 2*height, :]
red_ = img[2*height : 3*height, :]
print('origin size rgb', red_.shape, green_.shape, blue_.shape)

print(smallimg.shape)
plt.imshow(smallimg)


new_height = int(new_w/3)
blue = smallimg[0 : new_height, :]
green = smallimg[new_height : 2*new_height, :]
red = smallimg[2*new_height : 3*new_height, :]
print('new size rgb', red.shape, green.shape, blue.shape)

alignGtoB = nccAlign(blue,green,20)
alignRtoB = nccAlign(blue,red,20)
print(alignGtoB, alignRtoB)
g=np.roll(green_,[alignGtoB[0]*10,alignGtoB[1]*10],axis=(0,1))
r=np.roll(red_,[alignRtoB[0]*10,alignRtoB[1]*10],axis=(0,1))
print('final rgb', r.shape, g.shape, blue_.shape)

coloured = (np.dstack((r,g,blue_)).astype(np.uint8))
#coloured=coloured[int(coloured.shape[0]*0.05):int(coloured.shape[0]-coloured.shape[0]*0.05),int(coloured.shape[1]*0.05):int(coloured.shape[1]-coloured.shape[1]*0.05)]
coloured = Image.fromarray(coloured)
coloured.save('bigtest.tif')

plt.figure()
plt.imshow(coloured)
plt.show()
