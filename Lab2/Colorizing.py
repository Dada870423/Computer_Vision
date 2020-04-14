import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
import math
import cv2

def Ncc(a, b):
    a = a-a.mean(axis=0)
    b = b-b.mean(axis=0)
    return np.sum(((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))))

def NccAlign(a, b, window_size):
    min_ncc = -1
    ivalue = np.linspace(-window_size, window_size, 2*window_size, dtype=int)
    jvalue = np.linspace(-window_size, window_size, 2*window_size, dtype=int)
    for i in ivalue:
        for j in jvalue:
            nccDiff = Ncc(a, np.roll(b, [i, j], axis=(0, 1)))
            if nccDiff > min_ncc:
                min_ncc = nccDiff
                output = [i, j]
    return output

def subsampling(img, sub_size):
    size_w = img.shape[0]
    size_h = img.shape[1]
    new_w = math.floor(size_w / sub_size)
    new_h = math.floor(size_h / sub_size)
    new = np.zeros((new_w, new_h))
    #print(size_w,size_h)
    #print(new_w,new_h)
    for x in range(0, new_w):
        for y in range(0, new_h):
            new[x, y] = img[x*sub_size, y*sub_size]
            #print("x y :", x*sub_size, y*sub_size)
    return new


imname = glob.glob('hw2_data/task3_colorizing/*.jpg')
tif_imname = glob.glob('hw2_data/task3_colorizing/*.tif')

# for jpg
for idx, fname in enumerate(imname):
    print(fname)
    img = Image.open(fname)
    img = np.asarray(img)
    print(img.shape)
    
    #cut boarder 
    w, h = img.shape
    print('origin:', w, h)
    img = img[int(w*0.01) : int(w-w*0.02), int(h*0.05) : int(h-h*0.05)]
    w, h = img.shape
    print('small:', w, h)
    
    #spilt RGB
    height = int(w/3)
    blue = img[0 : height, :]
    green = img[height : 2*height, :]
    red = img[2*height : 3*height, :]
    print('origin size rgb', red.shape, green.shape, blue.shape)
    #allign
    alignGtoB = NccAlign(blue, green, 20)
    alignRtoB = NccAlign(blue, red, 20)
    print(alignGtoB, alignRtoB)
    g = np.roll(green, alignGtoB, axis=(0, 1))
    r = np.roll(red, alignRtoB, axis=(0, 1))
    
    #output
    colored = (np.dstack((r, g, blue))).astype(np.uint8)
    colored = Image.fromarray(colored)
    colored.save('test.jpg')
    plt.figure()
    plt.imshow(colored)   

#for tif
for idx, fname in enumerate(tif_imname):
    print(fname)
    img = Image.open(fname)
    w, h = img.size
    
    # resize to 1/10
    new_w = math.floor(w/10)
    new_h = math.floor(h/10)
    #smallimg = img.resize((new_w, new_h), Image.NEAREST)
    img = np.asarray(img)
    #smallimg = np.asarray(smallimg)
    smallimg = subsampling(img, 10)
    print(img.shape)
    print('w', w, 'h', h)
    print('w_small', new_w, 'h', new_h)
    
    #split RGB 
    height = int(h/3)
    print(height)
    blue_ = img[0 : height, :]
    green_ = img[height : 2*height, :]
    red_ = img[2*height : 3*height, :]
    print('origin size rgb', red_.shape, green_.shape, blue_.shape)
    print(smallimg.shape)
    new_height = int(new_h/3)
    blue = smallimg[0 : new_height, :]
    green = smallimg[new_height : 2*new_height, :]
    red = smallimg[2*new_height : 3*new_height, :]
    print('new size rgb', red.shape, green.shape, blue.shape)
    
    #allign
    alignGtoB = NccAlign(blue, green, 20)
    alignRtoB = NccAlign(blue, red, 20)
    print(alignGtoB, alignRtoB)
    g=np.roll(green_, [alignGtoB[0]*10, alignGtoB[1]*10], axis=(0, 1))
    r=np.roll(red_, [alignRtoB[0]*10, alignRtoB[1]*10], axis=(0, 1))
    print('final rgb', r.shape, g.shape, blue_.shape)

    r = r/256
    g = g/256
    blue_ = blue_/256

    colored = (np.dstack((r, g, blue_)).astype(np.uint8))
    colored = Image.fromarray(colored)
    colored.save('bigtest.tif')
    plt.figure()
    plt.imshow(colored)
    
plt.show()
