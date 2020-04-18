#!/usr/bin/env python
# coding: utf-8

# In[108]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


# In[117]:


def gau():
    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x**2+y**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()    
    return gaussian_kernel

def subsampling(img, sub_size):
    size_w = img.shape[0]
    size_h = img.shape[1]
    new_w = math.floor(size_w / sub_size)
    new_h = math.floor(size_h / sub_size)
    new = np.zeros((new_w, new_h))
    for x in range(0, new_w):
        for y in range(0, new_h):
            new[x, y] = img[x*sub_size, y*sub_size]
    return new

def upsampling(img, new_w, new_h):
    size_w = img.shape[0]
    size_h = img.shape[1]
    new = np.zeros((new_w, new_h))
    for x in range(0, size_w):
        for y in range(0, size_h):
            new[2*x, 2*y] = img[x,y] 
    return new


# In[118]:


#img = cv2.imread('task1and2_hybrid_pyramid/0_Afghan_girl_after.jpg')
img = cv2.imread('task1and2_hybrid_pyramid/flower.jpg')
#img = cv2.imread('task1and2_hybrid_pyramid/sky.jpg')
#img = cv2.imread('task1and2_hybrid_pyramid/4_einstein.bmp')
shape = img.shape
ori_w = shape[0]
ori_h = shape[1]
dir = '2'


# In[119]:


gau_pyramid = list()
gau_pyramid.append(img)
# create gaussian matrix
gaussian_kernel = gau()
## gaussian subsampling
index=[2,4,8,16]
for sub_size in index:
    w = math.floor(ori_w / sub_size)
    h = math.floor(ori_h / sub_size)
    sub_img = np.zeros((w, h, 3))
    ff = np.zeros((w, h,3))
    for i in range(0,3):
        temp = gau_pyramid[-1][:,:,i]
        new = np.zeros((temp.shape[0], temp.shape[1]))
        pad = np.pad(temp,((1,1),(1,1)),'constant',constant_values = (0,0))    
        for x in range(1, temp.shape[0] + 1):
            for y in range(1, temp.shape[1] + 1):
                new[x-1,y-1] = gaussian_kernel[0,0]*pad[x-1,y-1]+gaussian_kernel[1,0]*pad[x,y-1]+gaussian_kernel[2,0]*pad[x+1,y-1]+gaussian_kernel[0,1]*pad[x-1,y]+gaussian_kernel[1,1]*pad[x,y]+gaussian_kernel[2,1]*pad[x+1,y]+gaussian_kernel[0,2]*pad[x-1,y+1]+gaussian_kernel[1,2]*pad[x,y+1]+gaussian_kernel[2,2]*pad[x+1,y+1]
        sub_img[:,:,i] = subsampling(new.astype('uint8'), 2)
        ff[:,:,i] = 15*np.log(np.abs(np.fft.fftshift(np.fft.fft2(sub_img[:,:,i]))))
    imgpath = 'task1and2_hybrid_pyramid/'+ dir +'/gaussian/' + str(sub_size)+'.jpg'
    cv2.imwrite(imgpath, sub_img)
    imgpath = 'task1and2_hybrid_pyramid/'+dir+'/gaussian/spectrum/' + str(sub_size)+'.jpg'
    cv2.imwrite(imgpath, ff)
    gau_pyramid.append(sub_img)


# In[120]:


## laplacian
lap_pyramid = list()
lap_pyramid.append(gau_pyramid[-1])
for index in range(3, -1, -1):
    w = gau_pyramid[index].shape[0]
    h = gau_pyramid[index].shape[1]
    up_img = np.zeros((w, h, 3))
    ff = np.zeros((w, h, 3))
    for i in range(0, 3):
        temp = upsampling(lap_pyramid[-1][:,:,i], w, h)
        new = np.zeros((temp.shape[0], temp.shape[1]))
        pad = np.pad(temp,((1,1),(1,1)),'constant',constant_values = (0,0))    
        for x in range(1, temp.shape[0] + 1):
            for y in range(1, temp.shape[1] + 1):
                new[x-1,y-1] = gaussian_kernel[0,0]*pad[x-1,y-1]+gaussian_kernel[1,0]*pad[x,y-1]+gaussian_kernel[2,0]*pad[x+1,y-1]+gaussian_kernel[0,1]*pad[x-1,y]+gaussian_kernel[1,1]*pad[x,y]+gaussian_kernel[2,1]*pad[x+1,y]+gaussian_kernel[0,2]*pad[x-1,y+1]+gaussian_kernel[1,2]*pad[x,y+1]+gaussian_kernel[2,2]*pad[x+1,y+1]
        up_img[:,:,i] = new
        up_img[:,:,i] = new.astype('uint8')
        # get the spectrum
        ff[:,:,i] = 15*np.log(np.abs(np.fft.fftshift(np.fft.fft2(up_img[:,:,i]))))
    imgpath = 'task1and2_hybrid_pyramid/'+ dir +'/lap/' + str(index)+'.jpg'
    cv2.imwrite(imgpath, (up_img).astype('uint8'))
    imgpath = 'task1and2_hybrid_pyramid/'+dir+'/lap/spectrum/' + str(index)+'.jpg'
    cv2.imwrite(imgpath, ff)
    lap_pyramid.append(up_img)


# In[121]:


for i in range(0, 5):
    temp = gau_pyramid[i]-lap_pyramid[4-i]
    imgpath = 'task1and2_hybrid_pyramid/'+ dir +'/result/' + str(i)+'.jpg'
    cv2.imwrite(imgpath, temp.astype('uint8'))
    ff = 15*np.log(np.abs(np.fft.fftshift(np.fft.fft2(temp))))
    imgpath = 'task1and2_hybrid_pyramid/'+ dir +'/result/spectrum/' + str(i)+'.jpg'
    cv2.imwrite(imgpath, ff)


# In[122]:


temp = gau_pyramid[0]
for i in range(0,3):
    ff = 15*np.log(np.abs(np.fft.fftshift(np.fft.fft2(temp[:,:,i]))))
imgpath = 'task1and2_hybrid_pyramid/'+ dir +'/gaussian/spectrum/0' +'.jpg'
cv2.imwrite(imgpath, ff.astype('uint8'))


# In[ ]:




