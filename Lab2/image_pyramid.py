#!/usr/bin/env python
# coding: utf-8

# In[132]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[154]:


def gau():
    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x**2+y**2))
    r=5
    #Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    #gaussian_kernel = gaussian_kernel / ((2*r)*(2*r))
    
    return gaussian_kernel

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


# In[211]:


#img = cv2.imread('task1and2_hybrid_pyramid/0_Afghan_girl_after.jpg')
#img = cv2.imread('task1and2_hybrid_pyramid/dog.jpg')
img = cv2.imread('task1and2_hybrid_pyramid/rainbow.jpg')
#img = cv2.imread('task1and2_hybrid_pyramid/4_einstein.bmp')
shape = img.shape
w = shape[0]
h = shape[1]
dir = '2'


# In[212]:


## gaussian filter
gaussian_kernel = gau()
gau_img_after = np.zeros((w,h,3))
ff = np.zeros((w,h,3))
for i in range(0,3):
    temp = img[:,:,i]
    new = np.zeros((w,h))
    pad = np.pad(temp,((1,1),(1,1)),'constant',constant_values = (0,0))    
    for x in range(1,w+1):
        for y in range(1,h+1):
            new[x-1,y-1] = gaussian_kernel[0,0]*pad[x-1,y-1]+gaussian_kernel[1,0]*pad[x,y-1]+gaussian_kernel[2,0]*pad[x+1,y-1]+gaussian_kernel[0,1]*pad[x-1,y]+gaussian_kernel[1,1]*pad[x,y]+gaussian_kernel[2,1]*pad[x+1,y]+gaussian_kernel[0,2]*pad[x-1,y+1]+gaussian_kernel[1,2]*pad[x,y+1]+gaussian_kernel[2,2]*pad[x+1,y+1]
    temp3 = new.astype('uint8')
    gau_img_after[:,:,i] = temp3
    temp = np.fft.fft2(gau_img_after[:,:,i])
    temp = np.fft.fftshift(temp)
    ff[:,:,i] = 15*np.log(np.abs(temp))
imgpath = 'task1and2_hybrid_pyramid/'+ dir + '/gaussian/0.jpg'
cv2.imwrite(imgpath, gau_img_after)
imgpath = 'task1and2_hybrid_pyramid/' + dir + '/gaussian/spectrum/0.jpg'
cv2.imwrite(imgpath, ff)


# In[213]:


## gaussian subsampling
index=[2,4,8,16]
for sub_size in index:
    gau_img = np.zeros(( math.floor(w / sub_size),math.floor(h / sub_size),3))
    ff = gau_img
    for i in range(0,3):
        gau_img[:,:,i] = subsampling(gau_img_after[:,:,i], sub_size)
    imgpath = 'task1and2_hybrid_pyramid/'+dir+'/gaussian/' + str(sub_size)+'.jpg'
    cv2.imwrite(imgpath, gau_img)
    for i in range(0,3):
        temp = np.fft.fft2(gau_img[:,:,i])
        temp = np.fft.fftshift(temp)
        ff[:,:,i] = 15*np.log(np.abs(temp))
    imgpath = 'task1and2_hybrid_pyramid/'+dir+'/gaussian/spectrum/' + str(sub_size)+'.jpg'
    cv2.imwrite(imgpath, ff)


# In[215]:


## laplacian filter
lap_img_after = np.zeros((w,h,3))
ff = np.zeros((w,h,3))
for i in range(0,3):
    temp = img[:,:,i]
    new = np.zeros((w,h))
    pad = np.pad(temp,((1,1),(1,1)),'constant',constant_values = (0,0))
    
    for x in range(1,w+1):
        for y in range(1,h+1):
            new[x-1,y-1] = pad[x,y-1]+pad[x-1,y]-4*pad[x,y]+pad[x+1,y]+pad[x,y+1]
    new = temp - new
    temp3 = new.astype('uint8')
    #print(temp3.sum())
    lap_img_after[:,:,i] = temp3
    temp = np.fft.fft2(lap_img_after[:,:,i])
    temp = np.fft.fftshift(temp)
    ff[:,:,i] = 15*np.log(np.abs(temp))

imgpath = 'task1and2_hybrid_pyramid/'+dir+'/lap/0.jpg'
cv2.imwrite(imgpath,lap_img_after)
imgpath = 'task1and2_hybrid_pyramid/'+dir+'/lap/spectrum/0.jpg'
cv2.imwrite(imgpath, ff)


# In[216]:


## lap subsampling
index=[2,4,8,16]
for sub_size in index:
    lap_img = np.zeros(( math.floor(w / sub_size),math.floor(h / sub_size),3))
    ff = lap_img
    # save subsampling
    for i in range(0,3):
        lap_img[:,:,i] = subsampling(lap_img_after[:,:,i], sub_size)
    imgpath = 'task1and2_hybrid_pyramid/'+dir+'/lap/' + str(sub_size)+'.jpg'
    cv2.imwrite(imgpath, lap_img)
    # save spectrum
    for i in range(0,3):
        temp = lap_img[:,:,i]
        temp = np.fft.fft2(temp)
        temp = np.fft.fftshift(temp)
        ff[:,:,i] = 15*np.log(np.abs(temp))
    imgpath = 'task1and2_hybrid_pyramid/'+dir+'/lap/spectrum/' + str(sub_size)+'.jpg'
    cv2.imwrite(imgpath, ff)

