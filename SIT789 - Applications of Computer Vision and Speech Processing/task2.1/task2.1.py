#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import cv2 as cv 
from matplotlib import pyplot as plt 

img = cv.imread('empire.jpg') #load image 
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

avg_kernel = np.ones((5,5),np.float32) / 25 #kernel K defined above 
avg_result = cv.filter2D(img_gray,-1,avg_kernel) # 2nd parameter is always set to -1 

plt.imshow(img_gray, 'gray') 


# In[4]:


img_noise = cv.imread('empire_shotnoise.jpg') 
img_noise_gray = cv.cvtColor(img_noise, cv.COLOR_BGR2GRAY) 


# In[5]:


# Testing Gaussian filter
gau_result = cv.GaussianBlur(img_noise_gray, (5, 5), 1) 
plt.imshow(gau_result, 'gray')


# In[6]:


# Testing corner filter
corner_kernel = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float32)

# Apply corner detection kernel
corner_result = cv.filter2D(img_noise_gray, -1, corner_kernel)
plt.imshow(corner_result, 'gray')


# In[7]:


#Testing median filter 
ksize = 5 # neighbourhood of ksize x ksize; ksize must be an odd number 
med_result = cv.medianBlur(img_noise_gray, ksize) 
plt.imshow(med_result, 'gray')


# In[8]:


#Testing bilateral filter 
rad = 5 #radius to determine neighbourhood 
sigma_s = 10 #standard deviation for spatial distance (slide 21 in week 2 handouts) 
sigma_c = 30 #standard deviation for colour difference (slide 21 in week 2 handouts) 
bil_result = cv.bilateralFilter(img_noise_gray, rad, sigma_c, sigma_s) 
plt.imshow(bil_result, 'gray') 


# In[9]:


D_x = np.float32([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8 
der_x = cv.filter2D(img_gray, cv.CV_32F, D_x) # CV_32F is used to store negative values 
plt.imshow(der_x, 'gray')


# In[10]:


D_y = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8 
der_y = cv.filter2D(img_gray, cv.CV_32F, D_y) # CV_32F is used to store negative values 
plt.imshow(der_y, 'gray')


# In[11]:


import math 

height, width = img_gray.shape 
mag_img_gray = np.zeros((height, width), np.float32) # initialise the gradient magnitude image with 0s  

for i in range(0, height): 
    for j in range(0, width): 
        square_der_x = float(der_x[i, j]) * float(der_x[i, j]) 
        square_der_y = float(der_y[i, j]) * float(der_y[i, j]) 
        mag_img_gray[i, j] = int(math.sqrt(square_der_x + square_der_y))         

plt.imshow(mag_img_gray,'gray')


# In[12]:


minVal = 100 # minVal used in hysteresis thresholding 
maxVal = 200 # maxVal used in hysteresis thresholding 
Canny_edges = cv.Canny(img_gray, minVal, maxVal) 
plt.imshow(Canny_edges, 'gray')

