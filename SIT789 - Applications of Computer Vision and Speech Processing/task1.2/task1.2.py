#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt 
import math

img2 = cv.imread('img2.jpg')
img3 = cv.imread('img3.jpg')


# In[2]:


def color_hist(img):
    hist_blue = cv.calcHist([img],[0],None,[256],[0,256]) #[0] for the blue channel 
    hist_green = cv.calcHist([img],[1],None,[256],[0,256]) #[0] for the blue channel 
    hist_red = cv.calcHist([img],[2],None,[256],[0,256]) #[0] for the blue channel 
    
    plt.plot(hist_blue, color = 'b') 
    plt.xlim([0,256]) 
    plt.show()
    
    plt.plot(hist_green, color = 'g') 
    plt.xlim([0,256]) 
    plt.show()
    
    plt.plot(hist_red, color = 'r') 
    plt.xlim([0,256]) 
    plt.show()


# In[3]:


color_hist(img2)


# In[4]:


color_hist(img3)


# In[5]:


img_gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) 

hist_gray2 = cv.calcHist([img_gray2],[0],None,[256],[0,256]) 
plt.plot(hist_gray2) 
plt.xlim([0,256]) 
plt.show()


# In[6]:


img_gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY) 

hist_gray3 = cv.calcHist([img_gray3],[0],None,[256],[0,256]) 
plt.plot(hist_gray3) 
plt.xlim([0,256]) 
plt.show()


# In[7]:


def getCumulativeDis(hist): 
    c = [] #cummulative distribution 
    s = 0 
    for i in range(0, len(hist)): 
        s = s + hist[i] 
        c.append(s) 
    return c


# In[8]:


c2 = getCumulativeDis(hist_gray2) 
plt.plot(c2, label = 'cummulative distribution img 2', color = 'r') 
plt.legend(loc="upper left") 
plt.xlim([0,256]) 
plt.show() 


# In[9]:


c3 = getCumulativeDis(hist_gray3) 
plt.plot(c3, label = 'cummulative distribution img 3', color = 'r') 
plt.legend(loc="upper left") 
plt.xlim([0,256]) 
plt.show() 


# In[10]:


img_equ2 = cv.equalizeHist(img_gray2) 
plt.imshow(img_equ2, cmap='gray')


# In[12]:


img_equ3 = cv.equalizeHist(img_gray3) 
plt.imshow(img_equ3, cmap='gray')


# In[14]:


hist_equ2 = cv.calcHist([img_equ2],[0],None,[256],[0,256]) 
plt.plot(hist_equ2) 
plt.xlim([0,256]) 
plt.show() 


# In[16]:


hist_equ3 = cv.calcHist([img_equ3],[0],None,[256],[0,256]) 
plt.plot(hist_equ3) 
plt.xlim([0,256]) 
plt.show() 


# In[19]:


c_equ2 = getCumulativeDis(hist_equ2) 
plt.plot(c_equ2,label='cummulative distribution after histogram equalisation of img 2',color='r') 
plt.legend(loc="upper left") 
plt.xlim([0,256]) 
plt.show()


# In[23]:


c_equ3 = getCumulativeDis(hist_equ3) 
plt.plot(c_equ3,label='cummulative distribution after histogram equalisation of img 3',color='r') 
plt.legend(loc="upper left") 
plt.xlim([0,256]) 
plt.show()


# In[25]:


def Chi_2(H1, H2):
    h1 = H1 + 1e-10
    h2 = H2 + 1e-10
    
    num = (h1-h2) ** 2
    denorm = (h1 + h2)
    return np.sum(num / denorm)


# In[27]:


def kl_divergence(H1, H2):
    h1 = H1 + 1e-10
    h2 = H2 + 1e-10
    
    s1 = np.sum(h1)
    s2 = np.sum(h2)
    h1 /= s1
    h2 /= s2
    
    log = [math.log(y) for y in (h1/h2)]
    kl = h1 * log
    return np.sum(kl)


# In[35]:


Chi_2_hist2 = Chi_2(hist_gray2, hist_equ2)
print(Chi_2_hist2)


# In[37]:


kl_divergence_hist2 = kl_divergence(hist_gray2, hist_equ2)
print(kl_divergence_hist2)


# In[39]:


Chi_2_hist3 = Chi_2(hist_gray3, hist_equ3)
print(Chi_2_hist3)


# In[41]:


kl_divergence_hist3 = kl_divergence(hist_gray3, hist_equ3)
print(kl_divergence_hist3)

