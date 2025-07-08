# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:nomarker
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import cv2 as cv
img = cv.imread('Lenna.png')

height, width = img.shape[:2] 
print (height, width) 

from matplotlib import pyplot as plt 
plt.imshow(img) 

# cv.cvtColor(img, cv.COLOR_BGR2RGB): convert img from BGR to RGB 
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) 

cv.imwrite('Lenna.jpeg', img) 

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 
plt.imshow(img_hsv, cmap='hsv')

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
plt.imshow(img_gray, cmap='gray') 

height, width = img.shape[:2] 
h_scale = 0.5 
v_scale = 0.4 
new_height = (int) (height * v_scale) # (int) is used to make new_height an integer 
new_width = (int) (width * h_scale) # (int) is used to make new_width an integer 
img_resize = cv.resize(img, (new_width, new_height), interpolation = cv.INTER_LINEAR) 
plt.imshow(cv.cvtColor(img_resize, cv.COLOR_BGR2RGB)) 

t_x = 100 
t_y = 200 

M = np.float32([[1, 0, t_x], [0, 1, t_y]]) 

img_translation = cv.warpAffine(img, M, (width, height)) 
plt.imshow(cv.cvtColor(img_translation, cv.COLOR_BGR2RGB)) 

theta = 45 # rotate 45 degrees in anti-clockwise 

c_x = (width - 1) / 2.0 # column index varies in [0, width-1]
c_y = (height - 1) / 2.0 # row index varies in [0, height-1] 
c = (c_x, c_y) # A point is defined by x and y coordinate 
print(c) 

s = 1
M = cv.getRotationMatrix2D(c, theta, s) 
print(M) 

img_rotation = cv.warpAffine(img, M, (width, height)) 
plt.imshow(cv.cvtColor(img_rotation, cv.COLOR_BGR2RGB)) 

m00 = 0.5 
m01 = 0.3 
m02 = -47.18 
m10 = -0.4 
m11 = 0.5 
m12 = 95.32 
M = np.float32([[m00, m01, m02], [m10, m11, m12]])

height, width = img.shape[:2] 
img_affine = cv.warpAffine(img, M, (width, height)) 
plt.imshow(cv.cvtColor(img_affine, cv.COLOR_BGR2RGB)) 
