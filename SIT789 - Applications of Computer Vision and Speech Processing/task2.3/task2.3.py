# %%
import numpy as np 
import cv2 as cv 
from matplotlib import pyplot as plt 
import statistics 
import time

# %% [markdown]
# # 1. Hough transform for document skew estimation

# %%
# Step 1: Load an input image from file and binarise the image using a threshold, e.g., 200
doc = cv.imread('doc.jpg', 0) # 2nd parameter is set to 0 to read grayscale image
threshold = 200 
ret, doc_bin = cv.threshold(doc, threshold, 255, cv.THRESH_BINARY) 

# %%
# Step 2: Get negative version of the binarised image by subtracting the binarised image from 255.
doc_bin = 255 - doc_bin # convert black/white to white/black
plt.imshow(doc_bin, 'gray') 

# %%
# Step 3: Extract connected components from the negative image. Note that morphology is NOT used here.
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(doc_bin, connectivity=8)

# %%
# Step 4: Select candidate points on the negative image using one of the following strategies 
def selectCandidatePoints(strategy, img):
    # Strategy A: All foreground pixels
    if strategy == 'A':
        start_time = time.time()
        return strategy, img.copy(), time.time() - start_time

    # Strategy B: Component centres
    elif strategy == 'B':
        start_time = time.time()
        candidatePointsB = np.zeros_like(img)
        for i in range(1, num_labels):  # skip background
            x_centroid = int(centroids[i][0])
            y_centroid = int(centroids[i][1])
            candidatePointsB[y_centroid, x_centroid] = 255
        return strategy, candidatePointsB, time.time() - start_time

    # Strategy C: Lowest point (max y) in each component
    elif strategy == 'C':
        start_time = time.time()
        candidatePointsC = np.zeros_like(img)
        for i in range(1, num_labels):
            # Get mask of current component
            component_mask = (labels == i).astype(np.uint8)
            ys, xs = np.where(component_mask)
            if len(ys) > 0:
                max_y = np.max(ys)
                x_at_max_y = xs[np.argmax(ys)]
                candidatePointsC[max_y, x_at_max_y] = 255
        return strategy, candidatePointsC, time.time() - start_time

    else:
        print("Invalid strategy!!!")
        return None

# %%
# Step 5: Remove non-candidate points (already done in strategy images)
# Visualize results

candidatePoints = [selectCandidatePoints('A', doc_bin), 
                   selectCandidatePoints('B', doc_bin), 
                   selectCandidatePoints('C', doc_bin)]

# Plot to compare
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(candidatePoints[0][1], 'gray')
plt.title('Strategy a: All foreground pixels')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(candidatePoints[1][1], 'gray')
plt.title('Strategy b: Component centers')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(candidatePoints[2][1], 'gray')
plt.title('Strategy c: Max-y points')
plt.axis("off")

plt.tight_layout()
plt.show()


# %%
# Step 6: Initialize Hough Transform parameters
distance_resolution = 1
angular_resolution = np.pi / 180
density_threshold = [10, 15, 20]

# %%
def houghTransform(strategy, original, image, distance_resolution, angular_resolution, density_threshold):
    # Step 7: Apply the Hough transform on the negative image
    start_time = time.time()
    lines = cv.HoughLines(image, distance_resolution, angular_resolution, density_threshold)
    hough_time = time.time() - start_time

    # Step 8: Create an array to store the angles of all lines detected by Hough transform
    angles = []
    if lines is not None:
        for rho_theta in lines:
            rho, theta = rho_theta[0]
            angle_deg = np.rad2deg(theta)
            angles.append(angle_deg)

    # Step 9: Calculate median angle
    median_angle = statistics.median(angles)
    deskew_angle = median_angle - 90

    # Step 10: Deskew the image
    # rotate image 
    height, width = original.shape 
    c_x = (width - 1) / 2.0 # column index varies in [0, width-1] 
    c_y = (height - 1) / 2.0 # row index varies in [0, height-1] 
    c = (c_x, c_y) # A point is defined by x and y coordinate 
    M = cv.getRotationMatrix2D(c, deskew_angle, 1) 
    deskewed = cv.warpAffine(original, M, (width, height)) 
    total_time = time.time() - start_time

    print("Hough time:", hough_time, "seconds")
    print("Total time:", total_time, "seconds")

    # Show result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title(f"Strategy: {strategy} | Threshold: {density_threshold} | Original ({median_angle:.2f}°)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(deskewed, cmap='gray')
    plt.title(f"Strategy: {strategy} | Threshold: {density_threshold} | Deskewed ({deskew_angle:.2f}°)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return deskewed

# %% [markdown]
# # 2. Performance analysis 

# %% [markdown]
# ### 2.1. Candidate point selection

# %%
for point in candidatePoints:
    print(f"Computational speed of candidate points using strategy {point[0]}: {point[2]} seconds")

# %% [markdown]
# ### 2.2. Parameter setting

# %%
for point in candidatePoints:
    for threshold in density_threshold:
        houghTransform(point[0], doc, point[1], distance_resolution, angular_resolution, threshold)

# %%
cv.imwrite("doc_deskewed.jpg", houghTransform(candidatePoints[1][0], doc, candidatePoints[1][1], distance_resolution, angular_resolution, 15))

# %% [markdown]
# # 3. Other test cases

# %%
# image taken from https://docs.aspose.cloud/ocr/deskew-image/
name1 = 'test1'
doc1 = cv.imread(f'{name1}.png', 0) # 2nd parameter is set to 0 to read grayscale image
threshold = 200 
ret, doc1_bin = cv.threshold(doc1, threshold, 255, cv.THRESH_BINARY) 
doc1_bin = 255 - doc1_bin # convert black/white to white/black

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(doc1_bin, connectivity=8)

candidatePoints1 = selectCandidatePoints('B', doc1_bin)

print(f"Computational speed of candidate points using strategy {candidatePoints1[0]}: {candidatePoints1[2]} seconds")

cv.imwrite(f'{name1}_deskewed.png', 
           houghTransform(candidatePoints1[0], doc1, candidatePoints1[1], distance_resolution, angular_resolution, 15))

# %%
# image taken from https://betanews.com/2016/07/04/straighten-text-in-scanned-documents-with-deskew/
name2 = 'test2'
doc2 = cv.imread(f'{name2}.png', 0) # 2nd parameter is set to 0 to read grayscale image
threshold = 200 
ret, doc2_bin = cv.threshold(doc2, threshold, 255, cv.THRESH_BINARY) 
doc2_bin = 255 - doc2_bin # convert black/white to white/black

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(doc2_bin, connectivity=8)

candidatePoints2 = selectCandidatePoints('B', doc2_bin)

print(f"Computational speed of candidate points using strategy {candidatePoints2[0]}: {candidatePoints2[2]} seconds")

cv.imwrite(f'{name2}_deskewed.png', 
           houghTransform(candidatePoints2[0], doc2, candidatePoints2[1], distance_resolution, angular_resolution, 15))

# %%
# image taken from https://docs.aspose.com/imaging/net/deskewing-a-scanned-image/
name3 = 'test3'
doc3 = cv.imread(f'{name3}.png', 0) # 2nd parameter is set to 0 to read grayscale image
threshold = 200 
ret, doc3_bin = cv.threshold(doc3, threshold, 255, cv.THRESH_BINARY) 
doc3_bin = 255 - doc3_bin # convert black/white to white/black

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(doc3_bin, connectivity=8)

candidatePoints3 = selectCandidatePoints('B', doc3_bin)

print(f"Computational speed of candidate points using strategy {candidatePoints3[0]}: {candidatePoints3[2]} seconds")

cv.imwrite(f'{name3}_deskewed.png', 
           houghTransform(candidatePoints3[0], doc3, candidatePoints3[1], distance_resolution, angular_resolution, 15))

# %%
name4 = 'test4'
doc4 = cv.imread(f'{name4}.jpg', 0) # 2nd parameter is set to 0 to read grayscale image
threshold = 200 
ret, doc4_bin = cv.threshold(doc4, threshold, 255, cv.THRESH_BINARY) 
doc4_bin = 255 - doc4_bin # convert black/white to white/black

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(doc4_bin, connectivity=8)

candidatePoints4 = selectCandidatePoints('B', doc4_bin)

print(f"Computational speed of candidate points using strategy {candidatePoints4[0]}: {candidatePoints4[2]} seconds")

cv.imwrite(f'{name4}_deskewed.jpg', 
           houghTransform(candidatePoints4[0], doc4, candidatePoints4[1], distance_resolution, angular_resolution, 15))

# %%
# image taken from https://felix.abecassis.me/2011/09/opencv-detect-skew-angle/
name5 = 'test5'
doc5 = cv.imread(f'{name5}.jpg', 0) # 2nd parameter is set to 0 to read grayscale image
threshold = 200 
ret, doc5_bin = cv.threshold(doc5, threshold, 255, cv.THRESH_BINARY) 
doc5_bin = 255 - doc5_bin # convert black/white to white/black

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(doc5_bin, connectivity=8)

candidatePoints5 = selectCandidatePoints('B', doc5_bin)

print(f"Computational speed of candidate points using strategy {candidatePoints5[0]}: {candidatePoints5[2]} seconds")

cv.imwrite(f'{name5}_deskewed.jpg', 
           houghTransform(candidatePoints5[0], doc5, candidatePoints5[1], distance_resolution, angular_resolution, 15))

# %%
# image taken from https://netraneupane.medium.com/document-skewness-detection-and-correction-8ba8204b5577
name6 = 'test6'
doc6 = cv.imread(f'{name6}.jpg', 0) # 2nd parameter is set to 0 to read grayscale image
threshold = 200 
ret, doc6_bin = cv.threshold(doc6, threshold, 255, cv.THRESH_BINARY) 
doc6_bin = 255 - doc6_bin # convert black/white to white/black

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(doc6_bin, connectivity=8)

candidatePoints6 = selectCandidatePoints('B', doc6_bin)

print(f"Computational speed of candidate points using strategy {candidatePoints6[0]}: {candidatePoints6[2]} seconds")

cv.imwrite(f'{name6}_deskewed.jpg', 
           houghTransform(candidatePoints6[0], doc6, candidatePoints6[1], distance_resolution, angular_resolution, 15))

# %%
# image taken from https://www.researchgate.net/figure/a-Handwritten-skewed-document-b-Skew-corrected-image-skew-angle-887695-Result_fig2_281035858
name7 = 'test7'
doc7 = cv.imread(f'{name7}.jpg', 0) # 2nd parameter is set to 0 to read grayscale image
threshold = 200 
ret, doc7_bin = cv.threshold(doc7, threshold, 255, cv.THRESH_BINARY) 
doc7_bin = 255 - doc7_bin # convert black/white to white/black

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(doc7_bin, connectivity=8)

candidatePoints7 = selectCandidatePoints('B', doc7_bin)

print(f"Computational speed of candidate points using strategy {candidatePoints7[0]}: {candidatePoints7[2]} seconds")

cv.imwrite(f'{name7}_deskewed.jpg', 
           houghTransform(candidatePoints7[0], doc7, candidatePoints7[1], distance_resolution, angular_resolution, 15))

# %%
# image taken from https://stackoverflow.com/questions/41546181/how-to-deskew-a-scanned-text-page-with-imagemagick
name8 = 'test8'
doc8 = cv.imread(f'{name8}.jpg', 0) # 2nd parameter is set to 0 to read grayscale image
threshold = 200 
ret, doc8_bin = cv.threshold(doc8, threshold, 255, cv.THRESH_BINARY) 
doc8_bin = 255 - doc8_bin # convert black/white to white/black

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(doc8_bin, connectivity=8)

candidatePoints8 = selectCandidatePoints('B', doc8_bin)

print(f"Computational speed of candidate points using strategy {candidatePoints8[0]}: {candidatePoints8[2]} seconds")

cv.imwrite(f'{name8}_deskewed.jpg', 
           houghTransform(candidatePoints8[0], doc8, candidatePoints8[1], distance_resolution, angular_resolution, 15))

# %%
# image taken from https://ijcst.com/vol32/2/deepak.pdf
name9 = 'test9'
doc9 = cv.imread(f'{name9}.jpg', 0) # 2nd parameter is set to 0 to read grayscale image
threshold = 200 
ret, doc9_bin = cv.threshold(doc9, threshold, 255, cv.THRESH_BINARY) 
doc9_bin = 255 - doc9_bin # convert black/white to white/black

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(doc9_bin, connectivity=8)

candidatePoints9 = selectCandidatePoints('B', doc9_bin)

print(f"Computational speed of candidate points using strategy {candidatePoints9[0]}: {candidatePoints9[2]} seconds")

cv.imwrite(f'{name9}_deskewed.jpg', 
           houghTransform(candidatePoints9[0], doc9, candidatePoints9[1], distance_resolution, angular_resolution, 15))

# %%
# image taken from https://www.kaggle.com/datasets/sthabile/noisy-and-rotated-scanned-documents?resource=download
name10 = 'test10'
doc10 = cv.imread(f'{name10}.png', 0) # 2nd parameter is set to 0 to read grayscale image
threshold = 200 
ret, doc10_bin = cv.threshold(doc10, threshold, 255, cv.THRESH_BINARY) 
doc10_bin = 255 - doc10_bin # convert black/white to white/black

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(doc10_bin, connectivity=8)

candidatePoints10 = selectCandidatePoints('B', doc10_bin)

print(f"Computational speed of candidate points using strategy {candidatePoints10[0]}: {candidatePoints10[2]} seconds")

cv.imwrite(f'{name10}_deskewed.png', 
           houghTransform(candidatePoints10[0], doc10, candidatePoints10[1], distance_resolution, angular_resolution, 15))

# %%
# C:\Program Files\Tesseract-OCR

import pytesseract 

pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# %%
doc = cv.imread('doc.jpg')
text = pytesseract.image_to_string(doc) 

print("OCR result before deskew:")
print(text)

deskewed = cv.imread('doc_deskewed.jpg')
text = pytesseract.image_to_string(deskewed) 
print("OCR result after deskew:")
print(text)

# %%
doc = cv.imread(f'{name1}.png')
text = pytesseract.image_to_string(doc) 

print("OCR result before deskew:")
print(text)

deskewed = cv.imread(f'{name1}_deskewed.png')
text = pytesseract.image_to_string(deskewed) 
print("OCR result after deskew:")
print(text)

# %%
doc = cv.imread(f'{name2}.png')
text = pytesseract.image_to_string(doc) 

print("OCR result before deskew:")
print(text)

deskewed = cv.imread(f'{name2}_deskewed.png')
text = pytesseract.image_to_string(deskewed) 
print("OCR result after deskew:")
print(text)

# %%
doc = cv.imread(f'{name3}.png')
text = pytesseract.image_to_string(doc) 

print("OCR result before deskew:")
print(text)

deskewed = cv.imread(f'{name3}_deskewed.png')
text = pytesseract.image_to_string(deskewed) 
print("OCR result after deskew:")
print(text)

# %%
doc = cv.imread(f'{name4}.jpg')
text = pytesseract.image_to_string(doc) 

print("OCR result before deskew:")
print(text)

deskewed = cv.imread(f'{name4}_deskewed.jpg')
text = pytesseract.image_to_string(deskewed) 
print("OCR result after deskew:")
print(text)

# %%
doc = cv.imread(f'{name5}.jpg')
text = pytesseract.image_to_string(doc) 

print("OCR result before deskew:")
print(text)

deskewed = cv.imread(f'{name5}_deskewed.jpg')
text = pytesseract.image_to_string(deskewed) 
print("OCR result after deskew:")
print(text)

# %%
doc = cv.imread(f'{name6}.jpg')
text = pytesseract.image_to_string(doc) 

print("OCR result before deskew:")
print(text)

deskewed = cv.imread(f'{name6}_deskewed.jpg')
text = pytesseract.image_to_string(deskewed) 
print("OCR result after deskew:")
print(text)

# %%
doc = cv.imread(f'{name7}.jpg')
text = pytesseract.image_to_string(doc) 

print("OCR result before deskew:")
print(text)

deskewed = cv.imread(f'{name7}_deskewed.jpg')
text = pytesseract.image_to_string(deskewed) 
print("OCR result after deskew:")
print(text)

# %%
doc = cv.imread(f'{name8}.jpg')
text = pytesseract.image_to_string(doc) 

print("OCR result before deskew:")
print(text)

deskewed = cv.imread(f'{name8}_deskewed.jpg')
text = pytesseract.image_to_string(deskewed) 
print("OCR result after deskew:")
print(text)

# %%
doc = cv.imread(f'{name9}.jpg')
text = pytesseract.image_to_string(doc) 

print("OCR result before deskew:")
print(text)

deskewed = cv.imread(f'{name9}_deskewed.jpg')
text = pytesseract.image_to_string(deskewed) 
print("OCR result after deskew:")
print(text)

# %%
doc = cv.imread(f'{name10}.png')
text = pytesseract.image_to_string(doc) 

print("OCR result before deskew:")
print(text)

deskewed = cv.imread(f'{name10}_deskewed.png')
text = pytesseract.image_to_string(deskewed) 
print("OCR result after deskew:")
print(text)


