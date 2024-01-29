import cv2
import numpy as np 
import matplotlib.pyplot as plt 

image = cv2.imread('Images\Skeleton.tif',0)
#image = cv2.resize(image,(512,512))

histogram = [0]*256
threshold_hist = [0]*256
threshold_val = 27
segmented_img = image.copy()

m,n = image.shape
for i in range(m):
    for j in range(n):
        pixel = image[i,j]
        histogram[pixel]+=1

        if pixel>=threshold_val:
            segmented_img[i][j] = 255
            threshold_hist[255]+=1
        else:
            threshold_hist[0]+=1
            segmented_img[i][j] = 0

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(image,'gray')
plt.title('Original image')

plt.subplot(2,2,2)
plt.bar(range(256),histogram,width=1,color='gray')
plt.title('Original image')

plt.subplot(2,2,3)
plt.imshow(segmented_img,'gray')
plt.title('After segmentation')


plt.subplot(2,2,4)
plt.bar(range(256),threshold_hist,width=1,color='gray')
plt.title('Original image')

plt.show()        


