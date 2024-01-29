import cv2
import numpy as np 
import matplotlib.pyplot as plt 

image = cv2.imread('Images\Fractured.tif',0)
image = cv2.resize(image,(512,512))

min_int = int(input('Enter the minimum intensity value to enhance image: '))
max_int = int(input('Enter the maximum intensity value to enhance image: '))

m,n = image.shape
enhanced_image = image.copy()
for i in range(m):
    for j in range(n):
        pixel = image[i,j]
        if (min_int<=pixel and max_int>=pixel):
            enhanced_image[i,j]+=40
        enhanced_image[i,j] = 255 if enhanced_image[i,j]>255 else enhanced_image[i,j]


plt.figure(figsize=(12,8))

plt.subplot(1,2,1)
plt.imshow(image,'gray')
plt.title('Original image')

plt.subplot(1,2,2)
plt.imshow(enhanced_image,'gray')
plt.title('After brightness enhancement')

plt.show()        


