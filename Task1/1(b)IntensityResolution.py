import cv2
import numpy as np 
import matplotlib.pyplot as plt 

image = cv2.imread('Images\Skull.tif',0)

image = cv2.resize(image,(512,512))

m,n = image.shape


plt.figure(figsize = (12,8))
downsampled_img = image.copy()
total_level = 8

for i in range(total_level):
    reduced_level = 2 ** i
    downsampled_img = (image//reduced_level)*reduced_level

    plt.subplot(2,4,i+1)
    plt.imshow(downsampled_img,'gray')
    plt.title(f"{total_level-i} bits")


plt.show()        


