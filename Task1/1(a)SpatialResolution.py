import cv2
import numpy as np 
import matplotlib.pyplot as plt 

image = cv2.imread('Images\Rose.tif',0)

image = cv2.resize(image,(512,512))

m,n = image.shape

f = 2
i = 1
plt.figure(figsize = (12,8))
downsampled_img = image.copy()

while(i<=8):
    plt.subplot(2,4,i)
    plt.imshow(downsampled_img,'gray')
    plt.axis('off')
    m,n = downsampled_img.shape
    plt.title(f"{m}x{n}")

    nh = downsampled_img.shape[0] //2
    nw = downsampled_img.shape[1] // 2

    downsampled_img = np.zeros((nh,nw) , dtype = np.uint8)

    for x in range(nh):
        for y in range(nw):
            downsampled_img[x,y] = image[x*f,y*f]
    image = downsampled_img
    i+=1

plt.show()        


