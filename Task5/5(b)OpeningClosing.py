import cv2
import numpy as np 
import matplotlib.pyplot as plt 


def Opening(image,se):
    oi = image.copy()
    oi = Erosion(oi,se)
    mi = Dilation(oi,se)
    return mi

def Closing(image,se):
    ci = image.copy()
    ci = Dilation(ci,se)
    mi = Erosion(ci,se)
    return mi


def Erosion(image,structuring_element):
    row,col = image.shape
    se = structuring_element * 255
    srow,scol = se.shape
    sh,sw = srow//2,scol//2

    modified_image = image.copy()

    for i in range(row):
        for j in range(col):
            fit = True
            for x in range(-sh,sh+1):
                for y in range(-sw,sw+1):
                    if (i+x>=0 and i+x<row and j+y>=0 and j+y<col):
                        if (se[x,y] and image[i+x,j+y] != se[x+sh,y+sw]):  
                            fit = False
            modified_image[i,j] = 255 if fit else 0
    return modified_image  

def Dilation(image,structuring_element):
    row,col = image.shape
    se = structuring_element * 255
    srow,scol = se.shape
    sh,sw = srow//2,scol//2

    modified_image = image.copy()

    for i in range(row):
        for j in range(col):
            hit = False
            for x in range(-sh,sh+1):
                for y in range(-sw,sw+1):
                    if (i+x>=0 and i+x<row and j+y>=0 and j+y<col):
                        if (se[x,y] and image[i+x,j+y] == se[x+sh,y+sw]):  
                            hit = True
            modified_image[i,j] = 255 if hit else 0
    return modified_image                       




image = cv2.imread('Images\Fingerprint.tif',0)

structuring_element_size = 3
structuring_element = np.ones((structuring_element_size,structuring_element_size),dtype=int)

structuring_element = np.uint8(structuring_element)

opening_img = Opening(image,structuring_element)
closing_img = Closing(opening_img,structuring_element)



plt.figure(figsize=(12,8))
plt.subplot(1,3,1)
plt.imshow(image,'gray')
plt.title('Original image')

plt.subplot(1,3,2)
plt.imshow(opening_img,'gray')
plt.title('Opening')

plt.subplot(1,3,3)
plt.imshow(closing_img,'gray')
plt.title('Closing')

plt.show()