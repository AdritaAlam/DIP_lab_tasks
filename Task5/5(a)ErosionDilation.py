import cv2
import numpy as np 
import matplotlib.pyplot as plt 


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


after_erosion = Erosion(image,structuring_element)
after_dilation = Dilation(after_erosion,structuring_element)

plt.figure(figsize=(12,8))
plt.subplot(1,3,1)
plt.imshow(image,'gray')
plt.title('Original image')

plt.subplot(1,3,2)
plt.imshow(after_erosion,'gray')
plt.title('Image after erosion')

plt.subplot(1,3,3)
plt.imshow(after_dilation,'gray')
plt.title('Image after dilation')

plt.show()