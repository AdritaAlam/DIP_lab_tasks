import cv2
import numpy as np 
import matplotlib.pyplot as plt 


def BoundaryExtraction(image,se):
    modified_image = image.copy()
    modified_image = modified_image - Erosion(modified_image,se)
    return modified_image

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





image = cv2.imread('Images\Lincoln.tif',0)

structuring_element_size = 3
structuring_element = np.ones((structuring_element_size,structuring_element_size),dtype=int)

structuring_element = np.uint8(structuring_element)


after_boundary_extraction = BoundaryExtraction(image,structuring_element)


plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.imshow(image,'gray')
plt.title('Original image')

plt.subplot(1,2,2)
plt.imshow(after_boundary_extraction,'gray')
plt.title('Boundary extraction')


plt.show()