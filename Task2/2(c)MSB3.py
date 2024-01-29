import cv2
import numpy as np 
import matplotlib.pyplot as plt 

image = cv2.imread('Images\Dollar.tif',0)
#image = cv2.resize(image,(512,512))

modified_image = image & 224
diff_img = cv2.absdiff(modified_image,image)


plt.subplot(3,1,1)
plt.imshow(image,'gray')
plt.title('Original image')
plt.axis('off')

plt.subplot(3,1,2)
plt.imshow(modified_image,'gray')

plt.title('Image using last 3bits')
plt.axis('off')

plt.subplot(3,1,3)
plt.imshow(cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB))
plt.title('Difference image')
plt.axis('off')



plt.show()        


