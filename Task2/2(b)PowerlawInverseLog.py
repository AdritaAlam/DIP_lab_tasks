import cv2 
import numpy as np 
import matplotlib.pyplot as plt 


def Power(image):
    modified_image = image.copy()
    gamma = 0.3
    modified_image = (np.power((modified_image/255.0),gamma)*255.0)
    return modified_image


def Inverse(image):
    modified_image = image.copy()
    c = (255.0/np.log(1+np.max(image)))
    modified_image = np.exp(image/c)-1
    return modified_image

image = cv2.imread('Images\Aerial.tif',0)

power_law_image = Power(image)
inverse_image = Inverse(image)
diff_image = cv2.absdiff(power_law_image,inverse_image)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title('Original image')

plt.subplot(2,2,2)
plt.imshow(power_law_image,'gray')
#plt.imshow(cv2.cvtColor(power_law_image, cv2.COLOR_BGR2RGB))
plt.title('Powerlaw image')

plt.subplot(2,2,3)
plt.imshow(inverse_image,'gray')
#plt.imshow(cv2.cvtColor(inverse_image,cv2.COLOR_BGR2RGB))
plt.title('Inverse log image')


plt.subplot(2,2,4)
plt.imshow(diff_image,'gray')
#plt.imshow(cv2.cvtColor(diff_image,cv2.cv2.COLOR_BGR2RGB))
plt.title('Difference')

plt.show()   