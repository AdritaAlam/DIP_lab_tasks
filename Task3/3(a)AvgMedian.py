import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

def PSNR(image,modified_image,filename):
    image = np.array(image,dtype = np.float64)
    modified_image = np.array(modified_image,dtype = np.float64)

    MSE = np.mean((image-modified_image)**2)
    R = 255.0
    psnr_result = 20 * np.log10(R/(np.sqrt(MSE)))
    psnr_result = round(psnr_result,2)

    print('PSNR of ' + filename + 'is ' + str(psnr_result))

    return psnr_result


def Average_Filter(image,mask):
    avg_img = np.zeros_like(image)
    avg_fil = np.ones((mask,mask),dtype = np.float32) / (mask**2)

    for i in range(m):
        for j in range(n):
            neighbour_reg = image[max(0,i-mask//2) : min(m,i+mask//2+1),max(0,j-mask//2):min(n,j+mask//2+1)]

            sum = 0
            nh,nw = neighbour_reg.shape
            for x in range(nh):
                for y in range(nw):
                    sum+=neighbour_reg[x,y]*avg_fil[x,y]
            avg_img[i,j] = sum        
    avg_img = np.uint8(avg_img)
    return avg_img

def Median_Filter(image,mask):
    med_img = np.zeros_like(image)

    for i in range(m):
        for j in range(n):
            neighbour_reg = image[max(0,i-mask//2) : min(m,i+mask//2+1),max(0,j-mask//2):min(n,j+mask//2+1)]

            median_val = np.median(neighbour_reg)
            med_img[i,j] = median_val     
    med_img = np.uint8(med_img)
    return med_img


image = cv2.imread('Images\Characters.tif',0)
image = cv2.resize(image,(512,512))



nop = int(input('Enter the number of points for adding salt and pepper noise: '))
mask_size = int(input('Enter the size of mask for filtering '))
m,n = image.shape
noisy_image = image.copy()

#   salt
for i in range(nop):
    x = np.random.randint(0,m-1)
    y = np.random.randint(0,n-1)

    noisy_image[x,y] = 255

# pepper
for i in range(nop):
    x = np.random.randint(0,m-1)
    y = np.random.randint(0,n-1)

    noisy_image[x,y] = 0


after_average_filtering = Average_Filter(noisy_image,mask_size)  
after_median_filtering = Median_Filter(noisy_image,mask_size)  

psnr_1 = PSNR(image,noisy_image,'Noisy_image')
psnr_2 = PSNR(image,after_average_filtering,'Average_filtered_image')
psnr_3 = PSNR(image,after_median_filtering,'Median_filtered_image')



plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title('Original image')

plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(noisy_image,cv2.COLOR_BGR2RGB))
plt.title(f'Noisy image with psnr: {psnr_1}')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(after_average_filtering,cv2.COLOR_BGR2RGB))
plt.title(f'After using average image with psnr: {psnr_2}')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(after_median_filtering,cv2.COLOR_BGR2RGB))
plt.title(f'After using median image with psnr: {psnr_3}')
plt.axis('off')

plt.show()
