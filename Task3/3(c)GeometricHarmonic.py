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

    print('PSNR of ' + filename + ' is ' + str(psnr_result))

    return psnr_result


def Geometric(image,mask):
    geo_val = 1.0 / (mask**2)
    filtered_img = np.zeros_like(image,dtype=np.float32)
    for i in range(m):
        for j in range(n):
            neighbour_reg = image[max(0,i-mask//2):min(m,i+mask//2+1),max(0,j-mask//2):min(n,j+mask//2+1)]
            nh,nw = neighbour_reg.shape
            product = 1
            geo_mean = 1
            for x in range(nh):
                for y in range(nw):
                    pixel = neighbour_reg[x,y]
                    if (pixel!=0):
                        product = pixel**geo_val
                        geo_mean*=product
            filtered_img[i,j] = geo_mean
    filtered_img = np.uint8(filtered_img)
    return filtered_img 

def Harmonic(image,mask):
    filtered_img = np.zeros_like(image,dtype=np.float32)
    #har_val = mask ** 2
    for i in range(m):
        for j in range(n):
            neighbour_reg = image[max(0,i-mask//2):min(m,i+mask//2+1),max(0,j-mask//2):min(n,j+mask//2+1)]
            nh,nw = neighbour_reg.shape
            har_val = 0
            for x in range(nh):
                for y in range(nw):
                    pixel = neighbour_reg[x,y]
                    #if (pixel != 0):
                    har_val+=(1.0/(pixel+1e-4))
            har_mean =  (mask*mask)/har_val            
            filtered_img[i,j] = har_mean
    filtered_img = np.uint8(filtered_img)
    return filtered_img 
                 

image = cv2.imread('Images\SS.png',0)
image = cv2.resize(image,(512,512))


nop = int(input('Enter the number of points for adding salt and pepper noise: '))
mask = int(input('Enter the size of the mask: '))
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

m,n = image.shape
geometric_img = Geometric(noisy_image,mask)   
harmonic_img = Harmonic(noisy_image,mask) 


psnr_1 = PSNR(image,noisy_image,'Noisy_image')
psnr_2 = PSNR(image,geometric_img,'Geometric_mean_image')
psnr_3 = PSNR(image,harmonic_img,'Harmonic_mean_image')



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
plt.imshow(cv2.cvtColor(geometric_img,cv2.COLOR_BGR2RGB))
plt.title(f'Geometric image with psnr: {psnr_2}')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(harmonic_img,cv2.COLOR_BGR2RGB))
plt.title(f'Harmonic mean image with psnr: {psnr_3}')
plt.axis('off')

plt.show()