import cv2 
import numpy as np 
import matplotlib.pyplot as plt 


def Butterworth(image):
    M,N = fft_img.shape
    H = np.zeros((M,N),dtype = np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2) ** 2 + (v-N/2)**2)
            H[u,v] = (1.0/(1 + (D/D0)**(2*n)))

    G = fft_img * H
    filtered_img = np.fft.ifft2(np.fft.ifftshift(G))
    filtered_img = np.abs(filtered_img)
    return filtered_img   

def Gaussian(image):
    M,N = fft_img.shape
    H = np.zeros((M,N),dtype = np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2) ** 2 + (v-N/2)**2)
            H[u,v] = np.exp((-(D**2))/(2*(D0**2)))

    G = fft_img * H
    filtered_img = np.abs(np.fft.ifft2(G))
    return filtered_img          

image = cv2.imread('Images\Characters.tif',0)
image = cv2.resize(image,(512,512))


#m,n = image.shape

mean = 7
stddev = 13
noise = np.random.normal(mean,stddev,image.shape).astype(np.uint8)
noisy_img = cv2.add(image,noise)


#fft 
fft_img = np.fft.fftshift(np.fft.fft2(noisy_img))

D0 = 25
n = 2

butter_img = Butterworth(fft_img)
gauss_img = Gaussian(fft_img)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title('Original image')

plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(noisy_img,cv2.COLOR_BGR2RGB))
plt.title('Noisy image')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(butter_img,'gray')
#plt.imshow(cv2.cvtColor(butter_img,cv2.COLOR_BGR2RGB))
plt.title(f'Butterworth lowpass ')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(gauss_img,'gray')
#plt.imshow(cv2.cvtColor(gauss_img,cv2.COLOR_BGR2RGB))
plt.title(f'Gaussian lowpass')
plt.axis('off')

plt.show()