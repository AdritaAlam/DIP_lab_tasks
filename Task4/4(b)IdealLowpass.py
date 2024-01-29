import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

def CalculateD(image):
    M,N = image.shape
    D = np.zeros((M,N))
    for u in range(M):
        for v in range(N):
            D[u,v] = np.sqrt((u-M/2)** 2 + (v-N/2)**2)
    return D        



image  = cv2.imread('Images\Characters.tif',0)
image =cv2.resize(image,(512,512))


mean = 10
stddev = 25
noise = np.random.normal(mean,stddev,image.shape).astype(np.uint8)
noisy_img = cv2.add(image,noise)

fft_img = np.fft.fftshift(np.fft.fft2(noisy_img))

D0 = 5
n = 9
dimension = int(np.ceil(np.sqrt(n)))
D = CalculateD(fft_img)

for i in range(n):
    lowpassmask = D<=D0
    filtered_img = fft_img * lowpassmask
    temp_img = np.abs(np.fft.ifft2(filtered_img))
    lowpass_img = temp_img

    plt.subplot(dimension,dimension,i+1)
    plt.imshow(lowpass_img,'gray')
    #plt.imshow(cv2.cvtColor(lowpass_img,cv2.COLOR_BGR2RGB))
    plt.title(f'With radius D0 = {D0}')
    plt.axis('off')
    D0 = D0 + 5

plt.show()    

