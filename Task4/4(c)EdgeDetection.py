import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

def Gaussian(img):
    filtered_img = img*gaussian_highpass
    temp_img = np.abs(np.fft.ifft2(filtered_img))
    gaussian_highpass_img = temp_img/255
    return gaussian_highpass_img

def Ideal(img):
    filtered_img = img*lowpassmask
    temp_img = np.abs(np.fft.ifft2(filtered_img))
    ideal_highpass_img = temp_img/255
    return ideal_highpass_img

image = cv2.imread('Images\Characters.tif',0)
image = cv2.resize(image,(512,512))

mean = 7
stddev = 13
noise = np.random.normal(mean,stddev,image.shape).astype(np.uint8)

noisy_img = cv2.add(image,noise)

fft_img = np.fft.fftshift(np.fft.fft2(image))
fft_noisy_img = np.fft.fftshift(np.fft.fft2(noisy_img))

D0 = 40
n = 9
dimension = int(np.ceil(np.sqrt(n)))


row,col = fft_img.shape
D = np.zeros((row,col))

for u in range(row):
    for v in range(col):
        D[u,v] = np.sqrt((u-row/2)**2 + (v-col/2)**2)

gaussian_highpass = 1 - np.exp((-(D**2))/(2*(D0**2)))
gaussian_highpass_img = Gaussian(fft_img)
gaussian_highpass_noisy_img = Gaussian(fft_noisy_img)


lowpassmask = D>=D0
lowpass_img = Ideal(fft_img)
lowpass_noisy_img = Ideal(fft_noisy_img)

plt.figure(figsize=(12,8))
plt.subplot(2,3,1)
plt.imshow(image,'gray')
plt.title('Original image')

plt.subplot(2,3,2)
plt.imshow(lowpass_img,'gray')
plt.title('ideal highpass filtered image')

plt.subplot(2,3,3)
plt.imshow(gaussian_highpass_img,'gray')
plt.title('Gaussian highpass filtered image')

plt.subplot(2,3,4)
plt.imshow(noisy_img,'gray')
plt.title('Noisy image')

plt.subplot(2,3,5)
plt.imshow(lowpass_noisy_img,'gray')
plt.title('Ideal highpass filtered image')

plt.subplot(2,3,6)
plt.imshow(gaussian_highpass_noisy_img,'gray')
plt.title('Gaussian highpass filtered image')

plt.show()



