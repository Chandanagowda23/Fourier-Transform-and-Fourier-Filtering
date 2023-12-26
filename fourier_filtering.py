import cv2
import numpy as np
#____________________________________________________________________#

# ### Convert gray-scale image to Fourier domain: 

image = cv2.imread('Noisy_image.png', cv2.IMREAD_GRAYSCALE)

image_pad = np.zeros((426,408))
image_pad[:image.shape[0], :image.shape[1]] = image
 
dummy_image = np.zeros((426,408))

for i in range(dummy_image.shape[0]):
    for j in range(dummy_image.shape[1]):
        dummy_image[i, j] = image_pad[i, j] * (-1) ** (i + j)


fourier_shift = np.fft.fft2(dummy_image)
visual_purpose = 15*np.log(np.abs(fourier_shift))

cv2.imwrite('converted_fourier.png', visual_purpose)


#____________________________________________________________________#

# ### Gaussian Filter smoothing: 

image = cv2.imread('Noisy_image.png', cv2.IMREAD_GRAYSCALE)

image_pad = np.zeros((426,408))
image_pad[:image.shape[0], :image.shape[1]] = image
 
dummy_image = np.zeros((426,408))

for i in range(dummy_image.shape[0]):
    for j in range(dummy_image.shape[1]):
        dummy_image[i, j] = image_pad[i, j] * (-1) ** (i + j)

fourier_shift = np.fft.fft2(dummy_image)

dummy = np.zeros((image_pad.shape[0], image_pad.shape[1]),dtype=np.float32)

max_size = 10
for i in range(image_pad.shape[0]):
    for j in range(image_pad.shape[1]):
        size = np.sqrt((i-image_pad.shape[0]/2)**2 + (j-image_pad.shape[1]/2)**2)
        if size <= max_size**2:
            dummy[i,j] = 1
        else:
             dummy[i,j] = 0
              
low_frequency_shift = fourier_shift  * dummy
inverse_fourier_shift = np.fft.ifftshift(low_frequency_shift)
shifting_back = np.abs(np.fft.ifft2(inverse_fourier_shift))

image_without_pad = shifting_back[:image.shape[0], :image.shape[1]]

cv2.imwrite('gaussian_fourier.png', image_without_pad)

#____________________________________________________________________#
