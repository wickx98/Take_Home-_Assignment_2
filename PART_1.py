import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create an image with 2 objects and a total of 3-pixel values
img = np.zeros((100, 100), dtype=np.uint8)
img[20:40, 20:40] = 1
img[60:80, 60:80] = 2

# Add Gaussian noise to the image
gauss_noise=np.zeros((100,100),dtype=np.uint8)
cv2.randn(gauss_noise,0.6,0.6)
gauss_noise=(gauss_noise*0.5).astype(np.uint8)

#noise = np.random.normal(0, 1, img.shape)
noisy_img =  gauss_noise +  img

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(noisy_img,(5,5),1000)
threshold_value, thresholded_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Display the results
fig, axs = plt.subplots(1, 3, figsize=(10, 10))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(noisy_img, cmap='gray')
axs[1].set_title('Noisy Image')
axs[2].imshow(thresholded_img, cmap='gray')
axs[2].set_title('Thresholded Image (Otsu)')

plt.show()