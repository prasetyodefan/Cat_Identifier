import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import numpy as np

def unsharp(image, sigma, strength):
    # Median filtering
    image_mf = median_filter(image, size=sigma)

    # Calculate the Laplacian
    lap = cv2.Laplacian(image_mf, cv2.CV_64F)

    # Calculate the sharpened image
    sharp = image - strength * lap

    return sharp

original_image = plt.imread('pic.jpg')

sharp1 = np.zeros_like(original_image)
for i in range(3):
    sharp1[:, :, i] = unsharp(original_image[:, :, i], 5, 0.3)

# Show the original image
plt.subplot(121)
plt.imshow(original_image)
plt.title('Original Image')

# Show the sharpened image
plt.subplot(122)
plt.imshow(sharp1)
plt.title('Sharpened Image')

# Display the figure
plt.show()

# # Grayscale img --------------------------------------------------------------------
# import cv2
# import matplotlib.pyplot as plt
# from scipy.ndimage.filters import median_filter
# import numpy as np

# original_image = plt.imread('pic.jpg').astype('uint16')

# # Convert to grayscale
# gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# # Median filtering
# gray_image_mf = median_filter(gray_image, 1)

# # Calculate the Laplacian
# lap = cv2.Laplacian(gray_image_mf,cv2.CV_64F)

# # Calculate the sharpened image
# sharp = gray_image - 0.7*lap 

# # Show the original image
# plt.subplot(231)
# plt.imshow(original_image, cmap='gray')
# plt.title('Original Image')

# # Show the grayscale image
# plt.subplot(232)
# plt.imshow(gray_image, cmap='gray')
# plt.title('Grayscale Image')

# # Show the median-filtered image
# plt.subplot(233)
# plt.imshow(gray_image_mf, cmap='gray')
# plt.title('Median-Filtered Image')

# # Show the Laplacian
# plt.subplot(234)
# plt.imshow(lap, cmap='gray')
# plt.title('Laplacian')

# # Show the sharpened image
# plt.subplot(235)
# plt.imshow(sharp, cmap='gray')
# plt.title('Sharpened Image')

# # Adjust the spacing between subplots
# plt.tight_layout()

# # Display the figure
# plt.show()
