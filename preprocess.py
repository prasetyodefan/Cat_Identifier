

# Resize: 
# Grayscale: 
# Konversi ke nilai tertentu:
# Normalisasi:
# Denoising:
# Contrast stretching:

from PIL import Image
import numpy as np
from scipy import ndimage
from skimage import exposure

#! Load citra
img = Image.open('pic1.jpg')

#! Resize citra
resized_img = img.resize((258, 258))

#! Konversi citra ke grayscale
gray_img = resized_img.convert('L')

#! Konversi citra ke nilai tertentu
threshold = 150
binary_img = gray_img.point(lambda p: p > threshold and 255)

#! Normalisasi citra
normalized_img = np.array(binary_img) / 255.0

#! Denoising citra
denoised_img = ndimage.median_filter(normalized_img, size=3)

#! Contrast stretching citra
p2, p98 = np.percentile(denoised_img, (2, 98))
stretched_img = exposure.rescale_intensity(denoised_img, in_range=(p2, p98))

# np.set_printoptions(threshold=np.inf)
# print(denoised_img)
image_array_scaled = (stretched_img * 255).astype(np.uint8)
imagee = Image.fromarray(image_array_scaled)
# imagee.show()
# img.show()

from skimage.feature import hog
import cv2
from skimage import data, exposure
import matplotlib.pyplot as plt

# Load an example image
image = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE)
print(image)
# Calculate the HOG features and visualization
hog_features ,hog_image = hog(imagee, orientations=8, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), block_norm='L2', visualize=True)

# Plot the original image and its HOG visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Original Image')

ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('HOG Visualization')

plt.show()
