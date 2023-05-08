from skimage.feature import hog
import cv2
from skimage import data, exposure
import matplotlib.pyplot as plt

# Load an example image
image = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the HOG features and visualization
hog_features, hog_image = hog(image, visualize=True)

# Plot the original image and its HOG visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Original Image')

ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('HOG Visualization')

plt.show()
