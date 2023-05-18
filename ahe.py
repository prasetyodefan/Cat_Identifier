import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

def clahe(image, clip_limit, grid_size):
    # Normalize the image to the range of -1 to 1
    image = image.astype(np.float32) / 255.0 * 2.0 - 1.0

    # Apply CLAHE to the normalized image
    clahe_image = exposure.equalize_adapthist(image, clip_limit=clip_limit, kernel_size=grid_size)

    return clahe_image

# Read the image
image = plt.imread('pcc.jpg')

# Convert the image to grayscale if it is not already
if image.ndim == 3:
    image = np.mean(image, axis=2)

# Apply CLAHE with clip limit 0.03 and grid size 8x8
clahe_image = clahe(image, clip_limit=.01, grid_size=8)

# Display the original image and the CLAHE image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE Image')

plt.show()
