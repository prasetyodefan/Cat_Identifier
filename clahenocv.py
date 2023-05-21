import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

def clahe(image, clip_limit=0.03, grid_size=(4, 4), nbins=256):
    # Convert image to float
    image = image.astype(np.float32)

    # Calculate the size of each grid cell
    height, width = image.shape[:2]
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]

    # Calculate the number of pixels in each cell
    cell_pixels = cell_height * cell_width

    # Create a histogram for each grid cell
    histograms = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            cell = image[i * cell_height:(i + 1) * cell_height,
                         j * cell_width:(j + 1) * cell_width]
            histogram, _ = np.histogram(cell.flatten(), bins=nbins, range=(0, nbins-1))
            histograms.append(histogram)

    # Calculate the clip limit
    clip_limit = int(clip_limit * cell_pixels)

    # Clip and redistribute histogram bins
    redistributed = []
    for histogram in histograms:
        clipped_histogram = np.clip(histogram, 0, clip_limit)
        clipped_histogram = exposure.rescale_intensity(clipped_histogram, in_range=(0, clip_limit), out_range=(0, nbins-1))
        redistributed.append(clipped_histogram)

    # Combine redistributed histograms into a single LUT
    lut = np.concatenate(redistributed)

    # Apply the LUT to the image
    image_eq = lut[image.astype(np.uint8)]

    # Rescale the image to the original range
    image_eq = exposure.rescale_intensity(image_eq, in_range=(0, nbins-1), out_range=(0, 255))

    return image_eq.astype(np.uint8)


# Load an image using OpenCV
image = cv2.imread('pcc.jpg', cv2.IMREAD_GRAYSCALE)

# Apply CLAHE to the image
clahe_image = clahe(image)

# Visualize the original and CLAHE-enhanced images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(clahe_image, cmap='gray')
axes[1].set_title('CLAHE-enhanced Image')
axes[1].axis('off')
plt.show()
