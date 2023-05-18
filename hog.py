# Import the necessary libraries
import numpy as np
from skimage.io import imread
from skimage.feature import hog

# Load the image
image = imread("pic.jpg")

# Convert the image to grayscale
grayscale_image = image.mean(axis=2)

# Calculate the gradient of the image
gradient = np.gradient(grayscale_image)

# Calculate the orientation of each gradient
gradient_orientations = np.arctan2(gradient[0], gradient[1])

# Define the HOG parameters
# The number of orientations to use
orientations = 9
# The number of pixels per cell
pixels_per_cell = (8, 8)
# The number of cells per block
cells_per_block = (2, 2)

# Calculate the HOG features
hog_features = hog(grayscale_image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)

# Print the HOG features
print(hog_features)
