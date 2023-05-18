# Import the necessary libraries
import cv2
import numpy as np
from PIL import Image

# Load the image
image = Image.open("pic.jpg").load().convert("L")

# Convert the image to grayscale
grayscale_image = np.array(image)

# Define the HOG parameters
# The number of orientations to use
orientations = 9
# The number of pixels per cell
pixels_per_cell = (8, 8)
# The number of cells per block
cells_per_block = (2, 2)

# Calculate the gradient of the image
# This calculates the magnitude and direction of the gradient at each pixel
gradient = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)

# Calculate the orientation of each gradient
# This calculates the angle of the gradient at each pixel
gradient_orientations = np.arctan2(gradient[:, :, 1], gradient[:, :, 0])

# Bin the gradient orientations into a histogram
# This creates a histogram of the gradient orientations at each cell
hog_features = np.zeros((image.shape[0] // pixels_per_cell[0], image.shape[1] // pixels_per_cell[1], orientations))
for i in range(hog_features.shape[0]):
    for j in range(hog_features.shape[1]):
        for k in range(orientations):
            hog_features[i, j, k] = np.sum(gradient_orientations[i * pixels_per_cell[0]:(i + 1) * pixels_per_cell[0], j * pixels_per_cell[1]:(j + 1) * pixels_per_cell[1]] == k)

# Visualize the HOG features
# This creates a heatmap of the HOG features
hog_visualization = np.zeros((image.shape[0], image.shape[1]))
for i in range(hog_features.shape[0]):
    for j in range(hog_features.shape[1]):
        for k in range(orientations):
            hog_visualization[i * pixels_per_cell[0] + pixels_per_cell[0] // 2, j * pixels_per_cell[1] + pixels_per_cell[1] // 2] += hog_features[i, j, k]

# Display the image and the HOG visualization
cv2.imshow("Image", image)
cv2.imshow("HOG Visualization", hog_visualization)
cv2.waitKey(0)
