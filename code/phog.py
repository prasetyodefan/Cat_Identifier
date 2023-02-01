import cv2
import numpy as np
from skimage.feature import pyramid_histogram_of_oriented_gradients

pathimg = '..\\Cat_Identifier\\asset\\dataset\\bengal55\\bengal (113).jpg'
# Load the image and convert it to grayscale
image = cv2.imread(pathimg, 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute the PHOG descriptor for the image
descriptor = pyramid_histogram_of_oriented_gradients(
    gray, orientations=8, visualize=False)

# Generate the histogram of oriented gradients
histogram = cv2.calcHist([descriptor], [0], None, [256], [0, 256])

# Normalize the histogram
cv2.normalize(histogram, histogram)

# Display the resulting histograms
cv2.imshow("PHOG Histogram", histogram)
cv2.waitKey(0)
