import cv2
import numpy as np

# Load the image and convert it to grayscale
image = cv2.imread("catt.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute the HOG descriptor for the image
hog = cv2.HOGDescriptor()
descriptor = hog.compute(gray)

# Reshape the descriptor into a 4-level pyramid
pyramid = np.reshape(descriptor, (4, -1))

# Generate the histogram of oriented gradients
histogram = cv2.calcHist(pyramid, [0, 1, 2, 3], None, [4, 4, 4, 4], [
                         0, 180, 0, 256, 0, 256, 0, 256])

# Normalize the histogram
cv2.normalize(histogram, histogram)

# Display the resulting histogram
cv2.imshow("HOG Pyramid Histogram", histogram)
cv2.waitKey(0)
