import cv2
import numpy as np

def phog(img, bin_size=9, levels=3):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate gradient magnitude and orientation
    gradient_magnitude, gradient_orientation = cv2.cartToPolar(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

    # Create a list to store the histograms at each scale
    histograms = []

    # Create the pyramid using cv2.pyrDown()
    pyramid = [gray]
    for i in range(levels-1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))

    # For each scale in the pyramid
    for i, level in enumerate(pyramid):
        # Create a histogram of gradient orientations
        histogram, _ = np.histogram(gradient_orientation[:level.shape[0], :level.shape[1]], bins=bin_size, range=(0, 180), weights=gradient_magnitude[:level.shape[0], :level.shape[1]])
        
        # Normalize the histogram
        histogram = histogram / np.sum(histogram)
        
        # Add the histogram to the list
        histograms.append(histogram)
        
    # Concatenate the histograms from all scales into a single feature vector
    phog_descriptor = np.concatenate(histograms)
    
    return phog_descriptor
readim = '..\\Cat_Identifier\\code\\phog code\\binary_image.jpg'
img = cv2.imread(readim,1)
phog_descriptor = phog(img)

cv2.imshow(phog_descriptor)