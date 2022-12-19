import numpy as np
import cv2


def otsu_threshold(image):
    """Determine the optimal threshold value for binarizing an image using Otsu's method.

    Parameters
    ----------
    image : numpy array
        The image to be thresholded.

    Returns
    -------
    int
        The optimal threshold value.
    """
    # Compute the histogram of the image
    histogram, _ = np.histogram(image, bins=256, range=(0, 255))

    # Compute the cumulative sum of the histogram
    cdf = histogram.cumsum()

    # Normalize the histogram
    cdf = 255 * cdf / cdf[-1]

    # Initialize variables for the loop
    optimal_threshold = 0
    optimal_variance = 0
    nbins = len(histogram)

    # Loop through all possible threshold values
    for i in range(nbins):
        # Compute the probability of each class
        p_0 = cdf[i] / cdf[-1]
        p_1 = 1 - p_0

        # Compute the means of each class
        mean_0 = sum(i * histogram[:i]) / cdf[i]
        mean_1 = sum((i - mean_0) * histogram[i:]) / (cdf[-1] - cdf[i])

        # Compute the variance of the classes
        variance = p_0 * p_1 * (mean_0 - mean_1) ** 2

        # If the variance is greater than the current optimal variance, update the optimal threshold
        if variance > optimal_variance:
            optimal_variance = variance
            optimal_threshold = i

    return optimal_threshold


# Load the image and convert it to grayscale
image = '..\\Cat_Identifier\\asset\\dataset\\bengal55\\bengal (113).jpg'
image = cv2.imread(image, 1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Determine the optimal threshold value
threshold = otsu_threshold(image)

# Binarize the image using the threshold value
image_binarized = np.where(image > threshold, 255, 0)

cv2.imshow('Image', threshold)
cv2.waitKey(0)
