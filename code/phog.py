import cv2
import numpy as np


def compute_phog(img, num_bins=9, levels=3):
    # Compute the HOG descriptor for the image
    hog = cv2.HOGDescriptor()
    hog_descriptor = hog.compute(img, winStride=(8, 8), padding=(8, 8))

    # Initialize the PHOG descriptor with the HOG descriptor
    phog_descriptor = hog_descriptor

    # Compute the PHOG descriptor at each level of the pyramid
    for level in range(1, levels):
        # Resize the image to the next level of the pyramid
        resized_image = cv2.resize(
            img, None, fx=1/pow(2, level), fy=1/pow(2, level))

        # Compute the HOG descriptor for the resized image
        hog_descriptor = hog.compute(
            resized_image, winStride=(8, 8), padding=(8, 8))

        # Concatenate the HOG descriptor to the PHOG descriptor
        phog_descriptor = np.concatenate((phog_descriptor, hog_descriptor))

    return phog_descriptor


path = '..\\Cat_Identifier\\asset\\dataset\\bengal55\\bengal (113).jpg'
img = cv2.imread(path, 1)
# Perform Canny edge detection
phogg = compute_phog(img)

cv2.imshow('Edges', phogg)
