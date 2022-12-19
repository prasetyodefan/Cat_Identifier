import cv2
import numpy as np


def canny_edge_detection(image, sigma=0.2):
    """
    Perform Canny edge detection on an image.

    Parameters
    ----------
    image : numpy array
        Input image.
    sigma : float, optional
        Standard deviation of the Gaussian blur, by default 0.33

    Returns
    -------
    numpy array
        Output image with edges highlighted.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the median of the single channel pixel intensities
    v = np.median(gray)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(100, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lower, upper)

    # Return the edges image
    return edges


# Read in the image

pathimg = '..\\Cat_Identifier\\asset\\dataset\\bengal55\\bengal (78).jpg'

image = cv2.imread(pathimg, 1)

# Perform Canny edge detection
edges = canny_edge_detection(image)

# Display the edges image
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
