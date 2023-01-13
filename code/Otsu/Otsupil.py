import numpy as np
from PIL import Image

# Open the image
im = Image.open('cropfaces.jpg')

# Convert the image to grayscale, if it is not already in grayscale.
if im.mode != 'L':
    im = im.convert('L')

# Compute the histogram of the image
histogram = np.histogram(im, bins=range(256))[0]

# Initialize the threshold value and maximum inter-class variance
threshold = 0
max_inter_class_variance = 0

# Iterate over all possible threshold values
for t in range(256):
    # Calculate the probability of the foreground pixels
    p_fg = np.sum(histogram[:t]) / np.sum(histogram)
    # Calculate the probability of the background pixels
    p_bg = 1 - p_fg
    # Calculate the mean of the foreground pixels
    mean_fg = np.sum(np.arange(t) * histogram[:t]) / np.sum(histogram[:t])
    # Calculate the mean of the background pixels
    mean_bg = np.sum(np.arange(t, 255) * histogram[t:]) / np.sum(histogram[t:])
    # Calculate the inter-class variance
    inter_class_variance = p_fg * p_bg * (mean_fg - mean_bg) ** 2
    # Update the threshold value and maximum inter-class variance if a threshold value with a higher inter-class variance is found
    if inter_class_variance > max_inter_class_variance:
        threshold = t
        max_inter_class_variance = inter_class_variance

# Threshold the image using the optimal threshold value
im = im.point(lambda x: 255 if x > threshold else 0)

# Save the binary image
im.save("..\\Cat_Identifier\\code\\Otsu\\binary_image.jpg")

# Work
