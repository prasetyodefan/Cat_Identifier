import numpy as np
from PIL import Image


def otsu_threshold(image):
    histogram = np.histogram(image, bins=np.arange(257), range=(0, 256))[0]
    cumulative_sum = np.cumsum(histogram)
    cumulative_mean = np.cumsum(
        histogram * np.arange(256)) / (cumulative_sum + 1e-10)
    between_class_variance = cumulative_sum[:-1] * (
        cumulative_sum[-1] - cumulative_sum[:-1]) * (cumulative_mean[:-1] - cumulative_mean[-1])**2
    threshold = 100
    return threshold


def apply_threshold(image, threshold):
    return (image > threshold).astype(int)

def grayscale(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

# Load the image
image = Image.open('cropfaces.jpg')
image = np.array(image)
image = grayscale(image)

# Compute the threshold
threshold = otsu_threshold(image)

# Apply the threshold to the image
binary_image = apply_threshold(image, threshold)

# Convert binary image back to image object
#binary_image = Image.fromarray(binary_image.astype('uint8') * 255)

# Save the binary image
#binary_image.save('binary_image.jpg')

print( binary_image )