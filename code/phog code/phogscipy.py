import numpy as np
from scipy import signal
from skimage import io

def phog(img, bin_size=9, levels=3):
    # Convert the image to grayscale
    gray = np.mean(img, axis=2)
    
    # Compute the gradient magnitude and orientation
    gradient_x = signal.convolve2d(gray, [[-1, 0, 1]], mode='same', boundary='symmetric')
    gradient_y = signal.convolve2d(gray, [[-1, 0, 1]].T, mode='same', boundary='symmetric')
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    orientation = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

    # Divide the gradient orientation into bins
    bin_edges = np.arange(0, 360, bin_size)
    digitized = np.digitize(orientation.ravel(), bin_edges)
    
    # Compute the histogram of gradient orientations for each level in the pyramid
    histograms = []
    for level in range(levels):
        # Downsample the image
        downscaled_magnitude = signal.resample(magnitude, magnitude.shape[0]//2, axis=0)
        downscaled_magnitude = signal.resample(downscaled_magnitude, downscaled_magnitude.shape[1]//2, axis=1)
        downscaled_orientation = signal.resample(orientation, orientation.shape[0]//2, axis=0)
        downscaled_orientation = signal.resample(downscaled_orientation, downscaled_orientation.shape[1]//2, axis=1)
        
        # Compute the histogram of gradient orientations
        histogram, _ = np.histogram(downscaled_orientation.ravel(), bins=bin_edges, weights=downscaled_magnitude.ravel())
        histograms.append(histogram)
        
        # Update the magnitude and orientation for the next level
        magnitude = downscaled_magnitude
        orientation = downscaled_orientation
    
    # Concatenate the histograms into a single feature vector
    feature_vector = np.concatenate(histograms)
    
    # Normalize the feature vector
    feature_vector = feature_vector / np.linalg.norm(feature_vector)
    
    return feature_vector

# Load an image using the skimage library
img = io.imread("cropfaces.jpg")

# Compute the PHOG feature vector for the image
feature_vector = phog(img)

# The feature vector can be used for further processing, such as classification or clustering.
print(feature_vector)