import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, data
from PIL import Image

path = "..\\Cat_Identifier\\asset\\dataset\\bengal55\\bengal (1).jpg"

im = Image.open(path)

# Convert the image to grayscale, if it is not already in grayscale.
if im.mode != 'L':
    im = im.convert('L')







def compute_phog(image, levels, bins):
    phog = []
    histograms = []
    for i in range(levels):
        histograms.append([])
        phog.append([])
    # Compute gradient magnitude and orientation
    gx = filters.sobel_v(image)
    gy = filters.sobel_h(image)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    gradient_orientation = np.arctan2(gy, gx)
    gradient_orientation[gradient_orientation < 0] += np.pi
    
    height, width = image.shape
    block_size = (width//(2**levels), height//(2**levels))
    # Compute gradient histograms
    for i in range(height//block_size[1]):
        for j in range(width//block_size[0]):
            bin_idx = np.floor(bins * gradient_orientation[i*block_size[1]:(i+1)*block_size[1], j*block_size[0]:(j+1)*block_size[0]] / (2*np.pi))
            level = np.log2(max(block_size)) - np.log2(min(block_size, width//(j+1), height//(i+1)))
            for k in range(int(bins)):
                histograms[int(level)][i*width//block_size[0]+j][k] = np.sum(gradient_magnitude[(gradient_orientation >= 2*np.pi*k/bins) & 
                                                                                                (gradient_orientation < 2*np.pi*(k+1)/bins) & 
                                                                                                (bin_idx == k)])
    # Concatenate histograms into a single feature vector
    for i in range(levels):
        phog[i] = np.concatenate(histograms[i])
    return np.concatenate(phog)

# Compute the PHOG descriptor
phog_descriptor = compute_phog(im, 3, 8)

print("PHOG Descriptor Shape:", phog_descriptor.shape)

# Plot the original image
plt.figure(figsize=(5,5))
plt.imshow(image)
plt.axis('off')
plt.show()


