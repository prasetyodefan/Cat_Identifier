import skimage.io as io
import skimage.exposure as exposure
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

# Read the image using skimage.imread
_img = io.imread("assets/Pred_Result/Bengal/bengalC_6.jpg")

# Apply Adaptive Histogram Equalization (AHE) to the grayscale image
clahe_image = exposure.equalize_adapthist(_img, clip_limit=3)

# Apply Unsharp Masking to the AHE result
blurred = ndimage.gaussian_filter(clahe_image, sigma=1)
unsharp_maskk = clahe_image - 0.5 * blurred

# Plot the original image
plt.subplot(231)
plt.imshow(_img, cmap='gray')
plt.title('Original Image')

# Plot the histogram of the original image
plt.subplot(232)
plt.hist(_img.ravel(), bins=256, color='gray')
plt.title('Histogram (Original Image)')

# Plot the histogram of the unsharp masking result
plt.subplot(233)
plt.hist(unsharp_maskk.ravel(), bins=256, color='gray')
plt.title('Histogram (Unsharp Masking Result)')

# Plot the AHE result
plt.subplot(234)
plt.imshow(clahe_image, cmap='gray')
plt.title('AHE Result')

# Plot the histogram of the AHE result
plt.subplot(235)
plt.hist(clahe_image.ravel(), bins=256, color='gray')
plt.title('Histogram (CLAHE Result)')

# # Plot the unsharp masking result
# plt.subplot(235)
# plt.imshow(unsharp_maskk, cmap='gray')
# plt.title('Unsharp Masking Result')

# # Plot the histogram of the unsharp masking result
# plt.subplot(236)
# plt.hist(unsharp_maskk.ravel(), bins=256, color='gray')
# plt.title('Histogram (Unsharp Masking Result)')

# # Adjust spacing and display the plot
plt.tight_layout()
plt.show()
