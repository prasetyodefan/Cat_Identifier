

# Resize: 
# Grayscale: 
# Konversi ke nilai tertentu:
# Normalisasi:
# Denoising:
# Contrast stretching:

from PIL import Image
import numpy as np
from scipy import ndimage
from skimage import exposure

#! Load citra
img = Image.open('pic.jpg')

#! Resize citra
resized_img = img.resize((64, 64))

#! Konversi citra ke grayscale
gray_img = resized_img.convert('L')

#! Konversi citra ke nilai tertentu
threshold = 150
binary_img = gray_img.point(lambda p: p > threshold and 255)

#! Normalisasi citra
normalized_img = np.array(binary_img) / 255.0

#! Denoising citra
denoised_img = ndimage.median_filter(normalized_img, size=3)

#! Contrast stretching citra
p2, p98 = np.percentile(denoised_img, (2, 98))
stretched_img = exposure.rescale_intensity(denoised_img, in_range=(p2, p98))

np.set_printoptions(threshold=np.inf)
print(stretched_img)
