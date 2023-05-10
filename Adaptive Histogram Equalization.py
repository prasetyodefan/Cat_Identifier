import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# Membaca gambar
image = cv2.imread('pic1.jpg', cv2.IMREAD_GRAYSCALE)

# Metode Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_equalized = clahe.apply(image)

# Menampilkan gambar dalam bentuk panel
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Gambar asli
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original')

# Adaptive Histogram Equalization
axs[0, 1].imshow(clahe_equalized, cmap='gray')
axs[0, 1].set_title('Adaptive Histogram Equalization')


plt.tight_layout()
plt.show()
