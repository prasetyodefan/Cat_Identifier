import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load citra
image = cv2.imread('assets/Training/CPYY/rblue-205-_jpg.rf.18d9acadaf22c6bdf1257a2f3601315f.jpg')

# Konversi ke grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Menerapkan CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray_image)

# Menghaluskan citra dengan filter Gaussian
blurred_image = cv2.GaussianBlur(clahe_image, (5, 5), 0)

# Menghitung citra mask
mask = cv2.subtract(clahe_image, blurred_image)

# Menambahkan mask ke citra asli
unsharp_mask = cv2.add(clahe_image, mask)

# Menampilkan semua proses dalam plot
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Citra Asli')

plt.subplot(222)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale')

plt.subplot(223)
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE')

plt.subplot(224)
plt.imshow(unsharp_mask, cmap='gray')
plt.title('Unsharp Masking')

plt.tight_layout()
plt.show()
