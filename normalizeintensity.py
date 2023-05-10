import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar grayscale
image = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE)

# Metode Min-Max Scaling
min_val = np.min(image)
max_val = np.max(image)
min_max_normalized = (image - min_val) / (max_val - min_val)

# Metode Z-Score Scaling
mean_val = np.mean(image)
std_val = np.std(image)
z_score_normalized = (image - mean_val) / std_val

# Metode Decimal Scaling
decimal_factor = 10**np.ceil(np.log10(np.max(image)))
decimal_scaled = image / decimal_factor

# Metode Log Transform
log_transformed = np.log1p(image)

# Metode Power-Law Transform
gamma = 0.5
power_transformed = np.power(image / 255.0, gamma)
power_transformed = np.uint8(power_transformed * 255)

# Menampilkan gambar dalam bentuk panel
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Gambar asli
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original')

# Min-Max Scaling
axs[0, 1].imshow(min_max_normalized, cmap='gray')
axs[0, 1].set_title('Min-Max Scaling')

# Z-Score Scaling
axs[0, 2].imshow(z_score_normalized, cmap='gray')
axs[0, 2].set_title('Z-Score Scaling')

# Decimal Scaling
axs[1, 0].imshow(decimal_scaled, cmap='gray')
axs[1, 0].set_title('Decimal Scaling')

# Log Transform
axs[1, 1].imshow(log_transformed, cmap='gray')
axs[1, 1].set_title('Log Transform')

# Power-Law Transform
axs[1, 2].imshow(power_transformed, cmap='gray')
axs[1, 2].set_title('Power-Law Transform')

plt.tight_layout()
plt.show()
