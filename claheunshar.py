import cv2
import numpy as np
from skimage import filters, feature
from skimage import exposure
from matplotlib import pyplot as plt

# Membaca citra
image = cv2.imread('pic.jpg', cv2.IMREAD_COLOR)

# Mengkonversi citra ke citra grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Mengaplikasikan Adaptive Histogram Equalization (AHE) pada citra grayscale
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray_image)

# Menerapkan Unsharp Masking pada citra hasil AHE
blurred = cv2.GaussianBlur(clahe_image, (5, 5), 0)
unsharp_mask = cv2.addWeighted(clahe_image, 1.5, blurred, -0.5, 0)

# Melakukan segmentasi dengan Thresholding
threshold_value = filters.threshold_otsu(unsharp_mask)
binary_image = np.uint8(unsharp_mask > threshold_value) * 255

# Menghitung Histogram of Oriented Gradients (HOG)
def calculate_gradient(img):
    sobel_x = sobel(img, axis=1)
    sobel_y = sobel(img, axis=0)
    mag = np.sqrt(sobel_x**2 + sobel_y**2)
    ang = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)
    return mag, ang

def calculate_histogram(ang, mag, nbins):
    hist = np.zeros(nbins)
    bin_width = 180 / nbins
    flattened_ang = ang.flatten()
    flattened_mag = mag.flatten()
    for i in range(len(flattened_ang)):
        bin_index = int((flattened_ang[i] + 90) / bin_width)
        hist[bin_index] += flattened_mag[i]
    return hist

def calculate_hog(img, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    # Step 1: Calculate gradient image
    mag, ang = calculate_gradient(img)

    # Step 2: Divide image into cells
    num_cells_y = img.shape[0] // cell_size[0]
    num_cells_x = img.shape[1] // cell_size[1]
    cells = []
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            cell_mag = mag[i*cell_size[0]:(i+1)*cell_size[0], j*cell_size[1]:(j+1)*cell_size[1]]
            cell_ang = ang[i*cell_size[0]:(i+1)*cell_size[0], j*cell_size[1]:(j+1)*cell_size[1]]
            cells.append((cell_mag, cell_ang))

    # Step 3: Calculate histogram for each cell
    cell_hists = []
    for cell_mag, cell_ang in cells:
        cell_hist = calculate_histogram(cell_ang, cell_mag, nbins)
        cell_hists.append(cell_hist)

    # Step 4: Divide cells into blocks and concatenate histograms
    blocks = []
    block_size_cells_y = block_size[0]
    block_size_cells_x = block_size[1]
    for i in range(num_cells_y - block_size_cells_y + 1):
        for j in range(num_cells_x - block_size_cells_x + 1):
            block_hist = np.concatenate([
                cell_hists[(i+k)*num_cells_x + j+l]
                for k in range(block_size_cells_y)
                for l in range(block_size_cells_x)
            ])
            blocks.append(block_hist)

    # Step 5: Normalize blocks using L2-norm
    blocks = np.array(blocks)
    block_norm = np.sqrt(np.sum(blocks**2, axis=1)) + 1e-6  # Add small epsilon to avoid division by zero
    blocks /= block_norm[:, np.newaxis]

    # Step 6: Concatenate normalized blocks into HOG descriptor
    hog_descriptor = blocks.flatten()

    return hog_descriptor

hog_image = feature.hog(unsharp_mask, orientations=9, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

# Menampilkan citra asli, citra hasil AHE, citra hasil Unsharp Masking,
# citra hasil segmentasi, dan visualisasi HOG
titles = ['Original', 'AHE', 'Unsharp Masking', 'Segmentation', 'HOG']
images = [image, clahe_image, unsharp_mask, binary_image, hog_image]

for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i], cmap='pink')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
