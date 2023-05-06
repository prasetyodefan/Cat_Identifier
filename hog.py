import numpy as np
from scipy.ndimage import sobel
from skimage.feature import hog

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

# Test HOG calculation on an example image
image = np.random.randint(0, 255, (128, 128)).astype(np.uint8)
nbins = 12  # Change the number of bins
cell_size = (16, 16)  # Change the cell size
block_size = (3, 3)  # Change the block size
hog = calculate_hog(image, cell_size=cell_size, block_size=block_size, nbins=nbins)

print(hog.shape)  # Output: Varies based on the number of bins, cell size, and block size


# from skimage.feature import hog

# # Menghitung HOG descriptor dari citra yang telah diregangkan (stretched_img)
# hog_features, hog_image = hog(stretched_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

# # Menampilkan citra HOG
# plt.imshow(hog_image, cmap='gray')
# plt.axis('off')
# plt.show()