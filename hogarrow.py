import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_gradient(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
    return mag, ang

def calculate_histogram(ang, mag, nbins):
    hist = np.zeros(nbins)
    bin_width = 360 / nbins
    for i in range(ang.shape[0]):
        for j in range(ang.shape[1]):
            bin_index = int(ang[i, j] / bin_width)
            hist[bin_index] += mag[i, j]
    return hist

def calculate_hog(img, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    # Step 1: Calculate gradient image
    mag, ang = calculate_gradient(img)

    # Step 2: Divide image into cells
    cells = []
    for i in range(0, img.shape[0], cell_size[0]):
        for j in range(0, img.shape[1], cell_size[1]):
            cell_mag = mag[i:i+cell_size[0], j:j+cell_size[1]]
            cell_ang = ang[i:i+cell_size[0], j:j+cell_size[1]]
            cells.append((cell_mag, cell_ang))

    # Step 3: Calculate histogram for each cell
    cell_hists = []
    for cell_mag, cell_ang in cells:
        cell_hist = calculate_histogram(cell_ang, cell_mag, nbins)
        cell_hists.append(cell_hist)

    # Step 4: Divide cells into blocks and concatenate histograms
    blocks = []
    block_size_pixels = (cell_size[0] * block_size[0], cell_size[1] * block_size[1])
    for i in range(0, img.shape[0]-block_size_pixels[0]+1, cell_size[0]):
        for j in range(0, img.shape[1]-block_size_pixels[1]+1, cell_size[1]):
            block_hist = np.concatenate([
                cell_hists[(i//cell_size[0])*img.shape[1]//cell_size[1] + j//cell_size[1]],
                cell_hists[(i//cell_size[0])*img.shape[1]//cell_size[1] + (j+1)//cell_size[1]],
                cell_hists[((i+1)//cell_size[0])*img.shape[1]//cell_size[1] + j//cell_size[1]],
                cell_hists[((i+1)//cell_size[0])*img.shape[1]//cell_size[1] + (j+1)//cell_size[1]]
            ])
            blocks.append(block_hist)

    # Step 5: Normalize blocks using L2-norm
    blocks = np.array(blocks)
    block_norm = np.sqrt(np.sum(blocks**2, axis=1)) + 1e-6  # Add small epsilon to avoid division by zero
    blocks /= block_norm[:, np.newaxis]

    # Step 6: Concatenate normalized blocks into HOG descriptor
    hog_descriptor = blocks.flatten()

    return hog_descriptor


# Load an example image
image = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate HOG features
hog = calculate_hog(image)

# Reshape HOG features to match the image size
hog_reshaped = hog.reshape(image.shape[0] // 8 - 1, image.shape[1] // 8 - 1, -1)

# Calculate the center positions of each cell
cell_size = (8, 8)
cell_centers_x = np.arange(cell_size[1] // 2, image.shape[1], cell_size[1])
cell_centers_y = np.arange(cell_size[0] // 2, image.shape[0], cell_size[0])

# Create a grid of cell centers
grid_x, grid_y = np.meshgrid(cell_centers_x, cell_centers_y)

# Plot the image with HOG directions
plt.imshow(image, cmap='gray')
plt.quiver(grid_x, grid_y, np.cos(hog_reshaped[:,:,0]), np.sin(hog_reshaped[:,:,0]), angles='xy', scale_units='xy', scale=1, color='r')
plt.axis('off')
plt.show()
