import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_histogram(image, num_bins):
    histogram = np.zeros(num_bins, dtype=np.uint32)
    for pixel in image.flatten():
        histogram[pixel] += 1
    return histogram

def clip_histogram(histogram, clip_limit):
    clipped_histogram = np.copy(histogram)
    total_excess = 0
    for i in range(len(histogram)):
        excess = histogram[i] - clip_limit
        if excess > 0:
            clipped_histogram[i] = clip_limit
            total_excess += excess
    return clipped_histogram, total_excess

def redistribute_histogram(clipped_histogram, total_excess, num_bins):
    num_pixels = clipped_histogram.sum()
    excess_per_bin = total_excess // num_bins
    for i in range(len(clipped_histogram)):
        clipped_histogram[i] += excess_per_bin
    remaining_excess = total_excess - excess_per_bin * num_bins
    for i in range(int(remaining_excess)):  # Convert to integer
        clipped_histogram[i] += 1
    return clipped_histogram


def compute_cdf(histogram):
    cdf = np.cumsum(histogram)
    return cdf

def interpolate_cdf(cdf_values, tile_indices, tile_size):
    min_row = tile_indices[0][0]
    min_col = tile_indices[0][1]
    max_row = tile_indices[-1][0]
    max_col = tile_indices[-1][1]

    num_pixels = tile_size[0] * tile_size[1]
    interpolated_cdf = cdf_values[0] + ((cdf_values[-1] - cdf_values[0]) / num_pixels) * (np.arange(num_pixels).reshape(tile_size) - min_row * tile_size[0] - min_col)

    return interpolated_cd
w
def apply_clahe(image, clip_limit, tile_size):
    # Resize image to ensure even division
    height, width = image.shape[:2]
    new_height = (height // tile_size[0]) * tile_size[0]
    new_width = (width // tile_size[1]) * tile_size[1]
    image = cv2.resize(image, (new_width, new_height))

    num_bins = 256

    # Divide image into tiles
    num_rows = new_height // tile_size[0]
    num_cols = new_width // tile_size[1]
    tiles = np.split(image, num_rows, axis=0)
    tiles = [np.split(tile, num_cols, axis=1) for tile in tiles]

    # Compute and clip histograms for each tile
    histograms = []
    total_excess = 0
    for row in tiles:
        row_histograms = []
        for tile in row:
            histogram = compute_histogram(tile, num_bins)
            clipped_histogram, excess = clip_histogram(histogram, clip_limit)
            row_histograms.append(clipped_histogram)
            total_excess += excess
        histograms.append(row_histograms)

    # Redistribute clipped histograms
    for row in histograms:
        for histogram in row:
            redistributed_histogram = redistribute_histogram(histogram, total_excess, num_bins)
            histogram[:] = redistributed_histogram

    # Compute cumulative distribution function (CDF)
    cdfs = []
    for row in histograms:
        row_cdfs = []
        for histogram in row:
            cdf = compute_cdf(histogram)
            row_cdfs.append(cdf)
        cdfs.append(row_cdfs)

    # Interpolate CDFs
    equalized_tiles = []
    for i in range(num_rows):
        equalized_row = []
        for j in range(num_cols):
            min_row = max(i - 1, 0)
            min_col = max(j - 1, 0)
            max_row = min(i + 1, num_rows - 1)
            max_col = min(j + 1, num_cols - 1)
            tile_indices = [(min_row, min_col), (min_row, j), (min_row, max_col),
                            (i, min_col), (i, j), (i, max_col),
                            (max_row, min_col), (max_row, j), (max_row, max_col)]
            cdf_values = [cdfs[row][col] for row, col in tile_indices]
            interpolated_cdf = interpolate_cdf(cdf_values, tile_indices, tile_size)
            equalized_tile = np.interp(tiles[i][j].flatten(), interpolated_cdf.flatten(), np.arange(num_bins))
            equalized_tile = equalized_tile.reshape(tile_size)
            equalized_row.append(equalized_tile)
        equalized_tiles.append(equalized_row)

    # Merge equalized tiles into the final enhanced image
    equalized_image = np.vstack([np.hstack(row) for row in equalized_tiles])

    return equalized_image



# Rest of the code remains the same



# Load image
image = plt.imread('pcc.jpg')

# Apply CLAHE
clip_limit = 2.0
tile_size = (8, 8)
equalized_image = apply_clahe(image, clip_limit, tile_size)

# Display or save the equalized image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('CLAHE Image')

plt.show()
