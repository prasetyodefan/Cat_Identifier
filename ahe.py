import numpy as np
from PIL import Image

def adaptive_histogram_equalization(image, tile_size):
    # Convert the image to grayscale if it's in RGB format
    if image.mode == 'RGB':
        image = image.convert('L')

    # Get the image size
    width, height = image.size

    # Calculate the number of rows and columns for tiles
    num_rows = height // tile_size
    num_cols = width // tile_size

    # Initialize the enhanced image
    enhanced_image = Image.new('L', (width, height))

    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the tile boundaries
            start_x = col * tile_size
            end_x = start_x + tile_size
            start_y = row * tile_size
            end_y = start_y + tile_size

            # Extract the tile from the image
            tile = image.crop((start_x, start_y, end_x, end_y))

            # Calculate the histogram of the tile
            histogram = [0] * 256
            pixels = np.array(tile)
            for i in range(tile_size):
                for j in range(tile_size):
                    pixel = pixels[i, j]
                    histogram[pixel] += 1

            # Calculate the cumulative distribution function (CDF)
            cdf = [0] * 256
            cdf[0] = histogram[0]
            for i in range(1, 256):
                cdf[i] = cdf[i-1] + histogram[i]

            # Calculate the equalized tile
            equalized_tile = Image.new('L', (tile_size, tile_size))
            equalized_pixels = []
            for i in range(tile_size):
                for j in range(tile_size):
                    pixel = pixels[i, j]
                    equalized_pixel = int(cdf[pixel] * 255 / (tile_size ** 2))
                    equalized_pixels.append(equalized_pixel)
            equalized_tile.putdata(equalized_pixels)

            # Paste the equalized tile onto the enhanced image
            enhanced_image.paste(equalized_tile, (start_x, start_y))

    return enhanced_image


# Open the image
image = Image.open('pic.jpg')

# Set the tile size for Adaptive Histogram Equalization
tile_size = 8

# Apply Adaptive Histogram Equalization
enhanced_image = adaptive_histogram_equalization(image, tile_size)

# Display the original image and the enhanced image
image.show()
enhanced_image.show()
