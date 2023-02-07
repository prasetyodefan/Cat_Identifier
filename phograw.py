import cv2
import numpy as np
from scipy.ndimage import convolve

def phog(img, bin_size=16, levels=3):

    # Step 1: Pre-processing
    #---------------------------------------------------------------------------

    # Convert RGB to G by using the dot product of the input 
    # image with a weighting array [0.2989, 0.5870, 0.1140].
    # array represents the scaling factors for the RGB channels of the image
    def grayscale(img):
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    # Perform a 2D convolution of an image with a kernel.
    # Parameter Usage | image
    #                 | kernel convolution kernel represent 2D np.array 
    #                 | mode 'same' output will have the same size as the input, 
    #                   with the result padded with zeros if necessary
    def convolve2d(img, kernel, mode='same'):
        m, n = img.shape
        k, l = kernel.shape
        if mode == 'same':
            pad_size = (k - 1) // 2
            pad = np.zeros((m + 2 * pad_size, n + 2 * pad_size))
            pad[pad_size:-pad_size, pad_size:-pad_size] = img
            result = np.zeros_like(img)
        else:
            pad = img
            result = np.zeros((m - k + 1, n - l + 1))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = (pad[i:i+k, j:j+l] * kernel).sum()
        return result
    
    # Convert img to grayscale with float 32 bit DataType
    gray = grayscale(img).astype(np.float32)


    # Step 2: Gradient computation
    #---------------------------------------------------------------------------
    # The gradient magnitude and orientation are computed using the Sobel operator

    # highlight areas of the image with sharp intensity changes (edges).
    def sobelx(img):
        sobel_x_kernel = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
        return convolve2d(gray, sobel_x_kernel, mode='same')

    def sobely(img):
        sobel_y_kernel = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])
        return convolve2d(gray, sobel_y_kernel, mode='same')


    sobel_x = sobelx(img)
    sobel_y = sobely(img)

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    gradient_orientation = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    # Step 3: Binning
    #---------------------------------------------------------------------------

    # The gradient orientation is divided into bin_size bins using integer division
    binned_orientation = (gradient_orientation /
                          bin_size).astype(np.int32) % bin_size

    # Step 4: Pyramidal representation
    #---------------------------------------------------------------------------
    # downsampling the image using the cv2.pyrDown function and computing the histograms of 
    # oriented gradients for each level. The histograms are stored in a list pyramid.

    pyramid = []
    def pyr_down(img, bin_size=16):
        # Define the downsampling kernel

        # The values in the 5x5 array are chosen based on the Gaussian function, which is a symmetric bell-
        # shaped curve that has a peak at the center and falls off symmetrically in both directions. 
        kernel = np.array([[1, 4, 6, 4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1]])
        
        # Normalize the kernel based on the factor
        kernel = 1.0/bin_size * kernel
        
        # Convolve the image with the kernel

        #  mode = 'constant' means that the values of the image at the edges 
        #  are assumed to be a constant value, which is typically set to 0.
        convolved = convolve(img, kernel, mode='constant')
        
        # Downsample the image by taking every other row and column
        downsampled = convolved[::2, ::2]
        
        return downsampled


    for i in range(levels):
        histograms = np.zeros((bin_size,))
        for y in range(gray.shape[0]):
            for x in range(gray.shape[1]):
                histograms[binned_orientation[y, x]] += gradient_magnitude[y, x]
        pyramid.append(histograms)
        gray = pyr_down(gray)

    # Step 5: Normalization
    #---------------------------------------------------------------------------

    normalized_pyramid = []
    for histograms in pyramid:
        normalization_factor = np.sum(histograms**2)**0.5
        if normalization_factor > 1e-12:
            histograms /= normalization_factor
        normalized_pyramid.append(histograms)

    # Step 6: Concatenation
    #---------------------------------------------------------------------------

    phog_descriptor = np.concatenate(normalized_pyramid)

    # Step 7: Representation (linear vector)
    #---------------------------------------------------------------------------

    return phog_descriptor


img = cv2.imread('Otsus.jpg')
result = phog(img)

print('Res PHOG', result)

# arr1 = np.array(
#     [0.42643104, 0.1535161,  0.1946256, 0.08233067, 0.07843661, 0.37725342,
#      0.1588102, 0.16553102, 0.43746586, 0.17057883, 0.16378863, 0.48821009,
#      0.08053733, 0.08347219, 0.17475933, 0.13854332, 0.3713421, 0.13950375,
#      0.13627793, 0.04548031, 0.06300616, 0.37603769, 0.16349112, 0.13134395,
#      0.4777487, 0.15067947, 0.17219277, 0.53367939, 0.08137838, 0.07088765,
#      0.17186279, 0.16140766, 0.36530032, 0.06870931, 0.07742774, 0.03312399,
#      0.05331249, 0.39420692, 0.13523713, 0.10587605, 0.47488013, 0.09909915,
#      0.12837234, 0.57637263, 0.10752364, 0.11185599, 0.18315421, 0.15927055])

# arr2 = np.array(
#     [0.26214832, 0.15421595, 0.23543935, 0.09984472, 0.10085099, 0.44098715,
#      0.17027293, 0.16937414, 0.41133146, 0.1529832, 0.15802398, 0.5371558,
#      0.0928122, 0.09539572, 0.19187187, 0.13834511, 0.36471999, 0.09291538,
#      0.12034196, 0.06480162, 0.06771878, 0.41574574, 0.18117029, 0.18823172,
#      0.46154389, 0.1840628, 0.17553509, 0.44507083, 0.11598644, 0.13506505,
#      0.23417323, 0.17789156, 0.40667392, 0.1048819, 0.14892111, 0.11462809,
#      0.11274134, 0.56873694, 0.21992118, 0.17797193, 0.32866541, 0.18658167,
#      0.20115071, 0.36614555, 0.09447911, 0.08864264, 0.15990961, 0.11124125])

# diff = arr1 - arr2
# distance = np.linalg.norm(diff)

# print('Diff', diff)
# print('Euclidean Dist', distance)
