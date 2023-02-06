import cv2
import numpy as np


def phog(img, bin_size=16, levels=3):
    # Step 1: Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Gradient computation

    # Use cv2
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # def sobel_x(gray):
    #     kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    #     return np.abs(np.sum(gray * kernel, axis=(0, 1)))

    # def sobel_y(gray):
    #     kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    #     return np.abs(np.sum(gray * kernel, axis=(1, 0)))

    # sobelx = sobel_x(gray)
    # sobely = sobel_y(gray)

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_orientation = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    # Step 3: Binning
    binned_orientation = (gradient_orientation /
                          bin_size).astype(np.int32) % bin_size

    # Step 4: Pyramidal representation
    pyramid = []
    for i in range(levels):
        histograms = np.zeros((bin_size,))
        for y in range(gray.shape[0]):
            for x in range(gray.shape[1]):
                histograms[binned_orientation[y, x]
                           ] += gradient_magnitude[y, x]
        pyramid.append(histograms)
        gray = cv2.pyrDown(gray)

    # Step 5: Normalization
    normalized_pyramid = []
    for histograms in pyramid:
        normalization_factor = np.sum(histograms**2)**0.5
        if normalization_factor > 1e-12:
            histograms /= normalization_factor
        normalized_pyramid.append(histograms)

    # Step 6: Concatenation
    phog_descriptor = np.concatenate(normalized_pyramid)

    # Step 7: Representation (linear vector)
    return phog_descriptor


img = cv2.imread('Otsuses1.jpg')
result = phog(img)

print('res', result)

arr1 = np.array(
    [0.42643104, 0.1535161,  0.1946256, 0.08233067, 0.07843661, 0.37725342,
     0.1588102, 0.16553102, 0.43746586, 0.17057883, 0.16378863, 0.48821009,
     0.08053733, 0.08347219, 0.17475933, 0.13854332, 0.3713421, 0.13950375,
     0.13627793, 0.04548031, 0.06300616, 0.37603769, 0.16349112, 0.13134395,
     0.4777487, 0.15067947, 0.17219277, 0.53367939, 0.08137838, 0.07088765,
     0.17186279, 0.16140766, 0.36530032, 0.06870931, 0.07742774, 0.03312399,
     0.05331249, 0.39420692, 0.13523713, 0.10587605, 0.47488013, 0.09909915,
     0.12837234, 0.57637263, 0.10752364, 0.11185599, 0.18315421, 0.15927055])

arr2 = np.array(
    [0.26214832, 0.15421595, 0.23543935, 0.09984472, 0.10085099, 0.44098715,
     0.17027293, 0.16937414, 0.41133146, 0.1529832, 0.15802398, 0.5371558,
     0.0928122, 0.09539572, 0.19187187, 0.13834511, 0.36471999, 0.09291538,
     0.12034196, 0.06480162, 0.06771878, 0.41574574, 0.18117029, 0.18823172,
     0.46154389, 0.1840628, 0.17553509, 0.44507083, 0.11598644, 0.13506505,
     0.23417323, 0.17789156, 0.40667392, 0.1048819, 0.14892111, 0.11462809,
     0.11274134, 0.56873694, 0.21992118, 0.17797193, 0.32866541, 0.18658167,
     0.20115071, 0.36614555, 0.09447911, 0.08864264, 0.15990961, 0.11124125])

diff = arr1 - arr2
distance = np.linalg.norm(diff)

print('Diff', diff)
print('Euclidean Dist', distance)
