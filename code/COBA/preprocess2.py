# proses 1 filter image or xml file
import os
import time

img_names = []
xml_names = []
for dirname, dirs, filenames in os.walk('../asset/dataset/Deff_catface_data.v2i.voc/train'):
  starttime = time.time()
  for filename in filenames:
    if filename[-3:] != "xml":
      img_names.append(filename)
    else:
      xml_names.append(filename)
  endtime = time.time()
  print("Time taken: ", endtime-starttime)

print(len(img_names), "images")
print(len(xml_names), "xml files")


# proses 2 crop image
import xmltodict
from matplotlib import pyplot as plt
from skimage.io import imread

path_annotations = "../asset/dataset/Deff_catface_data.v2i.voc/train/"
path_images = "../asset/dataset/Deff_catface_data.v2i.voc/train/"

class_names = ['cat-face']
images = []
target = []

def crop_bounding_box(img, bnd):
  x1, x2, y1, y2 = list(map(int, bnd.values()))
  _img = img.copy()
  _img = _img[y1:y2, x1:x2]
  _img = _img[:,:,:3]
  return _img

for img_name in img_names:
  with open(path_annotations+img_name[:-4]+".xml") as fd:
    doc = xmltodict.parse(fd.read())

  img = imread(path_images+img_name)
  temp = doc["annotation"]["object"]
  if type(temp) == list:
    for i in range(len(temp)):
      if temp[i]["name"] not in class_names:
        continue
      images.append(crop_bounding_box(img, temp[i]["bndbox"]))
      target.append(temp[i]["name"])
  else:
    if temp["name"] not in class_names:
        continue
    images.append(crop_bounding_box(img, temp["bndbox"]))
    target.append(temp["name"])


# proses 3 resize image
import torch

from torchvision import transforms

# Define preprocessing
preprocess = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize((128, 128)),
  transforms.ToTensor(),
  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Apply preprocess
image_tensor = torch.stack([preprocess(image) for image in images])
image_tensor[0].shape
image_numpy = [image.numpy().transpose(1, 2, 0) for image in image_tensor]
print("AA", image_numpy[0].shape)


# proses 4
from PIL import Image
import numpy as np
from scipy.ndimage import convolve


def phog(img, bin_size=16, levels=3):

    # Step 1: Pre-processing
    # ---------------------------------------------------------------------------

    # Convert RGB to G by using the dot product of the input
    # image with a weighting array [0.2989, 0.5870, 0.1140].
    # array represents the scaling factors for the RGB channels of the image
    def grayscale(img):
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    # Set Threshold Value
    threshold = 80

    def otsu_threshold(img):
        histogram = np.histogram(img, bins=np.arange(257), range=(0, 256))[0]
        cumulative_sum = np.cumsum(histogram)
        cumulative_mean = np.cumsum(
            histogram * np.arange(256)) / (cumulative_sum + 1e-10)
        between_class_variance = cumulative_sum[:-1] * (
            cumulative_sum[-1] - cumulative_sum[:-1]) * (cumulative_mean[:-1] - cumulative_mean[-1])**2

        return threshold

    def apply_threshold(img, threshold):
        return (img > threshold).astype(np.float32)

    img = grayscale(img)
    threshold = otsu_threshold(img)
    gray = apply_threshold(img, threshold)

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

    # Step 2: Gradient computation
    # ---------------------------------------------------------------------------
    # The gradient magnitude and orientation are computed using the Sobel operator

    # highlight areas of the image with sharp intensity changes (edges).
    def sobelx(img):
        sobel_x_kernel = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]])
        return convolve2d(gray, sobel_x_kernel, mode='same')

    def sobely(img):
        sobel_y_kernel = np.array([[-1, -2, -1],
                                   [0,  0,  0],
                                   [1,  2,  1]])
        return convolve2d(gray, sobel_y_kernel, mode='same')

    sobel_x = sobelx(img)
    sobel_y = sobely(img)

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    gradient_orientation = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    # Step 3: Binning
    # ---------------------------------------------------------------------------

    # The gradient orientation is divided into bin_size bins using integer division
    binned_orientation = (gradient_orientation /
                          bin_size).astype(np.int32) % bin_size

    # Step 4: Pyramidal representation
    # ---------------------------------------------------------------------------
    # downsampling the image using the pyrDown function and computing the histograms of
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
                histograms[binned_orientation[y, x]
                           ] += gradient_magnitude[y, x]
        pyramid.append(histograms)
        gray = pyr_down(gray)

    # Step 5: Normalization
    # ---------------------------------------------------------------------------

    normalized_pyramid = []
    for histograms in pyramid:
        normalization_factor = np.sum(histograms**2)**0.5
        if normalization_factor > 1e-12:
            histograms /= normalization_factor
        normalized_pyramid.append(histograms)

    # Step 6: Concatenation
    # ---------------------------------------------------------------------------

    phog_descriptor = np.concatenate(normalized_pyramid)

    # Step 7: Representation (linear vector)
    # ---------------------------------------------------------------------------

    return phog_descriptor

img = Image.open('/Users/ilhamygp/project/Cat_Identifier/asset/dataset/bengal/bengal (1).jpg')
img = np.array(img)
#img = arrim.shape
print('Shape', img.shape)

result = phog(img)
print('Res PHOG', result)