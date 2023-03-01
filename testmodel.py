## Preprocessing step 1 - filtered images and xml files
import os

dir_data = 'asset/dataset/test/'
file_list = os.listdir(dir_data)

img_names = 'asset/dataset/test/bengal-6-_jpg.rf.966f506dea0904fef8c84045dae9c69f.jpg'
xml_names = []

for filename in file_list:
    if filename[-3:] == "xml":
        xml_names.append(filename)

print(" Total files")
print(len(img_names), "images")
print(len(xml_names), "xml files")

## Preprocessing step 2 - cropped images by bounding box using xml files 
import xmltodict
from matplotlib import pyplot as plt
from skimage.io import imread

path_annotations = "asset/dataset/test/"
path_images = "asset/dataset/test/"

class_names = ['bengal','rblue']

images = []
target = []

def crop_bounding_box(img, bnd):
  x1, x2, y1, y2 = list(map(int, bnd.values()))
  _img = img.copy()
  _img = _img[y1:y2, x1:x2]
  _img = _img[:,:,:3]
  return _img

with open(path_annotations+img_names[:-4]+".xml") as fd:
    doc = xmltodict.parse(fd.read())

img = imread(path_images+img_names)
temp = doc["annotation"]["object"]


images.append(crop_bounding_box(img, temp["bndbox"]))
target.append(temp["name"])

# print total target by class
print("# Total target by class")
for i in class_names:
  print(i, ":", target.count(i))


## Preprocessing step 3 - resize images to 258x258 and normalize (remove background and grayscale)
import numpy as np
import skimage
from skimage.transform import resize

def resize_image(img, size=248):
  _img = img.copy() 
  _img = resize(_img, (size, size))
  return _img

def remove_background(img):
  _img = img.copy()
  thresh = skimage.filters.threshold_otsu(_img)
  _img = (_img > thresh).astype(np.float32)
  return _img

def grayscale(img):
  _img = img.copy()
  _img = np.dot(_img[...,:3], [0.299, 0.587, 0.114])
  return _img

images = resize_image(images)
images = grayscale(images)
images = remove_background(images)

## Extration step 1 - extract features using PHOG (Pyramid Histogram of Oriented Gradients)
from PIL import Image
from scipy.ndimage import convolve

def phog(img, bin_size=16, levels=3):
    # Compute the gradient magnitude and orientation
    gx = convolve(img, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = convolve(img, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi

    # Binning
    # ---------------------------------------------------------------------------
    # The gradient orientation is divided into bin_size bins using integer division
    binned_orientation = (orientation / bin_size).astype(np.int32) % bin_size

    # Pyramidal representation
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
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                histograms[binned_orientation[y, x]] += magnitude[y, x]
        pyramid.append(histograms)
        img = pyr_down(img)

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

features = phog(images)


# Load SVM Model
import pickle

# Load the trained SVM model from the pickle file
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

X_test_processed = features

# Predict the object classes using the SVM model
y_pred = svm_model.predict(X_test_processed)

accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# create confusion matrix

# Calculate the confusion matrix
cm = confusion_matrix(X_test_processed, y_pred)

# Display the confusion matrix
labels = ['Bengal','rblue'] # Replace with your own class labels
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
display.plot()