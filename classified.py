import numpy as np
from skimage.io import imread
import skimage
from skimage.transform import resize
from scipy.ndimage import convolve
from PIL import Image
import pickle

# Load the trained model
pkl_filename = 'svm_model.pkl'
with open(pkl_filename, 'rb') as file:
    final_clf = pickle.load(file)

class_names = ['bengal', 'persian', 'ragdoll', 'rblue', 'siamese']

# Load and preprocess the single image
image_path = 'asset/dataset/mix/Siamese_29_jpg.rf.59026415dfafd96375c8fccc8d9c32a3.jpg'  # Replace with the path to your image
img = imread(image_path)

def crop_bounding_box(img, bnd):
    x1, x2, y1, y2 = list(map(int, bnd.values()))
    _img = img.copy()
    _img = _img[y1:y2, x1:x2]
    _img = _img[:,:,:3]
    return _img

# Preprocessing step 2 - cropped images by bounding box using xml files
bounding_box = {'xmin': 10, 'xmax': 100, 'ymin': 20, 'ymax': 200}  # Replace with the actual bounding box values
img_cropped = crop_bounding_box(img, bounding_box)

# Preprocessing step 3 - resize image to 258x258 and normalize (remove background and grayscale)
def remove_background(img):
    _img = img.copy()
    thresh = skimage.filters.threshold_otsu(_img)
    _img = (_img > thresh).astype(np.float32)
    return _img

def grayscale(img):
    _img = img.copy()
    _img = np.dot(_img[...,:3], [0.299, 0.587, 0.114])
    return _img

def resize_image(img, size=258):
    _img = img.copy() 
    _img = resize(_img, (size, size))
    return _img

img_resized = resize_image(img_cropped)
img_grayscale = grayscale(img_resized)
img_normalized = remove_background(img_grayscale)

# Feature Extraction - PHOG
def compute_hog(img, bin_size=16):
    # Compute the gradient magnitude and orientation
    gx = convolve(img, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = convolve(img, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi

    # Binning
    binned_orientation = (orientation / bin_size).astype(np.int32) % bin_size

    return binned_orientation


# Extract features from the single image
features = compute_hog(img_normalized)

# Flatten the features array
features_flat = features.flatten()

# Reshape the flattened array into a 2D array with a single sample
features_2d = features_flat.reshape(1, -1)

# Classify the image using the trained model
predicted_class = final_clf.predict(features_2d)[0]
predicted_breed = class_names[predicted_class]
print("Predicted breed:", predicted_breed)
