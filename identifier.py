## Preprocessing step 1 - filtered images and xml files
import os

img_names = []
xml_names = []

for dirname, subdirs, filenames in os.walk('asset/dataset/mix/'):
  for filename in filenames:
    if filename[-3:] != "xml":
      img_names.append(filename)
    else:
      xml_names.append(filename)

print(" Total files")
print(len(img_names), "images")
print(len(xml_names), "xml files")

## Preprocessing step 2 - cropped images by bounding box using xml files 
import xmltodict
from matplotlib import pyplot as plt
from skimage.io import imread

path_annotations = "asset/dataset/mix/"
path_images = "asset/dataset/mix/"

class_names = ['bengal','persian','rblue','siamese','ragdoll']
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

# print total target by class
print("# Total target by class")
for i in class_names:
  print(i, ":", target.count(i))


## Preprocessing step 3 - resize images to 258x258 and normalize (remove background and grayscale)
import numpy as np
import skimage
from skimage.transform import resize

def resize_image(img, size=128):
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

for i in range(len(images)):
  images[i] = resize_image(images[i])
  images[i] = grayscale(images[i])
  images[i] = remove_background(images[i])

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

features = []
for i in range(len(images)):
  print("Processing item", i+1, "of ",len(images),"...")
  features.append(phog(images[i]))

# print(features[0])

from sklearn.model_selection import train_test_split

X, y = features, np.array(target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training data\n", np.asarray(np.unique(y_train, return_counts=True)).T)
print("Test data\n", np.asarray(np.unique(y_test, return_counts=True)).T)

## Classification step 1


# # # ------------------------------------------------------
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression

# clf = StackingClassifier(
#     estimators=[('svm', SVC(random_state=42)),
#                 ('tree', DecisionTreeClassifier(random_state=42))],
#     final_estimator=LogisticRegression(random_state=42),
#     n_jobs=-1)

# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'svm__C': [1.6, 1.7, 1.8],
#     'svm__kernel': ['rbf'],
#     'tree__criterion': ['entropy'],
#     'tree__max_depth': [9, 10, 11],
#     'final_estimator__C': [1.3, 1.4, 1.5]
# }

# grid = GridSearchCV(
#     estimator=clf,
#     param_grid=param_grid,
#     scoring='accuracy',
#     n_jobs=-1)

# grid.fit(X_train, y_train)

# print('Best parameters: %s' % grid.best_params_)
# print('Accuracy: %.2f' % grid.best_score_)
# # # ------------------------------------------------------

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

final_clf = StackingClassifier(
    estimators=[('svm', SVC(C=1.8, kernel='rbf', random_state=42)),
                ('tree', DecisionTreeClassifier(criterion='entropy', max_depth=9, random_state=42))],
    final_estimator=LogisticRegression(C=1.5, random_state=42),
    n_jobs=-1)

final_clf.fit(X_train, y_train)
y_pred = final_clf.predict(X_test)

print('Accuracy score : ', accuracy_score(y_test, y_pred))
print('Precision score : ', precision_score(y_test, y_pred, average='weighted'))
print('Recall score : ', recall_score(y_test, y_pred, average='weighted'))
print('F1 score : ', f1_score(y_test, y_pred, average='weighted'))


# Save SVM Model

import pickle

pkl_filename = 'svm_model.pkl'
with open(pkl_filename, 'wb') as file:
  pickle.dump(final_clf, file)

# create confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=['bengal','persian','rblue','siamese','ragdoll'])

# print confusion matrix

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        final_clf,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()