## Preprocessing step 1 - filtered images and xml files
import os
import cv2
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
import cv2
import numpy as np
from skimage import filters, feature
from skimage import exposure
from matplotlib import pyplot as plt
from skimage.filters import sobel

path_annotations = "asset/dataset/mix/"
path_images = "asset/dataset/mix/"

class_names = ['bengal','persian','siamese','ragdoll','rblue']
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
from skimage.transform import resize
from scipy import ndimage
from skimage import io, color, exposure, filters

def resize_image(img, size=258):
  _img = img.copy() 
  _img = resize(_img, (size, size))
  return _img

def prepo(img):
  _img = img.copy()

  # Apply Adaptive Histogram Equalization (AHE) to the grayscale image
  clahe_image = exposure.equalize_adapthist(_img , clip_limit=0.02)

  # Apply Unsharp Masking to the AHE result
  blurred = ndimage.gaussian_filter(clahe_image, sigma=1)
  unsharp_mask = clahe_image - 0.7 * blurred

  # Perform segmentation using Thresholding
  threshold_value = filters.threshold_otsu(unsharp_mask)
  binary_image = unsharp_mask > threshold_value

  # Convert the binary image to uint8 and scale it to 0-255
  binary_image = np.uint8(binary_image) * 255
  return _img  

def grayscale(img):
  _img = img.copy()
  _img = np.dot(_img[...,:3], [0.299, 0.587, 0.114])
  return _img

for i in range(len(images)):
  images[i] = resize_image(images[i])
  images[i] = grayscale(images[i])
  images[i] = prepo(images[i])

## Extration step 1 - extract features using PHOG (Pyramid Histogram of Oriented Gradients)
from PIL import Image
from scipy.ndimage import convolve
import numpy as np
from scipy.ndimage import sobel

def calculate_hog(img, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    def calculate_gradient(img):
        sobel_x = sobel(img, axis=1)
        sobel_y = sobel(img, axis=0)
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        ang = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)
        return mag, ang

    def calculate_histogram(ang, mag, nbins):
        hist = np.zeros(nbins)
        bin_width = 180.0 / nbins
        flattened_ang = ang.flatten()
        flattened_mag = mag.flatten()
        for i in range(len(flattened_ang)):
            bin_index = int(flattened_ang[i] / bin_width)
            if bin_index == nbins:  # Handle the case when the angle is exactly 180 degrees
                bin_index = 0
            hist[bin_index] += flattened_mag[i]
        return hist

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
    

features = []
for i in range(len(images)):
  print("Processing item", i+1, "of ",len(images),"...")
  features.append(calculate_hog(images[i], cell_size=(8,8), block_size=(2,2), nbins=9))

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

cm = confusion_matrix(y_test, y_pred, labels=['bengal', 'persian', 'ragdoll', 'rblue', 'siamese'])

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