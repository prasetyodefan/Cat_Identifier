
# ----------------------------------------------------------------
#?                         PREPROCESS                            |
# ----------------------------------------------------------------

import os
import time

start_pascal = time.time()

img_names = []
xml_names = []

for dirname, subdirs, filenames in os.walk('asset/dataset/CPYY/'):
  for filename in filenames:
    if filename[-3:] != "xml":
      img_names.append(filename)
    else:
      xml_names.append(filename)

print("Total Files")
print("Images    :",len(img_names))
print("Xml Files :",len(xml_names))
print()

## Preprocessing step 2 - cropped images by bounding box using xml files 
import xmltodict
from matplotlib import pyplot as plt
from skimage.io import imread

path_annotations = "asset/dataset/CPYY/"
path_images = "asset/dataset/CPYY/"
class_names = ['bengal','ragdoll','siamese','rblue','persian']
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
print("Total target by class")
for i in class_names:
  print(i , "   :", target.count(i))

# --------------------------------------------------------------------
end_pascal = time.time()
pascalex = end_pascal - start_pascal
print("Execution time Read Pascal : {} seconds".format(pascalex) ,"\n" )

# --------------------------------------------------------------------

start_prepo = time.time()
## Preprocessing step 3 - resize images to 258x258 and normalize (remove background and grayscale)
import numpy as np
from skimage.transform import resize

from scipy import ndimage
from skimage import io, color, exposure, filters
from skimage.filters import unsharp_mask

def resize_image(img, size = 32):
  _img = img.copy() 
  _img = resize(_img, (size, size))
  return _img

def grayscale(img):
  _img = img.copy()
  _img = np.dot(_img[...,:3], [0.299, 0.587, 0.114])
  return _img

def prepo(img):
  _img = img.copy()

  # Apply Adaptive Histogram Equalization (AHE) to the grayscale image
  clahe_image = exposure.equalize_adapthist(_img , clip_limit=3)

  # Apply Unsharp Masking to the AHE result
  blurred = ndimage.gaussian_filter(clahe_image, sigma=1)
  unsharp_maskk = clahe_image - 0.5 * blurred

  # Perform segmentation using Thresholding
  threshold_value = unsharp_mask(unsharp_maskk, radius=5, amount=2)
  binary_image = unsharp_maskk > threshold_value

  # Convert the binary image to uint8 and scale it to 0-255
  binary_image = binary_image.astype(np.float32)
  return _img  


for i in range(len(images)):
  images[i] = resize_image(images[i])
  images[i] = grayscale(images[i])
  images[i] = prepo(images[i])

# ----------------------------------------------------------------
#?                      EKSTRAKSI FITUR                          |
# ----------------------------------------------------------------
import cv2
def calculate_lbp(img, radius=3, neighbors=8):
    
    lbp = np.zeros_like(img)
    
    for i in range(radius, img.shape[0] - radius):
        for j in range(radius, img.shape[1] - radius):
            center = img[i, j]
            binary_code = 0
            
            for k in range(neighbors):
                x = i + int(radius * np.cos(2 * np.pi * k / neighbors))
                y = j - int(radius * np.sin(2 * np.pi * k / neighbors))
                
                if img[x, y] >= center:
                    binary_code |= (1 << (neighbors - 1 - k))
            
            lbp[i, j] = binary_code
    
    return lbp


#--------------------------------------------------------------------------------

features = []
for i in range(len(images)):
    print("Processing", i+1, "of", len(images))
    lbp = calculate_lbp(images[i])  # Menghitung LBP untuk gambar
    
    # Mengubah dimensi LBP menjadi 1D
    lbp_1d = lbp.reshape(-1)
    
    features.append(lbp_1d)

# Mengubah dimensi fitur menjadi 2D
features_2d = np.array(features)


# --------------------------------------------------------------------
end_prepo = time.time()
prepotime = end_prepo - start_prepo
print()
print("Execution time Prepp : {} seconds".format(prepotime))
# --------------------------------------------------------------------

start_split = time.time()
from sklearn.model_selection import train_test_split

X, y = features_2d, np.array(target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print()
print("Training Data :\n", np.asarray(np.unique(y_train, return_counts=True)).T)
print("Test Data     :\n", np.asarray(np.unique(y_test, return_counts=True)).T)

end_split = time.time()
splittime = end_split - start_split
print()
print("Execution Time Split Data : {} seconds".format(splittime))

# ----------------------------------------------------------------
#?                         KLASIFIKASI                           |
# ----------------------------------------------------------------

start_class = time.time()
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# clf = LinearSVC(multi_class='crammer_singer', dual=False)
clf = SVC(C=1, kernel='rbf',random_state=42, max_iter=-1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)



end_class = time.time()
classtime = end_class - start_class
print()
print("Execution time Classifier : {} seconds".format(classtime))


# ----------------------------------------------------------------
#?                         SAVE MODEL                            |
# ----------------------------------------------------------------

import pickle
pkl_filename = 'svm_model.pkl'
with open(pkl_filename, 'wb') as file:
  pickle.dump(clf, file)


# ----------------------------------------------------------------
#?                      EVALUASI MATRIX                          |
# ----------------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
# print("Predicting cat breed on the test set")
print()
print(classification_report( y_test, y_pred, target_names=class_names ))
cmd = ConfusionMatrixDisplay.from_estimator( clf, X_test, y_test, display_labels=class_names)

plt.tight_layout()
plt.show()

print()
print('Accuracy score   : ', accuracy_score(y_test, y_pred))
print('Precision score  : ', precision_score(y_test, y_pred, average='weighted'))
print('Recall score     : ', recall_score(y_test, y_pred, average='weighted'))
print('F1 score         : ', f1_score(y_test, y_pred, average='weighted'))






