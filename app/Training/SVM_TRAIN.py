# ----------------------------------------------------------------
#?                       Read Pascal VOC                         |
# ----------------------------------------------------------------

import os
import time

s_psc = time.time()

img_names = []
xml_names = []

for dirname, subdirs, filenames in os.walk('assets/Training/CPYY/'):
  for filename in filenames:
    if filename[-3:] != "xml":
      img_names.append(filename)
    else:
      xml_names.append(filename)

print("Total Files")
print("Images    :",len(img_names))
print("Xml Files :",len(xml_names))
print()

import xmltodict
from matplotlib import pyplot as plt
from skimage.io import imread

path_annotations = "assets/Training/CPYY//"
path_images = "assets/Training/CPYY/"
class_names = ['bengal',
               'siamese',
               'rblue',
               'persian',
               'ragdoll']
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
      print("Cropping", i+1, "of", len(temp))
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

e_psc = time.time()

# ----------------------------------------------------------------
#?                         PREPROCESS                            |
# ----------------------------------------------------------------
s_prep = time.time()

import numpy as np
from skimage.transform import resize

from scipy import ndimage
from skimage import exposure
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

  clahe_image = exposure.equalize_adapthist(_img , clip_limit=3)

  blurred = ndimage.gaussian_filter(clahe_image, sigma=1)
  unsharp_maskk = clahe_image - 0.5 * blurred

  threshold_value = unsharp_mask(unsharp_maskk, radius=5, amount=2)
  binary_image = unsharp_maskk > threshold_value

  # Convert the binary image to uint8 and scale it to 0-255
  binary_image = binary_image.astype(np.float32)
  return _img  


for i in range(len(images)):
  print("Processing prepo", i+1, "of", len(images))
  images[i] = resize_image(images[i])
  images[i] = grayscale(images[i])
  images[i] = prepo(images[i])

e_prep = time.time()

# ----------------------------------------------------------------
#?                      EKSTRAKSI FITUR                          |
# ----------------------------------------------------------------

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

s_ext = time.time()

features = []
for i in range(len(images)):
    print("Processing", i+1, "of", len(images))
    lbp = calculate_lbp(images[i])  # Menghitung LBP untuk gambar
    
    # Mengubah dimensi LBP menjadi 1D
    lbp_1d = lbp.reshape(-1)
    
    features.append(lbp_1d)

# Mengubah dimensi fitur menjadi 2D
features_2d = np.array(features)

e_ext = time.time()

# --------------------------------------------------------------------

s_split = time.time()
from sklearn.model_selection import train_test_split

X, y = features_2d, np.array(target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print()
print("Training Data :\n", np.asarray(np.unique(y_train, return_counts=True)).T)
print("Test Data     :\n", np.asarray(np.unique(y_test, return_counts=True)).T)

e_split = time.time()

# ----------------------------------------------------------------
#?                         KLASIFIKASI                           |
# ----------------------------------------------------------------

s_class = time.time()
from sklearn.svm import SVC

clf = SVC(C=10, 
          kernel='rbf',
          random_state=42, 
          max_iter=-1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

e_class = time.time()

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

t_psc = e_psc - s_psc
t_prep = e_prep - s_prep
t_split = e_split - s_split
t_ext = e_ext - s_ext
t_class = e_class - s_class

print()
print("Execution time Read Pascal VOC : {:.2f} Sec".format(t_psc))
print("Execution time Preprocces      : {:.2f} Sec".format(t_prep))
print("Execution time Feature ext     : {:.2f} Sec".format(t_ext))
print("Execution time Split Data      : {:.2f} Sec".format(t_split))
print("Execution time Classifier      : {:.2f} Sec".format(t_class))
print("Total Execution time           : {:.2f} Sec".format(t_split+t_class+t_split+t_ext+t_prep+t_psc))
print()
print('Accuracy                       : {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision                      : {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Recall                         : {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('F1                             : {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))
print()
print(classification_report( y_test, y_pred, target_names=class_names ))
cmd = ConfusionMatrixDisplay.from_estimator( clf, X_test, y_test, display_labels=class_names)

plt.tight_layout()
plt.show()






