## Preprocessing step 1 - filtered images and xml files
import os
import time
start_pascal = time.time()

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

## Preprocessing step 2 - cropped images by bounding box using xml files 
import xmltodict
from matplotlib import pyplot as plt
from skimage.io import imread

path_annotations = "assets/Training/CPYY/"
path_images = "assets/Training/CPYY/"

class_names = ['bengal','persian','siamese','rblue','ragdoll']

images = []
target = []
gmb = []
cim = []

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
      cim.append(crop_bounding_box(img, temp[i]["bndbox"]))
      gmb.append(img)
      target.append(temp[i]["name"])
  else:
    if temp["name"] not in class_names:
        continue
    images.append(crop_bounding_box(img, temp["bndbox"]))
    cim.append(crop_bounding_box(img, temp["bndbox"]))
    gmb.append(img)
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
import skimage
from skimage.transform import resize

from scipy import ndimage
from skimage import io, color, exposure, filters
from skimage.filters import unsharp_mask

def resize_image(img, size = 32):
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
  # images[i] = remove_background(images[i])

## Extration step 1 - extract features using PHOG (Pyramid Histogram of Oriented Gradients)
from PIL import Image
from scipy.ndimage import convolve

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

import pickle

pkl_filename = 'svm_model.pkl'
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

from sklearn.model_selection import train_test_split

X, y = features_2d, np.array(target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Menentukan rentang sumbu x dan y
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Membuat meshgrid dengan rentang sumbu x dan y yang sesuai
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                       np.linspace(x2_min, x2_max, 100))

# Memprediksi kelas untuk setiap titik pada meshgrid
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)


import random
rnd = random.randint(0, len(features))

pdt = features[rnd]

prediction = loaded_model.predict([pdt])
print('Len of Features : ', len(features))
print("Data Test  :", target[rnd])
print("Prediction :", prediction)

import matplotlib.pyplot as plt

# Membuat plot data training dan decision boundaries
plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

# Menandai support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=80, facecolors='none', edgecolors='k')

# Mengatur batasan sumbu x dan y
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

# Menampilkan plot
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundary SVM')

plt.show()