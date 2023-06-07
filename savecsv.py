## Preprocessing step 1 - filtered images and xml files
import os
import time
start_pascal = time.time()

img_names = []
xml_names = []

for dirname, subdirs, filenames in os.walk('asset/dataset/ragdoll/'):
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

path_annotations = "asset/dataset/ragdoll/"
path_images = "asset/dataset/ragdoll/"


class_names = ['bengal','persian','siamese','rblue','ragdoll']
images = []
target = []
gmb = []
cim= []


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
    loaded_model = pickle.load(file)

import random


import pandas as pd
leng = len(features)
# Create a DataFrame from your data
data = []
for p in range(leng):
    pdt = features_2d[p]
    prediction = loaded_model.predict([pdt])
    data_test = target[p]
    im_name = img_names[p]
    prediction_str = str(prediction).strip("['']")  # Remove square brackets and single quotes
    data.append([p, pdt, data_test, prediction_str])

df = pd.DataFrame(data, columns=['Index', 'Feature', 'Data Test', 'Prediction'])

# Export the DataFrame to an XLSX file
df.to_excel('data.xlsx', index=False)

print("Data exported to data.xlsx successfully.")


import random
rnd = random.randint(0, len(features))

pdt = features[rnd]

prediction = loaded_model.predict([pdt])
print('Len of Features : ', len(features))
print("Data Test  :", target[rnd])
print("Prediction :", prediction)

import matplotlib.pyplot as plt
# Create a figure and axes
fig, axes = plt.subplots(1, 4, figsize=(10, 5))

# Display the first image
axes[1].imshow(images[rnd])
axes[1].set_title('Cropped ')

# Display the second image
axes[2].imshow(cim[rnd])
axes[2].set_title('Cropped Original')

axes[3].imshow(gmb[rnd])
axes[3].set_title('Original')

plt.tight_layout()
plt.show()