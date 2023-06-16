## Preprocessing step 1 - filtered images and xml files
import os
import time
start_pascal = time.time()

img_names = []
xml_names = []

for dirname, subdirs, filenames in os.walk('assets/Predict/persian/'):
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

path_annotations = "assets/Predict/persian/"
path_images = "assets/Predict/persian/"
# 'bengal','','siamese','rblue','ragdoll'
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

from PIL import Image, ImageDraw

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
leng = len(features)
# Create a DataFrame from your data
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

# Membuat DataFrame
data = []
kosong = 0
for p in range(leng):
    pdt = features_2d[p]
    prediction = loaded_model.predict([pdt])
    data_test = target[p]
    im_name = img_names[p]
    im_data = gmb[p]
    prediction_str = str(prediction).strip("['']")
    data.append([p, im_name, data_test, prediction_str, kosong, im_data])

df = pd.DataFrame(data, columns=['Index', 'Name', 'Data Test', 'Prediction', 'True', 'Image'])

# Menyimpan DataFrame ke file Excel
output_filename = 'output.xlsx'
df.to_excel(output_filename, index=False)

# Membuka workbook yang sudah disimpan
workbook = load_workbook(output_filename)
sheet = workbook.active

# Mendapatkan huruf kolom untuk kolom ke-5 (True column)
column_letter = get_column_letter(5)

# Menambahkan rumus '=IF(C2=D2;1;0)' ke kolom True
for row in range(2, len(df) + 2):
    cell = f'{column_letter}{row}'
    sheet[cell].value = f'=IF(C{row}=D{row}, 1, 0)'

# Menambahkan rumus '=AVERAGE(E1:E51)' ke sel E5
cell = 'E52'
sheet[cell].value = '=AVERAGE(E1:E51)'
# Menyimpan workbook dengan rumus ke file Excel
workbook.save(output_filename)


print("Data exported to data.xlsx successfully.")


import random
rnd = random.randint(0, len(features))

pdt = features[rnd]
im_name = img_names[rnd]

predt = loaded_model.predict([pdt])
prediction = str(predt).strip("['']") 


print('Len of Features : ', len(features))
print('Name : ', im_name)
print("Data Test  :", target[rnd])
print("Prediction :", prediction)

import matplotlib.pyplot as plt
# Create a figure and axes
fig, axes = plt.subplots(1, 4, figsize=(10, 5))

axes[0].imshow(gmb[rnd])
axes[0].set_title('Original')

# Display the second image
axes[1].imshow(cim[rnd])
axes[1].set_title('Cropped Original')

# Display the first image
axes[2].imshow(images[rnd])
axes[2].set_title('Cropped ')

axes[3].set_title('Predict : ')
axes[3].text(0.3, 0.5, "Data Test  : " + str(target[rnd]))
axes[3].text(0.5, 0.3, "Prediction : " + str(prediction))

plt.tight_layout()
plt.show()