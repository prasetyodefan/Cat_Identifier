from PIL import Image
import numpy as np
import cv2

impil = Image.open('cropfaces.jpg')
arrim = np.array(impil)
img = arrim.shape

inn = cv2.imread('cropfaces.jpg')
i = inn.shape

print('PIL : ',arrim)
print('CV2 : ',inn)