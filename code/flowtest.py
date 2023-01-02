import cv2
import numpy as np
from matplotlib import pyplot as plt


# PATH LOCATE
ccpath = '..\\Cat_Identifier\\asset\\haardata\\haarcascade_frontalcatface.xml'
ccepath = '..\\Cat_Identifier\\asset\\haardata\\haarcascade_frontalcatface_extended.xml'
pathimg = '..\\Cat_Identifier\\asset\\dataset\\bengal55\\bengal (113).jpg'

# CAT CASCADE DETECTION
cat_cascade = cv2.CascadeClassifier(ccpath)
cat_ext_cascade = cv2.CascadeClassifier(ccepath)


img = cv2.imread(pathimg, 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SF 1.01 || N = 3 - 6
faces = cat_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
faces_ext = cat_ext_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=3)

# HAAR CAT FACE DETECTION
for (x, y, w, h) in faces:
    # BLUE
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #cv2.putText(img, 'Ras Kucing "Y" Terdeteksi', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # CUT DETECTED FACE
    detf = img[y:y+h, x:x+w]
    cv2.imshow("face", detf)
    cv2.imwrite('cropfaces.jpg', detf)

for (x, y, w, h) in faces_ext:
    # GREEN
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # faces = img[y:y+h, x:x+w]
    # cv2.imshow("face", faces)
    # cv2.imwrite('faces.jpg', faces)


def canny_edge_detection(detf, sigma=0.33):

    # Convert the image to grayscale
    gray = cv2.cvtColor(detf, cv2.COLOR_BGR2GRAY)

    # Compute the median of the single channel pixel intensities
    v = np.median(gray)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(100, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lower, upper)

    # Return the edges image
    return edges


# OTSU --------------------------------------------------------------------------


pathim = '..\\Cat_Identifier\\cropfaces.jpg'

imgg = cv2.imread(pathim, 0)
ret1, th1 = cv2.threshold(imgg, 5, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imwrite('Otsuses1.jpg', th1)

pathb = '..\\Cat_Identifier\\Otsuses1.jpg'
imgb = cv2.imread(pathb, 1)
# Perform Canny edge detection
edges = canny_edge_detection(detf)


# Display the result edges image
cv2.imshow('Otsu', imgb)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
