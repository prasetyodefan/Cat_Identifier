import cv2
import numpy as np
haarpath = "CatIdentifier\\asset\\haardata\\haarcascade_frontalcatface.xml"
haarpath_ext = "CatIdentifier\\asset\\haardata\\haarcascade_frontalcatface_extended.xml"
cat_cascade = cv2.CascadeClassifier(haarpath)
cat_ext_cascade = cv2.CascadeClassifier(haarpath_ext)

imgpath = "CatIdentifier\\asset\\dataset\\bengal\\bengal (30).jpg"
img = cv2.imread(imgpath)
img = img.astype("uint8")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SF 1.01 || N = 3 - 6
faces = cat_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6)
faces_ext = cat_ext_cascade.detectMultiScale(
    gray, scaleFactor=1.01, minNeighbors=6)

# print(type(faces))
# print(faces)
# HAAR CAT FACE DETECTION
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #cv2.putText(img, 'Ras Kucing "Y" Terdeteksi', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    faces = img[y:y+h, x:x+w]
    cv2.imshow("face", faces)
    #cv2.imwrite('faces.jpg', faces)

for (x, y, w, h) in faces_ext:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # faces = img[y:y+h, x:x+w]
    # cv2.imshow("face", faces)
    # cv2.imwrite('faces.jpg', faces)
detpath = "CatIdentifier\\code\\faces.jpg"

det = cv2.imread(detpath)

# APPLY OTSU THRESHOLDING


#cv2.imwrite('detected_faces.jpg', img)
cv2.imshow("Cat face", img)
cv2.waitKey(0)
