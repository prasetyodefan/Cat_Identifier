import os
import matplotlib.pyplot as plt
import cv2

path = "..\\Cat_Identifier\\asset\\dataset\\bengal55"
resized_path = "..\\Cat_Identifier\\asset\\dataset\\bengal55"

size = (500, 500)

for image_file in os.listdir(path):
    if image_file.endswith(".jpg"):
        im = plt.imread(os.path.join(path, image_file))
        im = cv2.resize(im, size)
        plt.imsave(os.path.join(resized_path, image_file), im)

