import os
import pickle
import random
import tkinter as tk
import cv2
import torch
import numpy as np

from PIL import Image, ImageTk
from torchvision import models, transforms

# Define tkinter
win = tk.Tk()
win.geometry("640x480")

label = tk.Label(win)
label.grid(row=0, column=0)

# Define video capture
cap = cv2.VideoCapture(1)

# Define preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Define model
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
for param in resnet.parameters():
    param.requires_grad = False
resnet.train(False)
resnet.cuda()

pkl_filename = 'face_mask_detection.pkl'
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

counter = 0
filename = random.choice(os.listdir("face-mask-detection/images/"))


# define show frames
def show_frames():
    global counter
    global filename

    # Get the latest frame and convert into Image
    img = cap.read()[1]
    # img = cv2.imread('face-mask-detection/images/' + filename)

    img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = detect_mask(img)
    img = Image.fromarray(img)

    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    counter += 1
    if counter % 5 == 0:
        filename = random.choice(os.listdir("face-mask-detection/images/"))

    # Repeat after an interval to capture continuously
    label.after(20, show_frames)


def detect_mask(img):
    # Define model
    prototxt = 'deploy.prototxt'
    model = 'res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Set size
    h, w = img.shape[:2]

    # Detect face
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Draw rectangle around the faces
    for i in range(0, detections.shape[2]):
        # Threshold confidence
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        # Crop face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype('int')
        # Classify face
        result, probability = classify(img[y1:y2, x1:x2])
        color = (0, 255, 0) if result == 'with_mask' else (255, 0, 0)

        # Draw rectangle
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        img = cv2.putText(img, f"{result} ({round(probability * 100)}%)", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def classify(img):
    # Apply preprocess
    image_tensor = preprocess(img).unsqueeze(0).cuda()

    # Extract Feature
    features = resnet(image_tensor)
    features = torch.flatten(features, 1).cpu().numpy()

    # Predict
    result = pickle_model.predict_proba(features)
    result = result[0]

    return pickle_model.classes_[np.argmax(result)], np.max(result)


show_frames()
win.mainloop()