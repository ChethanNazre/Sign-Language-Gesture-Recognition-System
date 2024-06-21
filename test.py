import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf 
from tensorflow.keras.models import load_model # type: ignore
import os

from tomlkit import key



# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize HandDetector and Classifier
detector = HandDetector(maxHands=1)
## model_path = "Users/prash/Desktop/Model/keras_model.h5"
model_path = "/Users/prash/Desktop/Model/keras_model.h5"
if os.path.exists(model_path):
    print(f"Found model file at {model_path}")
else:
    print(f"Model file not found at {model_path}")

labels_path = "/Users/prash/Desktop/Model/labels.txt"
classifier = Classifier("/Users/prash/Desktop/Model/keras_model.h5", "/Users/prash/Desktop/Model/labels.txt")

# Constants
offset = 2
imgSize = 300
labels = ["Hello", "Yes", "No", "Please", "Thank you", "Okay", "I love you"]

while True:
    # Capture frame-by-frame
    success, img = cap.read() 
    imgOutput = img.copy()    
    hands, img = detector.findHands(img)
    

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        imgCropShape = imgCrop.shape       
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction , index = classifier.getPrediction(imgWhite, draw= False)
            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction , index = classifier.getPrediction(imgWhite, draw= False)

        # Get prediction from the classifier
        #prediction, index = classifier.getPrediction(imgWhite, draw=False)
        print(prediction, index)

        # Draw rectangles and labels on the output image
        cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (0, 255, 0), 4)

        #imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        cv2.imshow('ImageCrop',imgCrop)
        cv2.imshow('ImageWhite',imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)

