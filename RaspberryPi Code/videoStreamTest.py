from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import imutils
from imutils.video import VideoStream
import time
import numpy as np
def detect_and_predict_mask(frame, faceNet, maskNet):
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    
    detections = faceNet.forward()
    
    faces = []
    locs =[]
    preds=[]
    
    for i in range(0, detections.shape[2]):
        
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    if len(faces) >0:
        faces = np.array(faces, dtype = "float32")
        pred = maskNet.predict(faces, batch_size=32)
        
    return (locs, preds)


vs = VideoStream(src=0).start()
time.sleep(2.0)

print("Loading face detector model")
protoPath = "./deploy.prototxt"
weightsPath = "./res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(protoPath, weightsPath)

print("Loading mask detector model")
maskNet = load_model("./mask_detector.model")


while True:
    
