from picamera import PiCamera
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import imutils
from imutils.video import VideoStream
import time
import numpy as np
import pygame
import os
pygame.mixer.init()

print("Loading audio file")
pygame.mixer.music.load("voice_message.mp3")
print("Loaded audio file")
print("Loading face detector model")
protoPath = "./deploy.prototxt"
weightsPath = "./res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(protoPath, weightsPath)
print("Loaded face detector model")
print("Loading mask detector model")
model = load_model("./mask_detector.model")
print("Loaded mask detector model")
camera = PiCamera()

wearingMask = True

def changeMaskVar(x):
	global wearingMask 
	wearingMask = x
	if x == False:
		pygame.mixer.music.play()
		while pygame.mixer.music.get_busy():
			pygame.time.Clock().tick(10)	
	#print("wearingMask has been changed to:" + str(wearingMask))
	
	
def checkMaskVar():
	global wearingMask
	print()
	#print("wearingMask has been changed to:" + str(wearingMask))
#	pygame.mixer.music.play()
#	while pygame.mixer.music.get_busy():
#		pygame.time.Clock().tick(10)


while True:
	
	#camera.start_preview()
	time.sleep(0.1)
	camera.capture('/home/pi/maskhack/image.jpg')
	#print("Image captured")
	#camera.stop_preview()
	image = cv2.imread("./image.jpg")
	orig = image.copy()
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
	#print("[INFO] computing face detections...")
	
	net.setInput(blob)
	detections = net.forward()
	
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		changeMaskVar(True)
		
		if confidence > 0.5:
			print("We have a face")
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			
			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
			print("Done with computations")
			
			(mask, withoutMask) = model.predict(face)[0]
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			if mask > withoutMask:
				changeMaskVar(True)
				print ("Mask")
			else:
				print("without Mask")
				changeMaskVar(False)
				os.system("curl 172.105.110.164/update")
				print("Waiting")
				#time.sleep(10)
			cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
			
	
	#changeMaskVar()
	
	image = imutils.resize(image, width=400)
	cv2.imshow("Output", image)
	
	
	
	key = cv2.waitKey(20) & 0xFF
	
	if key == ord("q"):
		break
		
cv2.destroyAllWindows()
			
	
	
