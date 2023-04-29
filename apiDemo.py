from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
from numpy import asarray
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2

app = FastAPI()

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

origins = [
    "http://localhost:3000/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("model.h5")
face_classifier = cv2.CascadeClassifier(r'C:\Users\Madhav Dhokiya\Documents\6TH_SEM_BTECH\SDP\Project\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
	contents = await file.read()
	
	#image = Image.open(r"C:\Users\Madhav Dhokiya\Downloads\my-image.jpeg")
	image = cv2.imread(r"C:\Users\Madhav Dhokiya\Downloads\my-image.jpeg")
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray)

	for(x,y,w,h) in faces:
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
		roi_gray = gray[y:y+h,x:x+w]
		roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

		if np.sum([roi_gray]) !=0:
			roi = roi_gray.astype('float')/255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi,axis=0)

			predicition = model.predict(roi)[0]
			label = emotion_labels[predicition.argmax()]
	
	"""
	imgtoarr = img_to_array(image)
	resize_image = tf.image.resize(imgtoarr,(48,48))
	imgtoarr = imgtoarr.astype('float')/255.0
	imgtoarr = np.expand_dims(resize_image,axis=0)
	predicition = model.predict(imgtoarr)[0]
	label = emotion_labels[predicition.argmax()]

	imgtoarr = asarray(image)
	print(imgtoarr.shape)
	imgtoarr = imgtoarr.shape[48,48]
	predicition = model.predict(imgtoarr)
	label = emotion_labels[predicition.argmax()]
	"""
	#os.remove(r"C:\Users\Madhav Dhokiya\Downloads\my-image.jpeg")
	return {"Emotion": label}
	

