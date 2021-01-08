import pandas as pd
import csv
from pathlib import Path
import urllib.request
import requests
import time
from bs4 import BeautifulSoup, SoupStrainer
import cv2 
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import math
import statistics
import re
import os
from deepface import DeepFace
import pandas as pd
import face_recognition
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

source_dir="Twitter/"
img_output='detected_faces/from_images/'
video_output='detected_faces/from_videos/'
output='Faces/'

img_list=[]
for subdir, dirs, files in os.walk(source_dir):
	for file in files:
		if ".jpg" in os.path.join(subdir, file):
			f=os.path.join(subdir, file)
			img_list.append(f)

video_list=[]
for subdir, dirs, files in os.walk(source_dir):
	for file in files:
		if ".mp4" in os.path.join(subdir, file):
			f=os.path.join(subdir, file)
			video_list.append(f)

def cropface(image,img_name, face_locations):
	print("IN FUNCTION ",img_name)
	count=0
	for (top, right, bottom, left) in face_locations:
		count+=1
		try:
			draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
			p=10
			face = image.crop((left-p, top-p, right+p, bottom+p))
			face.save("temp.jpg".format(count))
			detected_face = DeepFace.detectFace("temp.jpg", detector_backend = 'mtcnn')
			#obj = DeepFace.analyze(img_path = 'temp.jpg')
			#print(obj["age"]," years old ",obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])
			face.save(output+"face{0}_{1}.jpg".format(count,img_name))
			print("face{0}_{1}.jpg".format(count,img_name))
		except:
			pass

for v in video_list:
	cap = cv2.VideoCapture(v)
	video_name=v.split('/')[1].split('.')[0]
	frame_counter = 0
	while cap.isOpened():
		try:
			ret, frame = cap.read()
			if ret:
				frame_counter += 10 # grab every 10th frame 
				cap.set(1, frame_counter)
				img_name='frame{0}_'.format(frame_counter)+video_name
				frame = frame[:, :, ::-1]
				# Find all the faces and face encodings in the unknown image
				face_locations = face_recognition.face_locations(frame)
				# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
				pil_image = Image.fromarray(frame)
				# Create a Pillow ImageDraw Draw instance to draw with
				draw = ImageDraw.Draw(pil_image)
				pil_image.save("video_test.jpg")
				cropface(pil_image, img_name, face_locations)
			if frame_counter >=cap.get(cv2.CAP_PROP_FRAME_COUNT):
				break
				cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
		except:
			pass

quit()

for i in img_list:
	img_name=i.split('/')[1].split('.')[0]

	img = face_recognition.load_image_file(i)
	# Find all the faces and face encodings in the unknown image
	face_locations = face_recognition.face_locations(img)

	# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
	pil_image = Image.fromarray(img)
	# Create a Pillow ImageDraw Draw instance to draw with
	draw = ImageDraw.Draw(pil_image)
	cropface(pil_image, img_name, face_locations)

quit()
for i in img_list:
	img_name=i.split('/')[1].split('.')[0]
	image = cv2.imread(i)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	faces = faceCascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30),
	)

	count=0
	for (x, y, w, h) in faces:
		count=count+1
		p=10
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		crop_img = image[y-p:y+p+h, x-p:x+p+w]
		cv2.imwrite(img_output+'face{0}_'.format(count)+img_name+'.jpg', crop_img)
		print('face{0}_'.format(count)+img_name)
		#cv2.imshow("cropped", crop_img)
		#cv2.waitKey(0)



