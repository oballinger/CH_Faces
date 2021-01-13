import pandas as pd
import csv
import urllib.request
import requests
import time
from bs4 import BeautifulSoup, SoupStrainer
import cv2 
import numpy as np
from PIL import Image, ImageDraw
import re
import os
import json
from deepface import DeepFace
import pandas as pd
import face_recognition

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#output face tiles are named such that they contain the full name of the parent image/video, preceeded by a unique face number per image
#e.g., for an image named "boogaloo.jpg" with 3 unique faces we get 3 face tiles:
	#'face1_boogaloo.jpg'
	#'face2_boogaloo.jpg'
	#'face3_boogaloo.jpg'
#for videos, the protocol is the same but an additional prefix indicates the frame number of the video (e.g., "boogaloo.mp4"):
	#'face_1_frame3_boogaloo.jpg'


###### directory for input images and videos 
### specify input folder manually 
source_dir="Downloads/"

### or scrape online archive 
scrape=True
if scrape:
	dl_folder='Downloads/'
	base_url="https://capitol-hill-riots.s3.us-east-1.wasabisys.com/"
	url=base_url+'directory.html'
	response = requests.get(url)
	soup=BeautifulSoup(response.text, "html.parser")

	ids=[]
	extensions= '.jpg|.png' #".mp4|.jpg|.png"
	for c in soup.findAll('a', attrs={'href': re.compile(extensions)}):
		ids.append(c.get('href'))
	ids=list(dict.fromkeys(ids))
	index=0
	for i in ids:
		index+=1
		ext=i.split('/')[-1]
		print(ext)
		with urllib.request.urlopen(i) as url:
			with open(dl_folder+str(index)+ext, 'wb') as f:
				f.write(url.read())


#directory for extracted faces
output='Faces/'

extract_video=False
extract_images=True


def makelist(extension):
	templist=[]
	for subdir, dirs, files in os.walk(source_dir):
		for file in files:
			if extension in os.path.join(subdir, file):
				f=os.path.join(subdir, file)
				templist.append(f)
	return templist


#create list of image and video filepaths 
img_list=makelist('.jpg')
video_list=makelist('.mp4')
video_list=video_list[25:]

count=0
#Second stage face verification, cropping, and saving function
def cropface(image,img_name, face_locations):
	
	count=0
	for (top, right, bottom, left) in face_locations:
		count+=1
		try:
			#draw a rectangle around each face, crop with a 10px buffer, and save as a temporary file 
			draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
			p=10
			face = image.crop((left-p, top-p, right+p, bottom+p))
			face.save("temp.jpg")

			#retrieve temporary file, check for faces using different neural network (DeepFace MTCNN). Only proceed to saving if a face is detected.
			#https://github.com/serengil/deepface
			detected_face = DeepFace.detectFace("temp.jpg", detector_backend = 'mtcnn')

			#optional module for demographic analysis of images
			#obj = DeepFace.analyze(img_path = 'temp.jpg')
			#print(obj["age"]," years old ",obj["dominant_race"]," ", obj["gender"])

			face.save(output+"face{0}_{1}.jpg".format(count,img_name))
			print("face{0}_{1}.jpg".format(count,img_name))


		except:
			pass


# pipeline for analysis of videos 
if extract_video:
	for v in video_list:

		cap = cv2.VideoCapture(v)
		video_name=v.split('/')[1].split('.')[0]
		frame_counter = 0
		while cap.isOpened():

			try:
				ret, frame = cap.read()
				if ret:
					# grab every 10th frame 
					frame_counter += 10 
					cap.set(1, frame_counter)
					
					img_name='frame{0}_'.format(frame_counter)+video_name
					frame = frame[:, :, ::-1]

					# Find all the faces in the frame
					face_locations = face_recognition.face_locations(frame)
					# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
					pil_image = Image.fromarray(frame)
					# Create a Pillow ImageDraw Draw instance to draw with
					draw = ImageDraw.Draw(pil_image)
					#apply cropping and verification algorithm
					cropface(pil_image, img_name, face_locations)
				#kill once video is finished
				if frame_counter >=cap.get(cv2.CAP_PROP_FRAME_COUNT)-11:
					break
			except:
				pass


# pipeline for analysis of videos 
if extract_images:
	for i in img_list:
		img_name=i.split('/')[1].split('.')[0]

		img = face_recognition.load_image_file(i)

		# First stage facial recognition using the face_recognition package (github.com/ageitgey/face_recognition)
		face_locations = face_recognition.face_locations(img)
		pil_image = Image.fromarray(img)
		draw = ImageDraw.Draw(pil_image)

		#apply cropping and verification algorithm
		cropface(pil_image, img_name, face_locations)
