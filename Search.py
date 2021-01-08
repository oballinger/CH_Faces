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


output='Faces/'
#query
df = DeepFace.find(img_path = "mark1.jpg", db_path = output)

#get best match 
print(df)
matched_img=df.iat[0,0].split('_')[1]
image = cv2.imread(source_dir+matched_img)
cv2.imshow("cropped", image)
cv2.waitKey(0)
