---
layout: post
title: How to track Mario in Python
categories: [GUI tutorial series]
mathjax: true
featured-img: mario
summary: This code will track Mario game character using opencv and python
---


```python
# Welcome to PyShine
# First we require cv2 versionn 3.4.5.20
# Lets make sure we have the right version installed
# Lets uninstall the previous 
# UNINSTALL :
# pip uninstall opencv-contrib-python
# pip uninstall opencv-python
# INSTALL
# pip install opencv-contrib-python==3.4.5.20
# pip install imutils

# Lets start 
import cv2, imutils
import time
import numpy as np
tracker = cv2.TrackerCSRT_create()
camera=False # Set it to True for webcam, else its video
if camera: 
	video  = cv2.VideoCapture(0)
else:
	video = cv2.VideoCapture('videos/mario.mp4')
_,frame = video.read()
frame = imutils.resize(frame,width=720)
BB = cv2.selectROI(frame,False)
tracker.init(frame, BB)
while True:
	_,frame = video.read()
	frame = imutils.resize(frame,width=720)
	img_rgb=frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	track_success,BB = tracker.update(frame)
	if track_success:
		top_left = (int(BB[0]),int(BB[1]))
		bottom_right = (int(BB[0]+BB[2]), int(BB[1]+BB[3]))
		cv2.rectangle(img_rgb,top_left,bottom_right,(0,255,0),5)
		cv2.imshow('Output',img_rgb)
		key  =  cv2.waitKey(1) & 0xff
	if key == ord('q'):
		break
video.release()
cv2.destroyAllWindows()

```
[Download source code with video]
[Download source code with video]:https://drive.google.com/file/d/1VYaLU69NjXRg0TNVSfANVO4A6_Ij8GgI/view?usp=sharing
