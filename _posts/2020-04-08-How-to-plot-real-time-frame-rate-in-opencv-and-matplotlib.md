---
layout: post
title: How to plot realtime frame rate of a web camera 
categories: [GUI tutorial series]
mathjax: true
summary: Making of an OpenCV and Matplotlib data processing and visualization in Python
---


Hello there! Welcome to the PyShine Artificial Intelligence Learning series. Today we will learn, how to detect a Human face using Open CV library in Python, from a real-time web camera. We will also learn, how to plot the real-time frame rate of the camera.

Basically, the real-time video consists of the image frames that are shown multiple times in a second. The number of frames shown per second is known as frame rate. Artificial Intelligence techniques have their own processing time. The performance of and AI algorithm highly depends on its processing duration. Some Algorithms provide a very impressive short duration to process a video frame. For example, today we will use one such algorithm in the Open CV. The trained model is in the X M L file format. The name of the model is Haar cascade frontal face default.xml

The input to this model is an image frame matrix. It should be noted that this image is in Gray Scale so the dimension of the matrix is: Width times Height of the frame. The other input parameters play an important role in detection accuracy and processing speed as well. To simply find out the processing speed of a detection algorithm, we can find the frame rate of the output video of the processing algorithm.

Before going further towards implementation, let's discuss a little bit of theory behind the HAAR cascade classifier.

 
Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning-based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, Haar features shown in the image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting the sum of pixels, under the white rectangle from the sum of pixels under the black rectangle.

[![Everything Is AWESOME](https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/posts/face_youtube.png?raw=true)](https://youtu.be/tN5YbXmLGIE "Everything Is AWESOME")

The top row shows two good features. The first feature selected seems to focus on the property that the region of the eyes is often darker than the region of the nose and cheeks. The second feature selected relies on the property that the eyes are darker than the bridge of the nose. But the same windows applied to cheeks or any other place is irrelevant. 

OpenCV provides a training method or pre-trained models, that can be read using the Cascade Classifier load method. The pre-trained models are located in the data folder in the Open CV installation.

Let's take an example a simple face as you can see, it has two eyes as horizontal lines and a nose, and a mouse line. Let's try to find all the edge features of the shape as shown. The search window is of size 4 times 4. 

After this edge feature is applied to the face, it has detected the regions which have dark and while pixels region just like below the eyes, nose, and mouth.

Let's proceed with a different edge feature on the resulting image. And as you can see that after searching, the upper portion of eyes, and nose is also found.


Let's change the edge feature pixel values again and now we will find in the horizontal direction. The resulting image has now all the parts of eyes, nose and mouth are now detected. This was just a very simple example to illustrate the edge features. In real images, the pixels may take values from 0 to 1. And to find Delta from an edge feature of the eyebrow region, the values of all pixels are shown. 

On the left side, we have an ideal case of horizontal edge feature, all upper eight pixels are ones and all lower eight pixels are zeros. To find the Delta for this ideal case, we can subtract the average of lower region of black pixels from the upper region of white pixels. As we can see the delta is 1. 

On the right side we have a real image case, and to find out if the resulting delta from this image window is a horizontal edge feature, we can do the same subtraction process. If the value of delta is near to 1, we can say that this feature is a Haar edge feature like the one on the left. 

Similarly, other Haar features are applied to check its kind and finally we get all the Haar features. After we find all Delta values we get all Haar Features.


Let's start the coding to detect a face using open cv. 



 PyShine presents Matplotlib integration with Open CV to output the frame rate
 Lets write the face detection code
 This code has two parts: 1) Face.py 2) Plot.py
 Face.py will open Webcam and detect face in video stream
 Also it will generate a csv file which will have the current frame rate
 Plot.py will read the csv file and update the plot of frame rate
 The Face.py will run the Plot.py in a thread
 So lets start the code, by importing the required libraries


Put the .xml file in the current folder together with Face.py and Plot.py

Initiate the Webcam device with default id of 0

Generate the 'file.csv' with labels time, FPS: Frame Per Second


# All files are available here in a zip folder

[Download Code](https://drive.google.com/open?id=1yHu9lMS2sajj1uEmjg5-tTTDF4tt6upI)











```python 

# Subscribe to PyShine Youtube channel for the upcoming educational videos
# PyShine presents Matplotlib integration with Open CV to output the frame rate
# Lets write the face detection code
# This code has two parts: 1) Face.py 2) Plot.py
# Face.py will open Webcam and detect face in video stream
# Also it will generate a csv file which will have the current frame rate
# Plot.py will read the csv file and update the plot of frame rate
# The Face.py will run the Plot.py in a thread
# So lets start the code, by importing the required libraries
#################################### Face.py  Starts ################################################


import cv2
import time
import imutils
import _thread
import numpy as np
import os
# Put the .xml file in the current folder together with Face.py and Plot.py
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Initiate the Webcam device with default id of 0
video_capture = cv2.VideoCapture(0)
# Generate the 'file.csv' with labels time, FPS: Frame Per Second
print('time,FPS',  file=open('file.csv', 'w'))
# This function will run camera 
def camRun():
	# A counter to count the frames
	cnt=0
	# Number of frames to count after which the frame rate is obtained
	frames_to_count = 20
	# So we will count 20 frames and also note the time duration for these 20 frames 
	# And then simply divide frames by the duration in seconds to get frame rate
	st=0 # Start time st  = 0 seconds
	i=0 # This is a counter for the time samples for each FPS value
	while True:
		ret, frame = video_capture.read() # Get the frame 
		frame = imutils.resize(frame, width=320) # Resize it to 320 width , its optional 
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # RGB to Gray Matrix 
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.15,  
			minNeighbors=7, 
			minSize=(80, 80), 
			flags=cv2.CASCADE_SCALE_IMAGE
		)
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (214, 169, 33), 4) # Here we put rectangle on a frame
		cv2.imshow('FaceDetection', frame) # Display the frame in a window named FaceDetection
		k = cv2.waitKey(1) 
		if k == 27: # If click on video frame and press Esc, it will quit
			break
		# Frame rate calculation
		if cnt == frames_to_count:
			try: # To avoid divide by 0 we put it in try except
				print(frames_to_count/(time.time()-st),'FPS') 
				fps = frames_to_count/(time.time()-st) 
				print(str(i)+',' +str(fps),  file=open('file.csv','a')) 
				st = time.time()
				cnt=0
			except:
				pass
		# Counters are incremented here
		cnt+=1
		i+=1
# Lets call the Plot.py in a function plot
def plot():
	os.system('python Plot.py')
# Start the thread for the plot function
_thread.start_new_thread(plot,())
# Now run the camRun() function to generate the file.csv 
camRun()
# Relase the capture and windows
video_capture.release()
cv2.destroyAllWindows() 
# Please comment to provide feedback, if you have questions please ask, and
# Share and like , do subscribe to PyShine Youtube Channel.
#################################### Face.py  Ends ################################################
#################################### Plot.py  Starts ################################################

# Subscribe to PyShine Youtube channel for the upcoming educational videos
# Lets write the face detection code
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Get the Figure
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_facecolor((0,0,0)) # Set the background to black
# 
def animate(i):
	ax.clear()
	xs = []
	ys = []
	graph_data = open('file.csv','r').read() # Open file.csv generated by Face.py
	lines = graph_data.split('\n')
	for line in lines[1:]:
		if len(line) > 1: # Skip the first labels line in csv file
			x, y = line.split(',')
			xs.append(float(x))
			ys.append(float(y))
			print(xs,ys)
	# Lets add these lists xs, ys to the plot		
	ax.clear()
	ax.plot(xs, ys,'-o', color = (0,1,0.25))
	ax.set_xlabel("Samples")
	ax.set_ylabel("Frame Rate")
	ax.set_title("Live Plot of Camera Frame Rate")
	fig.tight_layout() # To remove outside borders
	ax.yaxis.grid(True)
# Lets call the animation function 	
ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()
#################################### Plot.py  Ends ################################################



```







	

