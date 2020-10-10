---
layout: post
title: Add text with transparent rectangle on an image
categories: [GUI tutorial series]
mathjax: true
featured-img: putBText
summary: How to add transparent box behind text in an open cv image
---

[![GIF](https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/posts/putBText.jpg?raw=true)](https://youtu.be/LStHozI2aDo "GIF")

Hello friends, today we will put some text with a background box behind it. It looks awesome and conveys to the point information on the image as well. It can be used in various outputs of computer vision applications e.g. object detection results, office or school auto-attendence system. You can use it for Python3.

## Requirements
cv2
pyshine
numpy

```
pip3 install opencv-contrib-python
pip3 install pyshine
pip3 install numpy
```
To detect face, we require a [cascasdeClassifier](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) file.
The input image lena.jpg is here.

[![GIF2](https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/posts/lena.jpg?raw=true)](https://youtu.be/LStHozI2aDo "GIF2")

## Run the code
To run the code make a new folder, save the main.py below, the lena.jpg and the 'haarcascade_frontalface_default.xml' files.
```
cd to your folder
python3 main.py
```

## main.py
```python3

# author:    PyShine
# website:   http://www.pyshine.com

# import the necessary packages
import pyshine as ps,cv2
import time
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('lena.jpg')

text  =  'ID: '+str(123)
image = ps.putBText(image,text,text_offset_x=20,text_offset_y=20,vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(255,255,1))
text = str(time.strftime("%H:%M %p"))
image = ps.putBText(image,text,text_offset_x=image.shape[1]-170,text_offset_y=20,vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(255,255,1))

text  =  '6843'
image = ps.putBText(image,text,text_offset_x=20,text_offset_y=272,vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(255,255,255))

text  =  "Name: Lena"
image = ps.putBText(image,text,text_offset_x=20,text_offset_y=325,vspace=20,hspace=10, font_scale=1.0,background_RGB=(20,210,4),text_RGB=(255,255,255))

text  =  'Status: '
image = ps.putBText(image,text,text_offset_x=image.shape[1]-130,text_offset_y=200,vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(255,255,255))
text  =  'On time'
image = ps.putBText(image,text,text_offset_x=image.shape[1]-130,text_offset_y=242,vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(255,255,255))


text  =  'Attendence: '
image = ps.putBText(image,text,text_offset_x=image.shape[1]-200,text_offset_y=294,vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(255,255,255))
text  =  '96.2%      '
image = ps.putBText(image,text,text_offset_x=image.shape[1]-200,text_offset_y=336,vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(255,255,255))


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
gray,
scaleFactor=1.15,
minNeighbors=7,
minSize=(80, 80),
flags=cv2.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x + w, y + h), (228,225,222), 2)

cv2.imshow('Output', image)
cv2.imwrite('out.jpg',image)
cv2.waitKey(0)

```
