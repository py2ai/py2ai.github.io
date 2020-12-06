---
layout: post
title: How to make OpenCV and PyQt5 based GUI for image processing applications
categories: [GUI tutorial series]
mathjax: true
featured-img: pyqt5
summary: Making an OpenCv GUI with brightness and blur adjustment
---

Hello friends, The tutorial is part 05 of PyQt5 GUI learning series. This video contains instructions to make a basic photo editing application in Python. 
The concept of MVC or Model View Controller is implemented with explanation. Enjoy, and do 
give back your feedback and suggestions. For more detail visit pyshine youtube channel.

<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/DLJM7o0B8zk" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>

Some friends have asked about the possibility of saving the image as an output instead of overwriting the input image. Yes, it is of course possible. In the savePhoto function simply change the self.filename to filename. Then provide the desired name such as a time stamp or filename = 'output.jpg' or filename = 'output.png' etc. The code below is already updated. Thanks for your great suggestions!


![]({{ "assets/img/pyqt5opencv/one.jpg | absolute_url }})

![]({{ "assets/img/pyqt5opencv/two.jpg | absolute_url }})

![]({{ "assets/img/pyqt5opencv/three.jpg | absolute_url }})


```python

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'process.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!
#
# Subscribe to PyShine Youtube channel for more detail! 

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2, imutils

class Ui_MainWindow(object):
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(536, 571)

		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
		self.gridLayout_2.setObjectName("gridLayout_2")
		self.gridLayout = QtWidgets.QGridLayout()
		self.gridLayout.setObjectName("gridLayout")
		self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
		self.horizontalLayout_3.setObjectName("horizontalLayout_3")
		self.label = QtWidgets.QLabel(self.centralwidget)
		self.label.setText("")
		#self.label.setPixmap(QtGui.QPixmap("images/2.jpg"))
		self.label.setObjectName("label")
		self.horizontalLayout_3.addWidget(self.label)
		self.horizontalLayout = QtWidgets.QHBoxLayout()
		self.horizontalLayout.setObjectName("horizontalLayout")
		self.verticalSlider = QtWidgets.QSlider(self.centralwidget)
		self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
		self.verticalSlider.setObjectName("verticalSlider")
		self.horizontalLayout.addWidget(self.verticalSlider)
		self.verticalSlider_2 = QtWidgets.QSlider(self.centralwidget)
		self.verticalSlider_2.setOrientation(QtCore.Qt.Vertical)
		self.verticalSlider_2.setObjectName("verticalSlider_2")
		self.horizontalLayout.addWidget(self.verticalSlider_2)
		self.horizontalLayout_3.addLayout(self.horizontalLayout)
		self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 2)
		self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
		self.horizontalLayout_2.setObjectName("horizontalLayout_2")
		self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_2.setObjectName("pushButton_2")
		self.horizontalLayout_2.addWidget(self.pushButton_2)
		self.pushButton = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton.setObjectName("pushButton")
		self.horizontalLayout_2.addWidget(self.pushButton)
		self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
		spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
		self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)
		self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
		MainWindow.setCentralWidget(self.centralwidget)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setObjectName("statusbar")
		MainWindow.setStatusBar(self.statusbar)

		self.retranslateUi(MainWindow)
		self.verticalSlider.valueChanged['int'].connect(self.brightness_value)
		self.verticalSlider_2.valueChanged['int'].connect(self.blur_value)
		self.pushButton_2.clicked.connect(self.loadImage)
		self.pushButton.clicked.connect(self.savePhoto)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)
		
		# Added code here
		self.filename = None # Will hold the image address location
		self.tmp = None # Will hold the temporary image for display
		self.brightness_value_now = 0 # Updated brightness value
		self.blur_value_now = 0 # Updated blur value
	
	def loadImage(self):
		""" This function will load the user selected image
			and set it to label using the setPhoto function
		"""
		self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
		self.image = cv2.imread(self.filename)
		self.setPhoto(self.image)
	
	def setPhoto(self,image):
		""" This function will take image input and resize it 
			only for display purpose and convert it to QImage
			to set at the label.
		"""
		self.tmp = image
		image = imutils.resize(image,width=640)
		frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
		self.label.setPixmap(QtGui.QPixmap.fromImage(image))
	
	def brightness_value(self,value):
		""" This function will take value from the slider
			for the brightness from 0 to 99
		"""
		self.brightness_value_now = value
		print('Brightness: ',value)
		self.update()
		
		
	def blur_value(self,value):
		""" This function will take value from the slider 
			for the blur from 0 to 99 """
		self.blur_value_now = value
		print('Blur: ',value)
		self.update()
	
	
	def changeBrightness(self,img,value):
		""" This function will take an image (img) and the brightness
			value. It will perform the brightness change using OpenCv
			and after split, will merge the img and return it.
		"""
		hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		h,s,v = cv2.split(hsv)
		lim = 255 - value
		v[v>lim] = 255
		v[v<=lim] += value
		final_hsv = cv2.merge((h,s,v))
		img = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
		return img
		
	def changeBlur(self,img,value):
		""" This function will take the img image and blur values as inputs.
			After perform blur operation using opencv function, it returns 
			the image img.
		"""
		kernel_size = (value+1,value+1) # +1 is to avoid 0
		img = cv2.blur(img,kernel_size)
		return img
	
	def update(self):
		""" This function will update the photo according to the 
			current values of blur and brightness and set it to photo label.
		"""
		img = self.changeBrightness(self.image,self.brightness_value_now)
		img = self.changeBlur(img,self.blur_value_now)
		self.setPhoto(img)
	
	def savePhoto(self):
		""" This function will save the image"""
		# here provide the output file name
		# lets say we want to save the output as a time stamp
		# uncomment the two lines below
		
		# import time
		# filename = 'Snapshot '+str(time.strftime("%Y-%b-%d at %H.%M.%S %p"))+'.png'
		
		# Or we can give any name such as output.jpg or output.png as well
		# filename = 'Snapshot.png'	
	
		# Or a much better option is to let user decide the location and the extension
          	# using a file dialog.
		
		filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
		
		cv2.imwrite(filename,self.tmp)
		print('Image saved as:',self.filename)
	
	
	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "Pyshine photo editor"))
		self.pushButton_2.setText(_translate("MainWindow", "Open"))
		self.pushButton.setText(_translate("MainWindow", "Save"))

# Subscribe to PyShine Youtube channel for more detail! 

# WEBSITE: www.pyshine.com


if __name__ == "__main__":
	import sys
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow)
	MainWindow.show()
	sys.exit(app.exec_())



```

