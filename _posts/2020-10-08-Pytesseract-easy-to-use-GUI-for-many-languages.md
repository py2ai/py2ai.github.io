---
layout: post
title: How to make an image to text converter GUI in Python
categories: [GUI tutorial series]
mathjax: true
featured-img: pyqt8
summary: Making a GUI to extract text of various languages by using the trained model of the specific language
---


Hello friends, here is the code for the new idea of making pytesseract based GUI for all languages in PyQt5. This tutorial is about creating a multi-language OCR GUI in PyQt5 in Python. We start from very basic GUI in the Qt designer. We have tested various languages for image to text extraction process of pytesseract. These languages are tested for OCR: 
ARABIC, BENGALI, BULGARIAN, CHINESE(TRADITIONAL), CHINESE(SIMPLIFIED), DANISH, ENGLISH, FINNISH, FRENCH, GERMAN, GREEK, GUJRATI,
HINDI, HUNGRARIAN, IGBO, ITALIAN, JAPANESE, KANNADA, KAZAKH, KHMER, KOREAN, LAO, MACEDONIAN, MALAYALAM, MARATHI, NEPALI, RUSSIAN,  TURKISH, URDU.
<p align="center">
  <img src="https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/posts/PyshineTessGui.gif?raw=true" width="350" title="PyShine OCR GUI">
</p>

In this tutorial you will learn:
How to use OpenCV in PyQt5 GUI?
How to use tesseract optical character recognition in PyQt5 GUI?
How to use event filter in PyQt5 GUI?
How to use Dock widget and Q Rubber band in PyQt5?
How to Perform OCR on multiple languages?
How to make use of .ui file without converting it to .py file?
Link to tesseract-OCR: https://github.com/UB-Mannheim/tesseract/wiki


<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/lGeM3lSdwRM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>

### To run this application, you require python 3. Copy the below main.ui file and save it in a new directory  as main.ui

### Also save the gui.py file in that directory and run the gui.py file. Open any image of your language of interest and play with it. If you have questions,
### suggestions please comment, share and subscribe to PyShine youtube channel

No more wait! Here is the ui file:

## main.ui

```python

<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>MainWindow</class>
 <widget class="QMainWindow" name="MainWindow">
  <property name="geometry">
   <rect>
    <x>0</x>
    <y>0</y>
    <width>800</width>
    <height>600</height>
   </rect>
  </property>
  <property name="windowTitle">
   <string>PyShine OCR GUI</string>
  </property>
  <widget class="QWidget" name="centralwidget">
   <layout class="QGridLayout" name="gridLayout_3">
    <item row="0" column="0">
     <widget class="QScrollArea" name="scrollArea">
      <property name="verticalScrollBarPolicy">
       <enum>Qt::ScrollBarAlwaysOn</enum>
      </property>
      <property name="horizontalScrollBarPolicy">
       <enum>Qt::ScrollBarAlwaysOn</enum>
      </property>
      <property name="widgetResizable">
       <bool>true</bool>
      </property>
      <widget class="QWidget" name="scrollAreaWidgetContents">
       <property name="geometry">
        <rect>
         <x>0</x>
         <y>0</y>
         <width>763</width>
         <height>275</height>
        </rect>
       </property>
       <layout class="QGridLayout" name="gridLayout">
        <item row="0" column="0">
         <widget class="QLabel" name="label_2">
          <property name="styleSheet">
           <string notr="true">background-color: rgb(170, 255, 255);</string>
          </property>
          <property name="text">
           <string>Photo</string>
          </property>
          <property name="alignment">
           <set>Qt::AlignCenter</set>
          </property>
         </widget>
        </item>
       </layout>
      </widget>
     </widget>
    </item>
   </layout>
  </widget>
  <widget class="QStatusBar" name="statusbar"/>
  <widget class="QDockWidget" name="dockWidget">
   <attribute name="dockWidgetArea">
    <number>8</number>
   </attribute>
   <widget class="QWidget" name="dockWidgetContents_2">
    <layout class="QGridLayout" name="gridLayout_2">
     <item row="1" column="0">
      <widget class="QTextEdit" name="textEdit"/>
     </item>
     <item row="0" column="0">
      <layout class="QHBoxLayout" name="horizontalLayout">
       <item>
        <widget class="QLabel" name="label_3">
         <property name="text">
          <string>Font Size:</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QComboBox" name="comboBox_2"/>
       </item>
       <item>
        <spacer name="horizontalSpacer">
         <property name="orientation">
          <enum>Qt::Horizontal</enum>
         </property>
         <property name="sizeHint" stdset="0">
          <size>
           <width>40</width>
           <height>20</height>
          </size>
         </property>
        </spacer>
       </item>
       <item>
        <widget class="QLabel" name="label">
         <property name="text">
          <string>Select Language:</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QComboBox" name="comboBox"/>
       </item>
       <item>
        <widget class="QPushButton" name="pushButton">
         <property name="text">
          <string>Open Image</string>
         </property>
        </widget>
       </item>
      </layout>
     </item>
    </layout>
   </widget>
  </widget>
 </widget>
 <resources/>
 <connections/>
</ui>

```
## gui.py

```python
# Welcome to PyShine
import pytesseract
import cv2, os,sys
from PIL import Image
import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore,QtGui,QtWidgets
import glob

# Here we will get the path of the tessdata
# For 64 bit installation of tesseract OCR 
language_path = 'C:\\Program Files\\Tesseract-OCR\\tessdata\\'
language_path_list = glob.glob(language_path+"*.traineddata")



language_names_list = []

for path in language_path_list:
	base_name =  os.path.basename(path)
	base_name = os.path.splitext(base_name)[0]
	language_names_list.append(base_name)



font_list = []
font = 2

for font in range(110):
	font+=2
	font_list.append(str(font))

# print('Font list:',font_list)

class PyShine_OCR_APP(QtWidgets.QMainWindow):
	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		self.ui = uic.loadUi('main.ui',self)
		self.image = None
		
		self.ui.pushButton.clicked.connect(self.open)
		self.rubberBand = QRubberBand(QRubberBand.Rectangle,self)
		self.ui.label_2.setMouseTracking(True)
		self.ui.label_2.installEventFilter(self)
		self.ui.label_2.setAlignment(PyQt5.QtCore.Qt.AlignTop)
		
		self.language = 'eng'
		self.comboBox.addItems(language_names_list)
		self.comboBox.currentIndexChanged['QString'].connect(self.update_now)
		self.comboBox.setCurrentIndex(language_names_list.index(self.language))
		
		self.font_size = '20'
		self.text = ''
		self.comboBox_2.addItems(font_list)
		self.comboBox_2.currentIndexChanged['QString'].connect(self.update_font_size)
		self.comboBox_2.setCurrentIndex(font_list.index(self.font_size))
		
		self.ui.textEdit.setFontPointSize(int(self.font_size))
		self.setAcceptDrops(True)
		
		
	def update_now(self,value):
		self.language = value
		print('Language Selected as:',self.language)
	
	def update_font_size(self,value):
		self.font_size = value
		self.ui.textEdit.setFontPointSize(int(self.font_size))
		self.ui.textEdit.setText(str(self.text))
	
	
	def open(self):
		filename = QFileDialog.getOpenFileName(self,'Select File')
		self.image = cv2.imread(str(filename[0]))
		frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
		image =  QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
		self.ui.label_2.setPixmap(QPixmap.fromImage(image))
	
	
	def image_to_text(self,crop_cvimage):
		gray = cv2.cvtColor(crop_cvimage,cv2.COLOR_BGR2GRAY)
		gray = cv2.medianBlur(gray,1)
		crop = Image.fromarray(gray)
		text = pytesseract.image_to_string(crop,lang = self.language)
		print('Text:',text)
		return text
	
	def eventFilter(self,source,event):
		width = 0
		height = 0
		if (event.type() == QEvent.MouseButtonPress and source is self.ui.label_2):
			self.org = self.mapFromGlobal(event.globalPos())
			self.left_top = event.pos()
			self.rubberBand.setGeometry(QRect(self.org,QSize()))
			self.rubberBand.show()
		elif (event.type() == QEvent.MouseMove and source is self.ui.label_2):
			if self.rubberBand.isVisible():
				self.rubberBand.setGeometry(QRect(self.org,self.mapFromGlobal(event.globalPos())).normalized())
		elif(event.type() == QEvent.MouseButtonRelease and source is self.ui.label_2):
			if self.rubberBand.isVisible():
				self.rubberBand.hide()
				rect = self.rubberBand.geometry()
				self.x1 = self.left_top.x()
				self.y1 = self. left_top.y()
				width = rect.width()
				height = rect.height()
				self.x2 = self.x1+ width
				self.y2 = self.y1+ height
			if width >=10 and height >= 10  and self.image is not None:
				self.crop = self.image[self.y1:self.y2, self.x1:self.x2]
				cv2.imwrite('cropped.png',self.crop)
				self.text = self.image_to_text(self.crop)
				self.ui.textEdit.setText(str(self.text))
			else:
				self.rubberBand.hide()
		else:
			return 0
		return QWidget.eventFilter(self,source,event)
	
# www.pyshine.com
app = QtWidgets.QApplication(sys.argv)
mainWindow = PyShine_OCR_APP()
mainWindow.show()
sys.exit(app.exec_())

```

After saving both files, follow the video tutorial above to run the GUI

```
python gui.py
```



