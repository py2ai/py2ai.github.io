---
layout: post
title: Learn how to make PySide6 based GUI from PyQt5
mathjax: true
featured-img: 26072022-python-logo
summary:  Making an Image to Text based GUI with PySide6
---

The Importance of PySide6: Transitioning from PyQt5
In the realm of Python GUI (Graphical User Interface) development, libraries play a pivotal role in simplifying the creation of visually appealing and interactive applications. Among the various options available, PyQt5 and PySide6 stand out as two prominent choices. However, recent developments have underscored the significance of transitioning from PyQt5 to PySide6. In this blog post, we delve into the importance of PySide6 and guide developers through the process of migrating their codebase from PyQt5 to PySide6.


In a previous blog we made a code for pytesseract based GUI for all languages in PyQt5. This tutorial is also about creating a multi-language OCR GUI, but instead in PySide6. We will use the old main.ui script as it is and only change the `gui.py`. We start from very basic GUI in the Qt designer. We have tested various languages for image to text extraction process of pytesseract. These languages are tested for OCR: 
ARABIC, BENGALI, BULGARIAN, CHINESE(TRADITIONAL), CHINESE(SIMPLIFIED), DANISH, ENGLISH, FINNISH, FRENCH, GERMAN, GREEK, GUJRATI,
HINDI, HUNGRARIAN, IGBO, ITALIAN, JAPANESE, KANNADA, KAZAKH, KHMER, KOREAN, LAO, MACEDONIAN, MALAYALAM, MARATHI, NEPALI, RUSSIAN,  TURKISH, URDU.
<p align="center">
  <img src="https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/posts/PyshineTessGui.gif?raw=true" width="350" title="PyShine OCR GUI">
</p>

As a recap previously we answered these questions:
How to use OpenCV in PyQt5 GUI?
How to use tesseract optical character recognition in PyQt5 GUI?
How to use event filter in PyQt5 GUI?
How to use Dock widget and Q Rubber band in PyQt5?
How to Perform OCR on multiple languages?
How to make use of .ui file without converting it to .py file?
Link to tesseract-OCR: https://github.com/UB-Mannheim/tesseract/wiki

You can visit the following video for the background in PyQt5.
<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/lGeM3lSdwRM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>


### Again to run this application, you require python 3.12 and PySide6. Copy the below main.ui file and save it in a new directory  as main.ui

### Also save the gui.py file in that directory and run the gui.py file. Open any image of your language of interest and play with it. If you have questions,
### suggestions please comment, share and subscribe to PyShine youtube channel

No more wait! Here is the ui file:

## main.ui
{% include codeHeader.html %}
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
### PySide6 Code

## gui.py
{% include codeHeader.html %}
```python
from PySide6.QtGui import QImage, QPixmap, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QRubberBand
from PySide6 import QtCore, QtWidgets
from PySide6.QtUiTools import QUiLoader
import glob
import cv2
import pytesseract
import os
from PIL import Image
import sys

import sys
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QFile, QIODevice
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

class PyShine_OCR_APP(QMainWindow):
    def __init__(self):
        super(PyShine_OCR_APP, self).__init__()
        ui_file_name = 'main.ui'
        ui_file = QFile(ui_file_name)
        if not ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
            sys.exit(-1)
        loader = QUiLoader()
        
        self.ui = loader.load(ui_file_name)
        self.setCentralWidget(self.ui)  # Set the loaded UI as the central widget

        self.image = None
        
        self.ui.pushButton.clicked.connect(self.open)
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.ui.label_2.setMouseTracking(True)
        self.ui.label_2.installEventFilter(self)
        self.ui.label_2.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Access comboBox and comboBox_2 through the central widget
        self.comboBox = self.ui.dockWidgetContents_2.findChild(QtWidgets.QComboBox, "comboBox")
        self.comboBox_2 = self.ui.dockWidgetContents_2.findChild(QtWidgets.QComboBox, "comboBox_2")
        
        self.language = 'eng'
        self.comboBox.addItems(language_names_list)
        self.comboBox.currentIndexChanged.connect(self.update_now)
        self.comboBox.setCurrentIndex(language_names_list.index(self.language))
        
        self.font_size = '20'
        self.text = ''
        self.comboBox_2.addItems(font_list)
        self.comboBox_2.currentIndexChanged.connect(self.update_font_size)
        self.comboBox_2.setCurrentIndex(font_list.index(self.font_size))
        
        self.ui.textEdit.setFontPointSize(int(self.font_size))
        self.setAcceptDrops(True)
        
    def update_now(self, value):
        self.language = language_names_list[value]
        
        print('Language Selected as:', self.language)
    
    def update_font_size(self, value):
        self.font_size = value
        self.ui.textEdit.setFontPointSize(int(self.font_size))
        self.ui.textEdit.setText(str(self.text))
    
    def open(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Select File')
        self.image = cv2.imread(filename)
        frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ui.label_2.setPixmap(QPixmap.fromImage(image))
    
    def image_to_text(self, crop_cvimage):
        gray = cv2.cvtColor(crop_cvimage, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 1)
        crop = Image.fromarray(gray)
        print(self.language)
        text = pytesseract.image_to_string(crop, lang=self.language)
        print('Text:', text)
        return text
    
    def eventFilter(self, source, event):
        width = 0
        height = 0
        if (event.type() == QtCore.QEvent.MouseButtonPress and source is self.ui.label_2):
            self.org = self.mapFromGlobal(event.globalPos())
            self.left_top = event.pos()
            self.rubberBand.setGeometry(QtCore.QRect(self.org, QtCore.QSize()))
            self.rubberBand.show()
        elif (event.type() == QtCore.QEvent.MouseMove and source is self.ui.label_2):
            if self.rubberBand.isVisible():
                self.rubberBand.setGeometry(QtCore.QRect(self.org, self.mapFromGlobal(event.globalPos())).normalized())
        elif (event.type() == QtCore.QEvent.MouseButtonRelease and source is self.ui.label_2):
            if self.rubberBand.isVisible():
                self.rubberBand.hide()
                rect = self.rubberBand.geometry()
                self.x1 = self.left_top.x()
                self.y1 = self.left_top.y()
                width = rect.width()
                height = rect.height()
                self.x2 = self.x1 + width
                self.y2 = self.y1 + height
            if width >= 10 and height >= 10 and self.image is not None:
                self.crop = self.image[self.y1:self.y2, self.x1:self.x2]
                cv2.imwrite('cropped.png', self.crop)
                self.text = self.image_to_text(self.crop)
                self.ui.textEdit.setText(str(self.text))
            else:
                self.rubberBand.hide()
        else:
            return 0
        return super(PyShine_OCR_APP, self).eventFilter(source, event)

# www.pyshine.com
app = QApplication(sys.argv)
mainWindow = PyShine_OCR_APP()
mainWindow.show()
sys.exit(app.exec())


```


Why PySide6?
1. Licensing Issues:
PyQt5 is released under the GPL (General Public License) which mandates certain obligations if the software is used in proprietary applications. This has posed challenges for commercial projects. PySide6, on the other hand, is distributed under the more permissive LGPL (Lesser General Public License) and Apache 2.0, making it a preferred choice for commercial development.

2. Community Support:
PySide6 is actively supported by The Qt Company, fostering a robust ecosystem of developers and contributors. This ensures timely updates, bug fixes, and support for new features, enhancing the reliability and stability of applications built using PySide6.

3. Compatibility with Qt:
PySide6 maintains compatibility with the latest Qt framework, providing access to the latest features and improvements introduced in Qt 6. This allows developers to leverage cutting-edge technologies and stay aligned with the evolving landscape of GUI development.

4. Ease of Use:
PySide6 offers a more Pythonic API compared to PyQt5, making it intuitive and easier to grasp for Python developers. The codebase is cleaner and more aligned with Pythonic conventions, resulting in improved readability and maintainability of code.

Migrating from PyQt5 to PySide6
Migrating existing code from PyQt5 to PySide6 can be a straightforward process with careful planning and execution. Here’s a step-by-step guide to facilitate the transition:

1. Assess Dependencies:
Start by identifying any PyQt5-specific dependencies in your project. This includes PyQt5 modules, classes, and functions that need to be replaced with their PySide6 counterparts.

2. Replace PyQt5 Imports:
Update import statements throughout your codebase to reference PySide6 instead of PyQt5. This involves replacing PyQt5 with PySide6 in import statements and adjusting module names if necessary.

3. Review API Changes:
PySide6 may introduce slight differences in the API compared to PyQt5. Review the PySide6 documentation to familiarize yourself with any changes and adjust your code accordingly.

4. Testing and Debugging:
Thoroughly test your application after migrating to PySide6 to ensure that it behaves as expected. Address any compatibility issues or unexpected behavior encountered during testing.

5. Optimize Performance:
Take advantage of any performance improvements or optimizations offered by PySide6 compared to PyQt5. This may involve revisiting certain aspects of your codebase to leverage new features or enhancements in PySide6.

6. Update Documentation:
Finally, update your project documentation to reflect the migration to PySide6. This includes updating README files, installation instructions, and any other relevant documentation for users and contributors.



After saving both files, follow the video tutorial above to run the GUI

```
python gui.py
```

Thats it! As for reference you can use the following PyQt5 code for reference: 
### Reference PyQt5 Code
## gui.py
{% include codeHeader.html %}
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
