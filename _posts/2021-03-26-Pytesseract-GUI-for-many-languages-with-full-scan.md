---
layout: post
title: How to make an image to text GUI in Python
categories: [GUI tutorial series]
mathjax: true
featured-img: pytessgui
summary: Extract full text of various languages by using the trained model of the specific language
---


Hello friends, first of all thanks for your appreciations, comments and suggestions. One interesting suggestion was recently asked as the possibility of 
adding an additional button to the GUI of from our previous tutorial related to the PyQt5 tesseract. This button should let user extract the whole text instead of some part of it. Moreover, the crop function should also be there just like before. 

So yes, we can add such button to the UI in the PyQt5 designer. If you haven't installed the designer tool or PyQt5 then use the following in PowerShell or Terminal: 

```
pip3 install pyqt5
pip3 install pyqt5-tools
```
After that simply run the command below in Terminal:

```
qt5-tools designer
```

Before reading further it is highly recommended that you read this tutorial: https://pyshine.com/Pytesseract-easy-to-use-GUI-for-many-languages/ for basic installation of Tesseract.

Once the desginer window is launched please open up the ```main.ui``` file below:

### main.ui
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
        <widget class="QPushButton" name="pushButton_2">
         <property name="enabled">
          <bool>false</bool>
         </property>
         <property name="text">
          <string>Full Scan</string>
         </property>
        </widget>
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
This will be the GUI that our ```gui.py``` code will use to load it ```self.ui = uic.loadUi('main.ui',self)```. To scan the full text we have a new button
named ```Full Scan``` in the GUI. We will connect this button to another function using the following:


```
self.ui.pushButton_2.clicked.connect(self.full_scan_text)
```

The full text can be scanned easily with this little funciton which will get the text and set it to the text Edit window below as:

```
    def full_scan_text(self):
       
        self.text=self.image_to_text(self.image)
        self.ui.textEdit.setText(str(self.text))
```
 
 And finally here is the gui.py complete code. Please watch our PyQt5 tutorial for basics about more information about the following code:


### gui.py
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
        self.ui.pushButton_2.clicked.connect(self.full_scan_text)
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
        self.ui.pushButton_2.setEnabled(True)

    def full_scan_text(self):
       
        self.text=self.image_to_text(self.image)
        self.ui.textEdit.setText(str(self.text))

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


