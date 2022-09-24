---
layout: post
title: How to make an image to text classifier application
mathjax: true
featured-img: ocr1917092022
summary:  This tutorial is about language classification based gui development with PyQt5 and Pytesseract
---
Hi friends following is the code for Part 19 of the PyQt5 learning series. For details visit the pyshine channel below.

<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/Gs9ESPWY7es" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>

# How to improve accuracty of tesseract

The default Page segmentation method (psm) in tesseract is page page of text. It means that tesseract expects a page of text when it segments an image. If you’re going to crop small region for OCR, try a different segmentation mode, using the --psm argument. Note that adding a white border to text which is too tightly cropped may also help. We can see a list of available psm modes using `tesseract --help-psm`.

```
(base) PS C:\Users\ps\Desktop> tesseract --help-psm
```

```
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
       bypassing hacks that are Tesseract-specific.
```

There are a number of other factors that can help in increasing the accuracy of tesseract
* Rotation / Deskewing - rotating image to align the text horizontally and deskewing to make better orientation of characters
* Rescaling - resizing the image
* Binarisation - convert an image to black and white
* Dilation and Erosion - fixing image artifacts via image processing
* Borders - removing dark borders
* Transparency / Alpha channel - removing the alpha channel from image before ocr 
* Noise Removal - removing noise like salt and pepper etc. before binarization
* Inverting images - using dark text on lighter background helps in improving accuracy
* oem - OCR Engine Mode

The available oem engines can be found via `tesseract --help-oem`:

```
(py38) PS C:\Users\ps\Desktop\ocr-19> tesseract --help-oem
```

```
OCR Engine modes:
  0    Legacy engine only.
  1    Neural nets LSTM engine only.
  2    Legacy + LSTM engines.
  3    Default, based on what is available.
```
Tesseract 4 contains LSTM neural net mode which mostly works best, but you are free to try.

In the following gui.py code we will use following configurations `custom_oem_psm_config = r'--oem 3 --psm 6'` and then use like this `pytesseract.image_to_string(image, config=custom_oem_psm_config)`


# main.ui
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
    <width>1536</width>
    <height>1006</height>
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
         <width>1464</width>
         <height>560</height>
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
        <widget class="QPushButton" name="pushButton_2">
         <property name="text">
          <string>Zoom In</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QPushButton" name="pushButton_3">
         <property name="text">
          <string>Zoom Out</string>
         </property>
        </widget>
       </item>
       <item>
        <widget class="QPushButton" name="pushButton_4">
         <property name="text">
          <string>Reset</string>
         </property>
        </widget>
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

# gui.py
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
from PIL import ImageQt
import numpy as np
from tesserocr import PyTessBaseAPI
# Here we will get the path of the tessdata
# For 64 bit installation of tesseract OCR 
language_path = 'C:\\Program Files\\Tesseract-OCR\\tessdata\\'
language_path_list = glob.glob(language_path+"*.traineddata")

custom_oem_psm_config = r'--oem 3 --psm 6' # Assume a single uniform block of text.

language_names_list = []

for path in language_path_list:
	base_name =  os.path.basename(path)
	base_name = os.path.splitext(base_name)[0]
	language_names_list.append(base_name)



languages = ['afr', 'amh', 'ara', 'asm', 'aze', 'aze_cyrl', 'bel', 'ben', 'bod', 
'bos', 'bre', 'bul', 'cat', 'ceb', 'ces', 'chi_sim', 'chi_sim_vert', 
'chi_tra', 'chi_tra_vert', 'chr', 'cym', 'dan',   'ell', 
'eng', 'enm', 'epo', 'equ', 'est', 'eus', 'fas', 'fil', 'fin', 'fra', 
'frk', 'frm', 'fry', 'gla', 'gle', 'glg', 'grc', 'guj', 'hat', 'heb', 
'hin', 'hrv', 'hun', 'hye', 'iku', 'ind', 'isl', 'ita', 'ita_old', 
'jav', 'jpn', 'jpn_vert', 'kan', 'kat', 'kat_old', 'kaz', 'khm', 
'kmr', 'kor', 'lao', 'lav', 'lit', 'ltz', 'mal', 'mar', 'mkd', 
'mlt', 'mon', 'mri', 'msa', 'mya', 'nep', 'nld', 'nor',  'ori', 
'osd', 'pan', 'pol', 'por', 'pus', 'que', 'ron', 'rus', 'san', 
'sin', 'slk', 'slv', 'snd', 'spa', 'spa_old', 'sqi', 'srp', 
'srp_latn', 'sun', 'swa', 'swe', 'syr', 'tam', 'tat', 'tel', 
'tgk', 'tha', 'tir', 'ton', 'tur', 'uig', 'ukr', 'urd', 'uzb', 'uzb_cyrl', 'vie', 'yid', 'yor']

def get_score(image, language):
    score =0
    with PyTessBaseAPI() as api:
        api.Init(lang=language)
        api.SetImageFile(image)
        awc = list(api.AllWordConfidences())
        score = sum(awc)/(1e-20+float(len(awc)))
    return score

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
        self.ui.pushButton_2.clicked.connect(self.zoom_in)
        self.ui.pushButton_3.clicked.connect(self.zoom_out)
        self.ui.pushButton_4.clicked.connect(self.reset_zoom)

        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidget(self.ui.label_2)
        self.scrollArea.setVisible(True)
        self.ui.setCentralWidget(self.scrollArea)


        self.rubberBand = QRubberBand(QRubberBand.Rectangle,self)
        self.ui.label_2.setMouseTracking(True)
        self.ui.label_2.setScaledContents(True)
        


        self.ui.label_2.installEventFilter(self)
        self.ui.label_2.setAlignment(PyQt5.QtCore.Qt.AlignTop)
        
        self.language = 'eng'
        self.comboBox.addItems(languages)
        self.comboBox.currentIndexChanged['QString'].connect(self.update_now)
        self.comboBox.setCurrentIndex(languages.index(self.language))
        
        self.font_size = '20'
        self.text = ''
        self.comboBox_2.addItems(font_list)
        self.comboBox_2.currentIndexChanged['QString'].connect(self.update_font_size)
        self.comboBox_2.setCurrentIndex(font_list.index(self.font_size))
        
        self.ui.textEdit.setFontPointSize(int(self.font_size))
        self.setAcceptDrops(True)

        self.zoom_in,self.zoom_out,self.reset_zoom = False,False, False
        self.ui.label_2.setScaledContents(True)
        self.scale_factor = 1.0
        self.ui.label_2.adjustSize()
        self.image = ''
    
    def zoom_in(self):
        self.zoom_in, self.zoom_out,self.reset_zoom = True, False, False
        self.zoom()
    
    def zoom_out(self):
        self.zoom_in, self.zoom_out,self.reset_zoom = False, True, False
        self.zoom()

    def reset_zoom(self):
        self.zoom_in, self.zoom_out,self.reset_zoom = False, False, True
        self.zoom()
    
    def zoom(self):

        if (self.scale_factor > 3.0 and self.zoom_in) or (self.scale_factor < 0.3 and self.zoom_out):
            return
        try:
            if self.zoom_in:
                frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                image =  QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
                self.ui.label_2.setPixmap(QPixmap.fromImage(image))
                self.scale_factor *= 1.1
                newScaleImage = self.scale_factor * self.ui.label_2.pixmap().size()
                self.ui.label_2.resize(newScaleImage)
                self.ui.label_2.setPixmap(QPixmap.fromImage(image).scaled(
                self.ui.label_2.size(), QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation))
            elif self.zoom_out:
                frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                image =  QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
                self.ui.label_2.setPixmap(QPixmap.fromImage(image))
                self.scale_factor /=1.1
                newScaleImage = self.scale_factor * self.ui.label_2.pixmap().size()
                self.ui.label_2.resize(newScaleImage)
                self.ui.label_2.setPixmap(QPixmap.fromImage(image).scaled(
                self.ui.label_2.size(), QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation))
            elif self.reset_zoom:
                frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                image =  QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
                self.ui.label_2.setPixmap(QPixmap.fromImage(image))
                self.scale_factor = 1.0
                self.ui.label_2.resize(self.ui.label_2.pixmap().size())
                self.ui.label_2.setPixmap(QPixmap.fromImage(image).scaled(
                self.ui.label_2.size(), QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation))
        except Exception as e:
            print(e)
            pass
            
        
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
        
        image = ImageQt.fromqpixmap(self.ui.label_2.pixmap())
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.ui.label_2.adjustSize()


    def image_to_text(self,crop_cvimage):
        gray = cv2.cvtColor(crop_cvimage,cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,1)
        crop = Image.fromarray(gray)
        

        output = [get_score('cropped.png',language) for language in  languages] # get scores of all languages
        index  = np.array(output).argmax() # get index of max in the list
        print(f'detected language is {languages[index]}  score {max(output)}')

        self.language = languages[index]
        self.comboBox.setCurrentIndex(languages.index(self.language))
        
        #text = pytesseract.image_to_string(crop,lang = self.language)
	text = pytesseract.image_to_string(crop,lang = self.language, config=custom_oem_psm_config)
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
                image = ImageQt.fromqpixmap(self.ui.label_2.pixmap())
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                self.crop = image[self.y1:self.y2, self.x1:self.x2]
                
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

