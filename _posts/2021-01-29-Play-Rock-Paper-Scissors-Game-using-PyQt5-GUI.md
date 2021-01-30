---
layout: post
title: Play Rock Paper Scissors Game using PyQt5 GUI
categories: [GUI tutorial series]
mathjax: true
featured-img: pyqtRPS
summary: This tutorial is about making a GUI in PyQt5 using OpenCV and Keras to play Rock Paper Scissors Game.
---

Hi and welcome! It is part 14 of the PyQt5 learning series. Today we will design a GUI to play Rock Paper Scissors (RPS) game. Most of you already have known or played this game. 
But let's make it straightforward for those who are new to this game. It is a priority-based hand playing game between
two people, where each player simultaneously makes three shapes with a hand. These shapes include a rock (or stone with a closed fist), a paper (with all fingers stretched),
and scissors (like two fingers making a victory sign). 

Possible outcomes of each draw of the game are either win/loss for one player or a tie for both. The decision
is made based on predecided priorities. The rock will beat scissors, the scissors will beat paper, and the paper will beat rock (because it can cover the rock). If
both players select the same sign, there will be a tie. This game is often played as a proper choosing method between two people, similar to a coin's flipping. Alright, from the machine learning perspectives, we have
some questions. If one player is human and another is a computer, how can one let computers know a human player's choice? How to let a computer decide a choice randomly?
How to make this game in a single graphical user interface (GUI)?

To answer these questions, we will develop a GUI. The computer will get a dataset from the user about the possible shapes. With the help of deep learning neural network, it will learn to classify during the game (each of the possible shapes in an image frame from the video or a live camera). The computer will simultaneously
select and decide the winner by knowing both preferences (from computer and human). The GUI will display the name of the winner using the above-described priorities.

The critical points of this GUI include:

1. Acquire the images from a user webcam using OpenCV; the amount of images data belonging to each class is given by the user besides the label's name.
2. Train a deep neural network model using Keras and generate an output .h5 model file.
3. Load the trained .h5 model and start playing the game. Notice that this model's purpose is only to classify the image as Rock, Paper, Scissors, or None.

Here is final look of this simple GUI:

[![GIF](https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/posts/rpsgui.gif?raw=true)](https://www.youtube.com/embed/UWxWcSxymHs "GIF")


We will proceed in several steps to finish this GUI project. First let's have look at the structure of this GUI project.

### 1. Structure of project directory 
In the main project directory ```14-Play Rock Paper Scissors Game using PyQt5 GUI``` we have these items:

 ```
14-Play Rock Paper Scissors Game using PyQt5 GUI/
├── dataset/
│   ├── None/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
|   |   |...... 
|   |   |...... 
│   │   └── 400.jpg
│   ├── Paper/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
|   |   |...... 
|   |   |...... 
│   │   └── 400.jpg
│   ├── Rock/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
|   |   |...... 
|   |   |...... 
│   │   └── 400.jpg
│   └── Scissors/
│   |   ├── 1.jpg
│   |   ├── 2.jpg
│   │   ├── 3.jpg
|   |   |...... 
|   |   |...... 
│   │   └── 400.jpg
├── images/
│   ├── Paper.jpg
│   ├── Rock.jpg
│   ├── Scissors.jpg
│   └── Try.png
├── out.avi
├── RPS.h5
└── RPS.py
 
 ```
 
 
  1. ```dataset``` directory: This will be generated automatically, once we press the Acquire data button on the GUI. Simply
  provide the label and the number of samples for each label. The label will be used to generate a subdirectory with the 
  same name as label. So please use it accordingly. The ```dataset``` should contain the acquired data (>100 files) each in separate subdirectories as: 
    
    (a) ```None```: any blank image image except the possible shapes like rock paper or scissors. 
    
    (b) ```Paper```: The images containing paper shaped hand with a plane background.
    
    (c) ```Rock```: The images containing rock shaped hand with a plane background.
            
    (d) ```Scissors```: The images containing scissors shaped hand with a plane background.
   
  
  2. ```images``` directory: contains images that the computer will show for its decisions on the GUI.
  
  3. ```out.avi``` file: A sample avi video file for the demonstration purpose.
  
  4. ```RPS.h5```: A keras trained model to predict the shape in an input image.
  
  5. ```RPS.py```: Our main code that will run all things in the GUI.
  
  In a project directory to run the GUI simply use:
  ```
  python RPS.py
  ```
  We will now proceed to explain the details.
  


### 2. Importing essentials
We can install them using pip install and then import them as:

```python
from random import choice
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import  imutils
import time
import numpy as np
import cv2,os
import pyshine as ps
from threading import Thread 
from keras.models import load_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

```
### 3. A global Epoch counter

```python
global epoch_cnt
epoch_cnt = 0
```
Here we have the ``epoch_cnt`` that will be used in the training process to update the progress bar.

### 4. PyShine_Callback for the end of each training Epoch

```python
class PyShine_Callback(Callback):
	def on_epoch_end(self, epoch, logs=None):
		""" This function will continue to update the current running Epoch count """
		global epoch_cnt
		epoch_cnt+=1
```
This class will continue to update the ```epoch_cnt``` counter for the display purpose of progress bar.

### 5. Main window class

```python
class Ui_MainWindow(object):
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(1280, 800)
		...
		...
		...
		self.W = 28
		self.H = 28
		self.winner='None'
		self.radioButton.setEnabled(False)
```
Here after initializing the setupUi for the MainWindow we will initialize all the parameters of the GUI just like we did in the
previous PyQt5 tutorials on pyshine.com.

### 6. Input modes functions

```python
def selectCam(self):
  """ This function will set the camera mode to True
    so that webcam frames can be played
  """
  self.cam = True

def selectVideo(self):
  """ This function will set the camera mode to False
    so that video file can be played
  """
  self.cam = False
```
The above two functions wil be used to select the mode, either video input or camera input.

### 7. Start the Training

```python
def start_training(self):
  """
  This function will initiate two threads once the training is required
  1. run the train_model function
  2. run the progress bar which will display the epochs in terms of percentage
  """
  global epoch_cnt
  self.th = Thread(target = self.train_model,args = ())
  self.th.start()	
  self.update_train_progress()
```
The above function will start the training process.

### 8. Load the trained model

```python
def loadModel(self):
  """
  This function will open a file dialog to let user load the model (.h5 file only)
  after that set the status message
  """
  model_filename = QFileDialog.getOpenFileName(filter="Keras (*.h5)")[0]
  try:
    self.loaded_model = load_model(model_filename)
    self.test = True
    self.pushButton_4.setEnabled(True)
    self.radioButton.setEnabled(True)
    self.label_3.setText("STATUS: Model loaded! Press Start")
  except Exception as e:
    pass
    print(e)
    self.label_3.setText("STATUS: {}".format(e))

```
The above function will be used to load the .h5 file using a file dialog. This step is important before starting the game.

### 9. Find the winner

```python
def find_winner(self,predicted_name, pc_selected_name):
  """
  This function will input the predicted_name (user image predicted by the model) and
  pc_selected_name (random guess of the pc) and then decided by returning the winner based on
  standard priority of the Rock Paper Scissors Game :)
  """
  if predicted_name == pc_selected_name:
    return "Tie"

  if predicted_name == "Rock":
    if pc_selected_name == "Scissors":
      return "User"
    if pc_selected_name == "Paper":
      return "Computer"

  if predicted_name == "Paper":
    if pc_selected_name == "Rock":
      return "User"
    if pc_selected_name == "Scissors":
      return "Computer"

  if predicted_name == "Scissors":
    if pc_selected_name == "Paper":
      return "User"
    if pc_selected_name == "Rock":
      return "Computer"
```
Once both players have made a choice, the human choice will be obtained via trained model and we will call it ```predicted_name```.
The above function will use another ```pc_selected_name``` to return the winner or a Tie.


### 10. Update the progress bar to show training progress
```python
def update_train_progress(self):
  """ This function is responsible to update the progress bar to show training percentage """

  print('Training started...')
  global epoch_cnt

  prev=0

  while True:
    if epoch_cnt>prev:
      value = int((epoch_cnt/self.EPOCHS)*100)
      self.progressBar_2.setValue(value)
      prev = epoch_cnt
    QtWidgets.QApplication.processEvents()	
    if epoch_cnt==self.EPOCHS:
      self.progressBar_2.setValue(100)
      break
```
The above function will update the progress bar 2 once the epoch is incremented. Notice how we map the EPOCHS to a percentage.


### 11. Mapping and Demapping the labels 

```python
	def mapper(self,labels):
		""" This function will map the string labels in self.CLASS_DICT
			to their corresponding integer values """
		return self.CLASS_DICT[labels]

	def demapper(self,val):
		""" This function will map the integer values in self.REV_CLASS_DICT
		to their corresponding string labels """
		return self.REV_CLASS_DICT[val]
```
The above two functions will perform mapping to let the machine learn the repective integer for a label. After training,
the demapper will be used to represent the predicted integer to its respective label such as Rock, Paper, Scissors, None.

### 12. Training the model

```python
def train_model(self):
  """ This function will call the Keras model and 
    map the data and labels, provide itto the model,
    compile and train to generate a .h5 file
  """

  self.label_3.setText("STATUS: Training in progress, please wait!")
  NUM_CLASSES = len(self.CLASS_DICT)

  dataset = []
  for directory in os.listdir(self.IMAGES_DATASET):
    path = os.path.join(self.IMAGES_DATASET, directory)
    if not os.path.isdir(path):
      continue
    for item in os.listdir(path):
      if item.startswith("."):
        continue
      img = cv2.imread(os.path.join(path, item))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, (self.W, self.H))
      img = img.astype('float32')
      img = img/255
      dataset.append([img, directory])

  data, labels = zip(*dataset)
  labels = list(map(self.mapper, labels))
  labels = np_utils.to_categorical(labels)


  (trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.25, random_state=42)

  # construct the image generator for data augmentation
  aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
  height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
  horizontal_flip=True, fill_mode="nearest")


  INIT_LR = 1e-3
  opt = Adam(lr=INIT_LR, decay=INIT_LR / self.EPOCHS)
  sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


  model =ps.RPSNET.build(width=self.W, height=self.H, depth=3, classes=NUM_CLASSES)
  model.compile(
  optimizer=opt,
  loss='categorical_crossentropy',
  metrics=['accuracy']
  )

  cl = PyShine_Callback() 	
  BS = 32


  H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
  validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
  epochs=self.EPOCHS, verbose=1,callbacks=[cl])
  print("Training network...")

  model.save("RPS.h5")
  K.clear_session()
  self.label_3.setText("STATUS: Training finished! Press Load Model")
```
The above function will scan the dataset directory and make the dataset and labels so that it can be used to 
train the model which has CNN architecture especially configured for the Rock Paper Scissors Network (RPSNET). 

### 13. Load video or camera input to start the game

```python
def loadImage(self):
  """ This function will load the camera device, obtain the image
    and set it to label using the setPhoto function
  """

  if self.started:
    self.started=False
    self.pushButton_4.setText('Start')	
    self.pushButton_2.setEnabled(True)
    self.pushButton_3.setEnabled(True)

  else:
    self.started=True
    self.pushButton_4.setText('Stop')
    self.pushButton_2.setEnabled(False)
    self.pushButton_3.setEnabled(False)


  if self.cam:
    vid = cv2.VideoCapture(0)

  else:

    video_filename =  'out.avi'
    vid = cv2.VideoCapture(video_filename)


  cnt=0
  frames_to_count=20
  st = 0
  fps=0
  sample_count=0
  prev_move = None
  while(vid.isOpened()):

    _, self.image = vid.read()
    try:
      self.image  = imutils.resize(self.image ,height = 480 )
    except:
      break



    if cnt == frames_to_count:
      try: # To avoid divide by 0 we put it in try except
        self.fps = round(frames_to_count/(time.time()-st)) 						
        st = time.time()
        cnt=0
      except:
        pass

    cnt+=1
    if self.acquire:

      roi = self.image[80:310, 80:310]
      save_path = os.path.join(self.IMG_CLASS_PATH, '{}.jpg'.format(sample_count + 1))
      sample_count+=1
      Total = int(self.samples)
      value = (sample_count/Total)*100
      self.progressBar.setValue(value)
      cv2.imwrite(save_path, roi)

      if sample_count == int(self.samples):
        self.acquire = False
        sample_count = 0
        self.pushButton_3.setEnabled(True)
        self.pushButton_2.setEnabled(True)

    if self.test:
      roi = self.image[80:310, 80:310]
      time.sleep(0.033)
      img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, (self.W, self.H))

      img = img.astype('float32')


      img = img/255
      pred = self.loaded_model.predict(np.array([img]))

      pred_key = np.argmax(pred[0])
      predicted_name = self.demapper(pred_key)

      self.image = ps.putBText(self.image,predicted_name.upper(),text_offset_x=80,text_offset_y=10,font_scale=1.5,text_RGB=(220,0,0))

      # Find who is the winner
      if prev_move != predicted_name:
        if predicted_name != "None":
          pc_selected_name = choice(['Rock', 'Paper', 'Scissors'])
          self.winner = self.find_winner(predicted_name, pc_selected_name)
          if self.winner == 'Computer':
            self.groupBox.setStyleSheet("background-color: rgb(255, 255, 255);")
            self.groupBox_2.setStyleSheet("background-color: rgb(0, 255, 127);")
          elif self.winner == 'User':
            self.groupBox_2.setStyleSheet("background-color: rgb(255, 255, 255);")
            self.groupBox.setStyleSheet("background-color: rgb(0, 255, 127);")
          else:
            self.groupBox_2.setStyleSheet("background-color: rgb(0, 255, 127);")
            self.groupBox.setStyleSheet("background-color: rgb(0, 255, 127);")

        else:
          pc_selected_name = "None"
          self.winner = "Waiting..."
      prev_move = predicted_name
      self.label_3.setText("STATUS: {}".format(self.winner).upper())

      if pc_selected_name =='Rock':
        self.DETECTED_IMAGE = self.ROCK_IMAGE
      elif pc_selected_name =='Paper':
        self.DETECTED_IMAGE = self.PAPER_IMAGE
      elif pc_selected_name =='Scissors':
        self.DETECTED_IMAGE = self.SCISSORS_IMAGE
      else:
        self.DETECTED_IMAGE = self.NONE_IMAGE



    self.update()
    key = cv2.waitKey(1) & 0xFF
    if self.started==False:
      break
```
This function is self explanatory, once the user hits the Start button, this function will be called. Based on the input type
the while loop will continue to call the update function. The inference of input image is performed via the loaded model
and the winner is displayed in the status.

### 14. Set photo on the label

```python

  def setPhoto(self,image):
    """ This function will take image input and resize it 
      only for display purpose and convert it to QImage
      to set at the label.
    """
    self.tmp = image
    self.tmp = cv2.rectangle(self.tmp, (80, 80), (310, 310), (0, 20, 200), 2)

    image = imutils.resize(image,height=480)
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
    self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    frame = cv2.cvtColor(self.DETECTED_IMAGE, cv2.COLOR_BGR2RGB)
    image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
    self.label_2.setPixmap(QtGui.QPixmap.fromImage(image))

```
Above function will display the image on the GUI with a red color rectangle of the region of interest (roi). 

### 15. Acquire the data
```python
def acquireData(self):
		""" This funciton will acquire the image data into the respective label directory """
		self.label_3.setText("STATUS: ")
		self.samples = self.lineEdit.text()
		self.label_name = self.lineEdit_2.text()
		self.generateDirs()
		self.acquire = True
		self.pushButton_4.setEnabled(True)
		self.pushButton_3.setEnabled(False)
		self.pushButton_2.setEnabled(False)
		self.radioButton.setEnabled(False)
    
 def generateDirs(self):
    """ This function will generate the Directorys for each label images data """

    self.IMG_CLASS_PATH = os.path.join(self.IMAGES_DATASET, self.label_name)

    try:
      os.mkdir(self.IMAGES_DATASET)
    except FileExistsError:
      pass
    try:
      os.mkdir(self.IMG_CLASS_PATH)
    except FileExistsError:
      print("{} FOLDER ALREADY EXISTS!".format(self.IMG_CLASS_PATH))
 ```
 Above function will set the flag ```self.acquire``` to True which initiate the process of acquiring data in the generated
 subdirectory once the user hit Start.
 
### 16. Update the photo on the label
 ```python
 
 	def update(self):
		""" This function will update the photo according to the 
			current values of blur and brightness and set it to photo label.
		"""
		img = self.image
		self.setPhoto(img)		
  ```
Above function will simply call the setPhoto function with the image img.

### 17. Retranslate the User Interface of GUI
```python
def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "PyShine RPS Application"))
		self.label_4.setText(_translate("MainWindow", "ROCK PAPER SCISSORS"))
		self.groupBox.setTitle(_translate("MainWindow", "User"))
		self.label.setText(_translate("MainWindow", "User Video"))
		self.groupBox_2.setTitle(_translate("MainWindow", "Computer"))
		self.label_2.setText(_translate("MainWindow", "Computer Video"))
		self.label_3.setText(_translate("MainWindow", "Please Load the Model (.h5) file..."))
		self.label_5.setText(_translate("MainWindow", "Enter Samples:"))
		self.lineEdit.setText(_translate("MainWindow", "400"))
		self.label_6.setText(_translate("MainWindow", "Enter Class Label:"))
		self.lineEdit_2.setText(_translate("MainWindow", "Paper"))
		self.pushButton.setText(_translate("MainWindow", "Acquire Data"))
		self.pushButton_2.setText(_translate("MainWindow", "Train Model"))
		self.pushButton_3.setText(_translate("MainWindow", "Load Model"))
		self.radioButton.setText(_translate("MainWindow", "Video input"))
		self.radioButton_2.setText(_translate("MainWindow", "Camera input"))
		self.pushButton_4.setText(_translate("MainWindow", "Start"))
	```
  
### 18. Run the application
```python
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
```
  
### COMPLETE CODE
Here is complete main code:
  
### RPS.py
  
  ```python
  
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RPS.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!
#
# Subscribe to PyShine Youtube channel for more detail! 
# 
# This code will let user to input: 1)video, 2) webcam and either train or test the model
#
# A user can generate data set by acquiring images and labels through GUI
#
# Train the model RPSNET via GUI, once .h5 file is generated deploy to start game
#
# Essentials can be installed through pip install: Tensorflow, cv2, PyQt5, numpy, imutils, pyshine, keras
#
# Usage: python RPS.py 
#
# Author: Pyshine www.pyshine.com


from random import choice
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import  imutils
import time
import numpy as np
import cv2,os
import pyshine as ps
from threading import Thread 
from keras.models import load_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


global epoch_cnt
epoch_cnt = 0

class PyShine_Callback(Callback):
	def on_epoch_end(self, epoch, logs=None):
		""" This function will continue to update the current running Epoch count """
		global epoch_cnt
		epoch_cnt+=1



class Ui_MainWindow(object):
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(1280, 800)
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		self.gridLayout_8 = QtWidgets.QGridLayout(self.centralwidget)
		self.gridLayout_8.setObjectName("gridLayout_8")
		self.gridLayout_5 = QtWidgets.QGridLayout()
		self.gridLayout_5.setObjectName("gridLayout_5")
		self.gridLayout = QtWidgets.QGridLayout()
		self.gridLayout.setObjectName("gridLayout")
		spacerItem = QtWidgets.QSpacerItem(158, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
		self.gridLayout.addItem(spacerItem, 0, 0, 1, 1)
		self.label_4 = QtWidgets.QLabel(self.centralwidget)
		self.label_4.setStyleSheet("font: 16pt \"MS Shell Dlg 2\";")
		self.label_4.setAlignment(QtCore.Qt.AlignCenter)
		self.label_4.setObjectName("label_4")
		self.gridLayout.addWidget(self.label_4, 0, 1, 1, 1)
		spacerItem1 = QtWidgets.QSpacerItem(118, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
		self.gridLayout.addItem(spacerItem1, 0, 2, 1, 1)
		self.gridLayout_5.addLayout(self.gridLayout, 0, 0, 1, 2)
		self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
		self.groupBox.setObjectName("groupBox")
		self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
		self.gridLayout_3.setObjectName("gridLayout_3")
		spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
		self.gridLayout_3.addItem(spacerItem2, 1, 1, 1, 1)
		self.label = QtWidgets.QLabel(self.groupBox)
		
		self.label.setObjectName("label")
		self.gridLayout_3.addWidget(self.label, 0, 1, 1, 1)
		spacerItem3 = QtWidgets.QSpacerItem(0, 188, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
		self.gridLayout_3.addItem(spacerItem3, 0, 2, 1, 1)
		self.gridLayout_5.addWidget(self.groupBox, 1, 0, 1, 1)
		self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
		self.groupBox_2.setObjectName("groupBox_2")
		self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
		self.gridLayout_2.setObjectName("gridLayout_2")
		self.label_2 = QtWidgets.QLabel(self.groupBox_2)

		self.label_2.setObjectName("label_2")
		self.gridLayout_2.addWidget(self.label_2, 0, 1, 1, 1)
		spacerItem4 = QtWidgets.QSpacerItem(0, 188, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
		self.gridLayout_2.addItem(spacerItem4, 0, 0, 1, 1)
		spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
		self.gridLayout_2.addItem(spacerItem5, 1, 1, 1, 1)
		self.gridLayout_5.addWidget(self.groupBox_2, 1, 1, 1, 1)
		self.gridLayout_4 = QtWidgets.QGridLayout()
		self.gridLayout_4.setObjectName("gridLayout_4")
		spacerItem6 = QtWidgets.QSpacerItem(148, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
		self.gridLayout_4.addItem(spacerItem6, 0, 0, 1, 1)
		self.label_3 = QtWidgets.QLabel(self.centralwidget)
		self.label_3.setStyleSheet("font: 16pt \"MS Shell Dlg 2\";")
		self.label_3.setAlignment(QtCore.Qt.AlignCenter)
		self.label_3.setObjectName("label_3")
		self.gridLayout_4.addWidget(self.label_3, 0, 1, 1, 1)
		spacerItem7 = QtWidgets.QSpacerItem(158, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
		self.gridLayout_4.addItem(spacerItem7, 0, 2, 1, 1)
		self.gridLayout_5.addLayout(self.gridLayout_4, 2, 0, 1, 2)
		self.gridLayout_8.addLayout(self.gridLayout_5, 0, 0, 1, 1)
		self.gridLayout_7 = QtWidgets.QGridLayout()
		self.gridLayout_7.setObjectName("gridLayout_7")
		self.gridLayout_6 = QtWidgets.QGridLayout()
		self.gridLayout_6.setObjectName("gridLayout_6")
		self.label_5 = QtWidgets.QLabel(self.centralwidget)
		self.label_5.setObjectName("label_5")
		self.gridLayout_6.addWidget(self.label_5, 0, 1, 1, 1)
		self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
		self.lineEdit.setObjectName("lineEdit")
		self.gridLayout_6.addWidget(self.lineEdit, 0, 2, 1, 1)
		self.label_6 = QtWidgets.QLabel(self.centralwidget)
		self.label_6.setObjectName("label_6")
		self.gridLayout_6.addWidget(self.label_6, 0, 3, 1, 1)
		self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
		self.lineEdit_2.setObjectName("lineEdit_2")
		self.gridLayout_6.addWidget(self.lineEdit_2, 0, 4, 1, 1)
		self.pushButton = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton.setObjectName("pushButton")
		self.gridLayout_6.addWidget(self.pushButton, 0, 5, 1, 1)
		self.gridLayout_7.addLayout(self.gridLayout_6, 0, 0, 1, 3)
		self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
		self.progressBar.setProperty("value", 0)
		self.progressBar.setObjectName("progressBar")
		self.gridLayout_7.addWidget(self.progressBar, 0, 3, 1, 3)
		self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_2.setObjectName("pushButton_2")
		self.gridLayout_7.addWidget(self.pushButton_2, 1, 0, 1, 1)
		self.progressBar_2 = QtWidgets.QProgressBar(self.centralwidget)
		self.progressBar_2.setProperty("value", 0)
		self.progressBar_2.setObjectName("progressBar_2")
		self.gridLayout_7.addWidget(self.progressBar_2, 1, 1, 1, 5)
		self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_3.setObjectName("pushButton_3")
		self.gridLayout_7.addWidget(self.pushButton_3, 2, 0, 1, 1)
		spacerItem8 = QtWidgets.QSpacerItem(346, 17, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
		self.gridLayout_7.addItem(spacerItem8, 2, 1, 1, 1)
		self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
		self.radioButton.setObjectName("radioButton")
		self.gridLayout_7.addWidget(self.radioButton, 2, 2, 1, 2)
		self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
		self.radioButton_2.setChecked(True)
		self.radioButton_2.setObjectName("radioButton_2")
		self.gridLayout_7.addWidget(self.radioButton_2, 2, 4, 1, 1)
		self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_4.setObjectName("pushButton_4")
		self.gridLayout_7.addWidget(self.pushButton_4, 2, 5, 1, 1)
		self.gridLayout_8.addLayout(self.gridLayout_7, 1, 0, 1, 1)
		MainWindow.setCentralWidget(self.centralwidget)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setObjectName("statusbar")
		MainWindow.setStatusBar(self.statusbar)

		self.pushButton_4.clicked.connect(self.loadImage)
		self.pushButton_3.clicked.connect(self.loadModel)
		self.pushButton_2.clicked.connect(self.start_training)
		self.pushButton.clicked.connect(self.acquireData)
		self.pushButton_4.setEnabled(False)
		self.radioButton.clicked.connect(self.selectVideo)
		self.radioButton_2.clicked.connect(self.selectCam)

		self.retranslateUi(MainWindow)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)
		self.started = False
		self.tmp = None 
		self.IMAGES_DATASET = 'dataset'
		self.acquire = False
		self.IMG_CLASS_PATH  = None
		self.loaded_model = None
		self.test  =  False
		self.cam = True 
		self.samples = self.lineEdit.text()
		self.label_name = self.lineEdit_2.text()
		self.EPOCHS = 10


		self.CLASS_DICT = {
		"Rock": 0,
		"Paper": 1,
		"Scissors": 2,
		"None": 3
		}
		
		self.REV_CLASS_DICT = {
		0: "Rock",
		1: "Paper",
		2: "Scissors",
		3: "None"
		}
		self.ROCK_IMAGE = cv2.imread('images/Rock.jpg')
		self.ROCK_IMAGE  = imutils.resize(self.ROCK_IMAGE ,height = 480 )
		self.PAPER_IMAGE = cv2.imread('images/Paper.jpg')
		self.PAPER_IMAGE  = imutils.resize(self.PAPER_IMAGE ,height = 480 )
		self.SCISSORS_IMAGE = cv2.imread('images/Scissors.jpg')
		self.SCISSORS_IMAGE  = imutils.resize(self.SCISSORS_IMAGE ,height = 480 )
		self.NONE_IMAGE = cv2.imread('images/Try.png')
		self.NONE_IMAGE  = imutils.resize(self.NONE_IMAGE ,height = 480 )
		self.DETECTED_IMAGE = self.NONE_IMAGE

		self.W = 28
		self.H = 28
		self.winner='None'
		self.radioButton.setEnabled(False)
		
	def selectCam(self):
		""" This function will set the camera mode to True
			so that webcam frames can be played
		"""
		self.cam = True
	
	def selectVideo(self):
		""" This function will set the camera mode to False
			so that video file can be played
		"""
		self.cam = False
	
	def start_training(self):
		"""
		This function will initiate two threads once the training is required
		1. run the train_model function
		2. run the progress bar which will display the epochs in terms of percentage
		"""
		global epoch_cnt
		self.th = Thread(target = self.train_model,args = ())
		self.th.start()	
		self.update_train_progress()
		
		
	def loadModel(self):
		"""
		This function will open a file dialog to let user load the model (.h5 file only)
		after that set the status message
		"""
		model_filename = QFileDialog.getOpenFileName(filter="Keras (*.h5)")[0]
		try:
			self.loaded_model = load_model(model_filename)
			self.test = True
			self.pushButton_4.setEnabled(True)
			self.radioButton.setEnabled(True)
			self.label_3.setText("STATUS: Model loaded! Press Start")
		except Exception as e:
			pass
			print(e)
			self.label_3.setText("STATUS: {}".format(e))

		
	def find_winner(self,predicted_name, pc_selected_name):
		"""
		This function will input the predicted_name (user image predicted by the model) and
		pc_selected_name (random guess of the pc) and then decided by returning the winner based on
		standard priority of the Rock Paper Scissors Game :)
		"""
		if predicted_name == pc_selected_name:
			return "Tie"

		if predicted_name == "Rock":
			if pc_selected_name == "Scissors":
				return "User"
			if pc_selected_name == "Paper":
				return "Computer"

		if predicted_name == "Paper":
			if pc_selected_name == "Rock":
				return "User"
			if pc_selected_name == "Scissors":
				return "Computer"

		if predicted_name == "Scissors":
			if pc_selected_name == "Paper":
				return "User"
			if pc_selected_name == "Rock":
				return "Computer"
		

	def update_train_progress(self):
		""" This function is responsible to update the progress bar to show training percentage """

		print('Training started...')
		global epoch_cnt
		
		prev=0
		
		while True:
			if epoch_cnt>prev:
				value = int((epoch_cnt/self.EPOCHS)*100)
				self.progressBar_2.setValue(value)
				prev = epoch_cnt
			QtWidgets.QApplication.processEvents()	
			if epoch_cnt==self.EPOCHS:
				self.progressBar_2.setValue(100)
				break
		
		
	def mapper(self,labels):
		""" This function will map the string labels in self.CLASS_DICT
			to their corresponding integer values """
		return self.CLASS_DICT[labels]

	def demapper(self,val):
		""" This function will map the integer values in self.REV_CLASS_DICT
		to their corresponding string labels """
		return self.REV_CLASS_DICT[val]
	
	def train_model(self):
		""" This function will call the Keras model and 
			map the data and labels, provide itto the model,
			compile and train to generate a .h5 file
		"""
		
		self.label_3.setText("STATUS: Training in progress, please wait!")
		NUM_CLASSES = len(self.CLASS_DICT)
		
		dataset = []
		for directory in os.listdir(self.IMAGES_DATASET):
			path = os.path.join(self.IMAGES_DATASET, directory)
			if not os.path.isdir(path):
				continue
			for item in os.listdir(path):
				if item.startswith("."):
					continue
				img = cv2.imread(os.path.join(path, item))
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				img = cv2.resize(img, (self.W, self.H))
				img = img.astype('float32')
				img = img/255
				dataset.append([img, directory])
		
		data, labels = zip(*dataset)
		labels = list(map(self.mapper, labels))
		labels = np_utils.to_categorical(labels)

		
		(trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.25, random_state=42)

		# construct the image generator for data augmentation
		aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
		height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
		horizontal_flip=True, fill_mode="nearest")

		
		INIT_LR = 1e-3
		opt = Adam(lr=INIT_LR, decay=INIT_LR / self.EPOCHS)
		sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

		
		model =ps.RPSNET.build(width=self.W, height=self.H, depth=3, classes=NUM_CLASSES)
		model.compile(
		optimizer=opt,
		loss='categorical_crossentropy',
		metrics=['accuracy']
		)

		cl = PyShine_Callback() 	
		BS = 32
		
		
		H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
		validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
		epochs=self.EPOCHS, verbose=1,callbacks=[cl])
		print("Training network...")

		model.save("RPS.h5")
		K.clear_session()
		self.label_3.setText("STATUS: Training finished! Press Load Model")
		
	def loadImage(self):
		""" This function will load the camera device, obtain the image
			and set it to label using the setPhoto function
		"""
		
		if self.started:
			self.started=False
			self.pushButton_4.setText('Start')	
			self.pushButton_2.setEnabled(True)
			self.pushButton_3.setEnabled(True)

		else:
			self.started=True
			self.pushButton_4.setText('Stop')
			self.pushButton_2.setEnabled(False)
			self.pushButton_3.setEnabled(False)
			
		
		if self.cam:
			vid = cv2.VideoCapture(0)
			
		else:
			
			video_filename =  'out.avi'
			vid = cv2.VideoCapture(video_filename)
			
		
		cnt=0
		frames_to_count=20
		st = 0
		fps=0
		sample_count=0
		prev_move = None
		while(vid.isOpened()):
			
			_, self.image = vid.read()
			try:
				self.image  = imutils.resize(self.image ,height = 480 )
			except:
				break
			
			
			
			if cnt == frames_to_count:
				try: # To avoid divide by 0 we put it in try except
					self.fps = round(frames_to_count/(time.time()-st)) 						
					st = time.time()
					cnt=0
				except:
					pass
			
			cnt+=1
			if self.acquire:
				
				roi = self.image[80:310, 80:310]
				save_path = os.path.join(self.IMG_CLASS_PATH, '{}.jpg'.format(sample_count + 1))
				sample_count+=1
				Total = int(self.samples)
				value = (sample_count/Total)*100
				self.progressBar.setValue(value)
				cv2.imwrite(save_path, roi)
				
				if sample_count == int(self.samples):
					self.acquire = False
					sample_count = 0
					self.pushButton_3.setEnabled(True)
					self.pushButton_2.setEnabled(True)
			
			if self.test:
				roi = self.image[80:310, 80:310]
				time.sleep(0.033)
				img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
				img = cv2.resize(img, (self.W, self.H))
				
				img = img.astype('float32')
		
				
				img = img/255
				pred = self.loaded_model.predict(np.array([img]))
				
				pred_key = np.argmax(pred[0])
				predicted_name = self.demapper(pred_key)
				
				self.image = ps.putBText(self.image,predicted_name.upper(),text_offset_x=80,text_offset_y=10,font_scale=1.5,text_RGB=(220,0,0))
	
				# Find who is the winner
				if prev_move != predicted_name:
					if predicted_name != "None":
						pc_selected_name = choice(['Rock', 'Paper', 'Scissors'])
						self.winner = self.find_winner(predicted_name, pc_selected_name)
						if self.winner == 'Computer':
							self.groupBox.setStyleSheet("background-color: rgb(255, 255, 255);")
							self.groupBox_2.setStyleSheet("background-color: rgb(0, 255, 127);")
						elif self.winner == 'User':
							self.groupBox_2.setStyleSheet("background-color: rgb(255, 255, 255);")
							self.groupBox.setStyleSheet("background-color: rgb(0, 255, 127);")
						else:
							self.groupBox_2.setStyleSheet("background-color: rgb(0, 255, 127);")
							self.groupBox.setStyleSheet("background-color: rgb(0, 255, 127);")

					else:
						pc_selected_name = "None"
						self.winner = "Waiting..."
				prev_move = predicted_name
				self.label_3.setText("STATUS: {}".format(self.winner).upper())
				
				if pc_selected_name =='Rock':
					self.DETECTED_IMAGE = self.ROCK_IMAGE
				elif pc_selected_name =='Paper':
					self.DETECTED_IMAGE = self.PAPER_IMAGE
				elif pc_selected_name =='Scissors':
					self.DETECTED_IMAGE = self.SCISSORS_IMAGE
				else:
					self.DETECTED_IMAGE = self.NONE_IMAGE
				
				

			self.update()
			key = cv2.waitKey(1) & 0xFF
			if self.started==False:
				break
				
	
	def setPhoto(self,image):
		""" This function will take image input and resize it 
			only for display purpose and convert it to QImage
			to set at the label.
		"""
		self.tmp = image
		self.tmp = cv2.rectangle(self.tmp, (80, 80), (310, 310), (0, 20, 200), 2)

		image = imutils.resize(image,height=480)
		frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
		self.label.setPixmap(QtGui.QPixmap.fromImage(image))

		frame = cv2.cvtColor(self.DETECTED_IMAGE, cv2.COLOR_BGR2RGB)
		image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
		self.label_2.setPixmap(QtGui.QPixmap.fromImage(image))
	
	def acquireData(self):
		""" This funciton will acquire the image data into the respective label directory """
		self.label_3.setText("STATUS: ")
		self.samples = self.lineEdit.text()
		self.label_name = self.lineEdit_2.text()
		self.generateDirs()
		self.acquire = True
		self.pushButton_4.setEnabled(True)
		self.pushButton_3.setEnabled(False)
		self.pushButton_2.setEnabled(False)
		self.radioButton.setEnabled(False)


	def generateDirs(self):
		""" This function will generate the Directorys for each label images data """
		
		self.IMG_CLASS_PATH = os.path.join(self.IMAGES_DATASET, self.label_name)

		try:
			os.mkdir(self.IMAGES_DATASET)
		except FileExistsError:
			pass
		try:
			os.mkdir(self.IMG_CLASS_PATH)
		except FileExistsError:
			print("{} FOLDER ALREADY EXISTS!".format(self.IMG_CLASS_PATH))


	
	def update(self):
		""" This function will update the photo according to the 
			current values of blur and brightness and set it to photo label.
		"""
		img = self.image
		self.setPhoto(img)			


	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "PyShine RPS Application"))
		self.label_4.setText(_translate("MainWindow", "ROCK PAPER SCISSORS"))
		self.groupBox.setTitle(_translate("MainWindow", "User"))
		self.label.setText(_translate("MainWindow", "User Video"))
		self.groupBox_2.setTitle(_translate("MainWindow", "Computer"))
		self.label_2.setText(_translate("MainWindow", "Computer Video"))
		self.label_3.setText(_translate("MainWindow", "Please Load the Model (.h5) file..."))
		self.label_5.setText(_translate("MainWindow", "Enter Samples:"))
		self.lineEdit.setText(_translate("MainWindow", "400"))
		self.label_6.setText(_translate("MainWindow", "Enter Class Label:"))
		self.lineEdit_2.setText(_translate("MainWindow", "Paper"))
		self.pushButton.setText(_translate("MainWindow", "Acquire Data"))
		self.pushButton_2.setText(_translate("MainWindow", "Train Model"))
		self.pushButton_3.setText(_translate("MainWindow", "Load Model"))
		self.radioButton.setText(_translate("MainWindow", "Video input"))
		self.radioButton_2.setText(_translate("MainWindow", "Camera input"))
		self.pushButton_4.setText(_translate("MainWindow", "Start"))

# www.pyshine.com




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

  ```
 
Thats all for today. Have a nice day!


