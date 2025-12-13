---
categories:
- GUI tutorial series
description: Threads run in parallel
featured-img: threadspy
keywords:
- PyQt5
- QThread
- Multi-threading
- GUI Development
layout: post
mathjax: true
tags:
- PyQt5
- QThread
- Multi-threading
- GUI Development
title: Working with multiple threads in PyQt5
---



<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/k5tIk7w50L4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>

Hello friends! Today we will design a relatively simple GUI. It contains six buttons, three for starting three threads and three for stopping them. 
The code below is kept as simple as possible to understand the basic concept in handling the Qthread. We used progress bars because they can easily show a counter's progress, 
especially in a while loop. To run this GUI code, make a new folder and put these two files below in it. 

Run the code as python main.py

# threads.ui

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
    <height>269</height>
   </rect>
  </property>
  <property name="windowTitle">
   <string>MainWindow</string>
  </property>
  <widget class="QWidget" name="centralwidget">
   <layout class="QGridLayout" name="gridLayout_2">
    <item row="0" column="0">
     <widget class="QGroupBox" name="groupBox">
      <property name="title">
       <string>Testing threads</string>
      </property>
      <layout class="QGridLayout" name="gridLayout_3">
       <item row="0" column="0">
        <layout class="QGridLayout" name="gridLayout">
         <item row="0" column="2">
          <widget class="QProgressBar" name="progressBar">
           <property name="value">
            <number>24</number>
           </property>
          </widget>
         </item>
         <item row="0" column="0">
          <widget class="QPushButton" name="pushButton">
           <property name="styleSheet">
            <string notr="true">background-color: rgb(0, 255, 0);</string>
           </property>
           <property name="text">
            <string>Start Thread 1</string>
           </property>
          </widget>
         </item>
         <item row="0" column="3" rowspan="3">
          <layout class="QVBoxLayout" name="verticalLayout"/>
         </item>
         <item row="1" column="0">
          <widget class="QPushButton" name="pushButton_2">
           <property name="styleSheet">
            <string notr="true">background-color: rgb(85, 255, 0);</string>
           </property>
           <property name="text">
            <string>Start Thread 2</string>
           </property>
          </widget>
         </item>
         <item row="2" column="0">
          <widget class="QPushButton" name="pushButton_3">
           <property name="styleSheet">
            <string notr="true">background-color: rgb(85, 255, 0);</string>
           </property>
           <property name="text">
            <string>Start Thread 3</string>
           </property>
          </widget>
         </item>
         <item row="2" column="2">
          <widget class="QProgressBar" name="progressBar_3">
           <property name="value">
            <number>24</number>
           </property>
          </widget>
         </item>
         <item row="1" column="2">
          <widget class="QProgressBar" name="progressBar_2">
           <property name="value">
            <number>24</number>
           </property>
          </widget>
         </item>
         <item row="0" column="1">
          <widget class="QPushButton" name="pushButton_4">
           <property name="styleSheet">
            <string notr="true">background-color: rgb(255, 85, 127);</string>
           </property>
           <property name="text">
            <string>Stop Thread 1</string>
           </property>
          </widget>
         </item>
         <item row="1" column="1">
          <widget class="QPushButton" name="pushButton_5">
           <property name="styleSheet">
            <string notr="true">background-color: rgb(255, 85, 127);</string>
           </property>
           <property name="text">
            <string>Stop Thread 2</string>
           </property>
          </widget>
         </item>
         <item row="2" column="1">
          <widget class="QPushButton" name="pushButton_6">
           <property name="styleSheet">
            <string notr="true">background-color: rgb(255, 85, 127);</string>
           </property>
           <property name="text">
            <string>Stop Thread 3</string>
           </property>
          </widget>
         </item>
        </layout>
       </item>
      </layout>
     </widget>
    </item>
   </layout>
  </widget>
  <widget class="QStatusBar" name="statusbar"/>
 </widget>
 <resources/>
 <connections>
  <connection>
   <sender>pushButton</sender>
   <signal>clicked()</signal>
   <receiver>progressBar</receiver>
   <slot>reset()</slot>
   <hints>
    <hint type="sourcelabel">
     <x>42</x>
     <y>52</y>
    </hint>
    <hint type="destinationlabel">
     <x>156</x>
     <y>56</y>
    </hint>
   </hints>
  </connection>
  <connection>
   <sender>pushButton_2</sender>
   <signal>clicked()</signal>
   <receiver>progressBar_2</receiver>
   <slot>reset()</slot>
   <hints>
    <hint type="sourcelabel">
     <x>52</x>
     <y>130</y>
    </hint>
    <hint type="destinationlabel">
     <x>97</x>
     <y>129</y>
    </hint>
   </hints>
  </connection>
  <connection>
   <sender>pushButton_3</sender>
   <signal>clicked()</signal>
   <receiver>progressBar_3</receiver>
   <slot>reset()</slot>
   <hints>
    <hint type="sourcelabel">
     <x>66</x>
     <y>192</y>
    </hint>
    <hint type="destinationlabel">
     <x>130</x>
     <y>192</y>
    </hint>
   </hints>
  </connection>
 </connections>
</ui>

```

## main.py

```python

## Welcome to PyShine
## This is part 12 of the PyQt5 learning series
## Start and Stop Qthreads
## Source code available: www.pyshine.com
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5 import uic
import sys, time

class PyShine_THREADS_APP(QtWidgets.QMainWindow):
	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		self.ui = uic.loadUi('threads.ui',self)
		self.resize(888, 200)
		icon = QtGui.QIcon()
		icon.addPixmap(QtGui.QPixmap("PyShine.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
		self.setWindowIcon(icon)
		
		self.thread={}
		self.pushButton.clicked.connect(self.start_worker_1)
		self.pushButton_2.clicked.connect(self.start_worker_2)
		self.pushButton_3.clicked.connect(self.start_worker_3)
		self.pushButton_4.clicked.connect(self.stop_worker_1)
		self.pushButton_5.clicked.connect(self.stop_worker_2)
		self.pushButton_6.clicked.connect(self.stop_worker_3)


	def start_worker_1(self):
		self.thread[1] = ThreadClass(parent=None,index=1)
		self.thread[1].start()
		self.thread[1].any_signal.connect(self.my_function)
		self.pushButton.setEnabled(False)
		
	def start_worker_2(self):
		self.thread[2] = ThreadClass(parent=None,index=2)
		self.thread[2].start()
		self.thread[2].any_signal.connect(self.my_function)
		self.pushButton_2.setEnabled(False)

	def start_worker_3(self):
		self.thread[3] = ThreadClass(parent=None,index=3)
		self.thread[3].start()
		self.thread[3].any_signal.connect(self.my_function)
		self.pushButton_3.setEnabled(False)
		
	def stop_worker_1(self):
		self.thread[1].stop()
		self.pushButton.setEnabled(True)

	def stop_worker_2(self):
		self.thread[2].stop()
		self.pushButton_2.setEnabled(True)

	def stop_worker_3(self):
		self.thread[3].stop()
		self.pushButton_3.setEnabled(True)

	def my_function(self,counter):
		
		cnt=counter
		index = self.sender().index
		if index==1:
			self.progressBar.setValue(cnt) 
		if index==2: 
			self.progressBar_2.setValue(cnt)  
		if index==3:
			self.progressBar_3.setValue(cnt)  
	
class ThreadClass(QtCore.QThread):
	
	any_signal = QtCore.pyqtSignal(int)
	def __init__(self, parent=None,index=0):
		super(ThreadClass, self).__init__(parent)
		self.index=index
		self.is_running = True
	def run(self):
		print('Starting thread...',self.index)
		cnt=0
		while (True):
			cnt+=1
			if cnt==99: cnt=0
			time.sleep(0.01)
			self.any_signal.emit(cnt) 
	def stop(self):
		self.is_running = False
		print('Stopping thread...',self.index)
		self.terminate()



app = QtWidgets.QApplication(sys.argv)
mainWindow = PyShine_THREADS_APP()
mainWindow.show()
sys.exit(app.exec_())

```

