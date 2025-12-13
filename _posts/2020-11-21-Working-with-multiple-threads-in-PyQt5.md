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

## Introduction

Hello friends!

In this tutorial, we will design a **simple but powerful PyQt5 GUI** that demonstrates how to work with **multiple threads using `QThread`**. The application contains **six buttons**:

<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/k5tIk7w50L4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>

- Three buttons to **start** three different threads
- Three buttons to **stop** those threads

Each thread updates its own **progress bar**, making it easy to visually understand how threads run independently without freezing the GUI.

This example is intentionally kept simple so beginners can clearly understand:

- Why threads are needed in GUI applications
- How `QThread` works
- How to safely update the GUI from a background thread

---

## Why Use Threads in PyQt5?

In GUI applications, **long-running tasks must never run in the main thread**, otherwise:

- The GUI freezes
- Buttons stop responding
- The window becomes unresponsive

`QThread` allows us to:

- Run tasks in the background
- Keep the GUI responsive
- Communicate safely with the UI using **signals and slots**

---

## Project Structure

Create a new folder and place the following files inside:

```
project_folder/
│── main.py
│── threads.ui
│── PyShine.png   # optional window icon
```

Run the application using:

```bash
python main.py
```

---

## threads.ui (Qt Designer File)

This UI file defines:

- 3 Start buttons
- 3 Stop buttons
- 3 Progress bars

Each progress bar represents a running thread.


## threads.ui

```xml
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




## main.py (Thread Logic)

This file contains:

- A main window class
- A custom `QThread` subclass
- Signal-slot communication between threads and UI

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

---

## Key Concepts Explained

### 1. QThread
Runs code in the background without blocking the GUI.

### 2. Signals and Slots
Threads **never update UI directly**. They emit signals, and the main thread updates widgets safely.

### 3. Multiple Threads
Each progress bar is controlled by a separate thread, running independently.

---

## Common Beginner Mistakes

❌ Updating UI directly inside `run()`

❌ Using `time.sleep()` in the main thread

❌ Not stopping threads properly

---

## Conclusion

This example demonstrates how to:

- Use multiple QThreads
- Start and stop threads safely
- Keep the GUI responsive

This pattern is essential for **real-world PyQt applications** like:

- Video processing
- File downloads
- Data analysis
- Hardware communication

Happy coding!





