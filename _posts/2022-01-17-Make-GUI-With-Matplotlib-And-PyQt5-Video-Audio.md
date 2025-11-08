---
categories:
- GUI tutorial series
description: How to plot audio and video from opencv matplotlib and PyQt5
featured-img: 2022-01-17-plot-vi
keywords:
- PyQt5 video GUI
- PyQt5 audio plot
- OpenCV video display
- Matplotlib audio visualization
- Python GUI tutorial
- Start and stop buttons PyQt5
- Live audio plotting PyQt5
- Video playback PyQt5
layout: post
mathjax: true
tags:
- PyQt5
- GUI
- Audio
- Video
- OpenCV
- Matplotlib
- Tutorial
- Python
title: PytQt5 Video and Audio GUI with start and stop buttons
---



Hello friends, this tutorial is about displaying mp4 video and its audio plot on the same GUI. This is part 16 of the PyQt5 learning series. More details about making the
backbone-gui is available here https://pyshine.com/PyQt5-Live-Audio-GUI-with-Start-and-Stop/. Below are two files: main.ui and gui.py. Put both in same directory and 
run `python3 gui.py`

[![GIF](https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/posts/test.gif?raw=true)](https://youtu.be/5vbIMWwWU5A "GIF")

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
    <width>1010</width>
    <height>1006</height>
   </rect>
  </property>
  <property name="windowTitle">
   <string>PyShine Video with Audio Plot GUI</string>
  </property>
  <widget class="QWidget" name="centralwidget">
   <layout class="QGridLayout" name="gridLayout_5">
    <item row="1" column="0">
     <layout class="QGridLayout" name="gridLayout_4">
      <item row="2" column="1">
       <widget class="QWidget" name="widget" native="true">
        <property name="minimumSize">
         <size>
          <width>320</width>
          <height>240</height>
         </size>
        </property>
        <property name="mouseTracking">
         <bool>true</bool>
        </property>
        <property name="styleSheet">
         <string notr="true">background-color: rgb(0, 0, 0);</string>
        </property>
       </widget>
      </item>
      <item row="1" column="1">
       <widget class="QGroupBox" name="groupBox">
        <property name="title">
         <string>Parameters</string>
        </property>
        <layout class="QGridLayout" name="gridLayout_3">
         <item row="0" column="0">
          <layout class="QGridLayout" name="gridLayout_2">
           <item row="0" column="0">
            <widget class="QLabel" name="label_2">
             <property name="text">
              <string>Audio Device</string>
             </property>
            </widget>
           </item>
           <item row="0" column="1">
            <widget class="QComboBox" name="comboBox"/>
           </item>
           <item row="1" column="0">
            <widget class="QLabel" name="label_3">
             <property name="text">
              <string>Window Length (&gt;28)</string>
             </property>
            </widget>
           </item>
           <item row="1" column="1">
            <widget class="QLineEdit" name="lineEdit">
             <property name="text">
              <string>1000</string>
             </property>
            </widget>
           </item>
           <item row="2" column="0">
            <widget class="QLabel" name="label_4">
             <property name="text">
              <string>Sampling Rate (&gt;1000 Hz)</string>
             </property>
            </widget>
           </item>
           <item row="2" column="1">
            <widget class="QLineEdit" name="lineEdit_2">
             <property name="text">
              <string>44100</string>
             </property>
            </widget>
           </item>
          </layout>
         </item>
         <item row="0" column="1">
          <layout class="QVBoxLayout" name="verticalLayout">
           <item>
            <layout class="QGridLayout" name="gridLayout">
             <item row="0" column="0">
              <widget class="QLabel" name="label_5">
               <property name="text">
                <string>Down Sample (&gt;0)</string>
               </property>
              </widget>
             </item>
             <item row="0" column="1">
              <widget class="QLineEdit" name="lineEdit_3">
               <property name="text">
                <string>1</string>
               </property>
              </widget>
             </item>
             <item row="1" column="0">
              <widget class="QLabel" name="label_6">
               <property name="text">
                <string>Update Interval (1 to 100 ms)</string>
               </property>
              </widget>
             </item>
             <item row="1" column="1">
              <widget class="QLineEdit" name="lineEdit_4">
               <property name="text">
                <string>30</string>
               </property>
              </widget>
             </item>
            </layout>
           </item>
           <item>
            <widget class="QPushButton" name="pushButton">
             <property name="text">
              <string>Select and Play MP4 Video</string>
             </property>
            </widget>
           </item>
           <item>
            <widget class="QPushButton" name="pushButton_2">
             <property name="text">
              <string>Stop</string>
             </property>
            </widget>
           </item>
          </layout>
         </item>
        </layout>
       </widget>
      </item>
      <item row="3" column="1">
       <spacer name="horizontalSpacer">
        <property name="orientation">
         <enum>Qt::Horizontal</enum>
        </property>
        <property name="sizeHint" stdset="0">
         <size>
          <width>778</width>
          <height>0</height>
         </size>
        </property>
       </spacer>
      </item>
      <item row="2" column="0">
       <spacer name="verticalSpacer">
        <property name="orientation">
         <enum>Qt::Vertical</enum>
        </property>
        <property name="sizeHint" stdset="0">
         <size>
          <width>0</width>
          <height>40</height>
         </size>
        </property>
       </spacer>
      </item>
      <item row="0" column="1">
       <widget class="QLabel" name="label">
        <property name="text">
         <string/>
        </property>
        <property name="pixmap">
         <pixmap>Snapshot 2022-Jan-15 at 11.55.30 AM.png</pixmap>
        </property>
        <property name="scaledContents">
         <bool>false</bool>
        </property>
        <property name="alignment">
         <set>Qt::AlignCenter</set>
        </property>
        <property name="margin">
         <number>4</number>
        </property>
        <property name="indent">
         <number>3</number>
        </property>
       </widget>
      </item>
     </layout>
    </item>
    <item row="0" column="0">
     <spacer name="verticalSpacer_2">
      <property name="orientation">
       <enum>Qt::Vertical</enum>
      </property>
      <property name="sizeHint" stdset="0">
       <size>
        <width>20</width>
        <height>40</height>
       </size>
      </property>
     </spacer>
    </item>
   </layout>
  </widget>
  <widget class="QStatusBar" name="statusbar"/>
 </widget>
 <resources/>
 <connections/>
</ui>


```

(Updated) Added QtWidgets.QApplication.processEvents(), runs well for Windows and Mac Os, please share your experiences below.

## gui.py  
{% include codeHeader.html %}
```python
## Welcome to PyShine
## This is part 16 of the PyQt5 learning series
## Based on parameters, the GUI will plot Video using OpenCV and Audio using Matplotlib in PyQt5
## We will use Qthreads to run the audio/Video streams

import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import queue
import numpy as np
import sounddevice as sd
from PyQt5.QtGui import QImage
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtMultimedia import QAudioDeviceInfo,QAudio,QCameraInfo
import time
import queue
import os
import wave, pyaudio, pdb
import cv2,imutils
from PyQt5.QtWidgets import QFileDialog
## For details visit pyshine.com




input_audio_deviceInfos = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)

class MplCanvas(FigureCanvas):
	def __init__(self, parent=None, width=5, height=4, dpi=100):
		fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = fig.add_subplot(111)
		super(MplCanvas, self).__init__(fig)
		fig.tight_layout()

class PyShine_LIVE_PLOT_APP(QtWidgets.QMainWindow):
	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		self.ui = uic.loadUi('main.ui',self)
		self.resize(888, 600)
		self.tmpfile = 'temp.wav'

		icon = QtGui.QIcon()
		icon.addPixmap(QtGui.QPixmap("PyShine.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
		self.setWindowIcon(icon)
		self.threadpool = QtCore.QThreadPool()	
		self.threadpool.setMaxThreadCount(2)
		self.CHUNK = 1024
		self.q = queue.Queue(maxsize=self.CHUNK)
		self.devices_list= []
		for device in input_audio_deviceInfos:
			self.devices_list.append(device.deviceName())
		
		self.comboBox.addItems(self.devices_list)
		self.comboBox.currentIndexChanged['QString'].connect(self.update_now)
		self.comboBox.setCurrentIndex(0)
		
		self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
		self.ui.gridLayout_4.addWidget(self.canvas, 2, 1, 1, 1)
		self.reference_plot = None
		

		self.device = self.devices_list[0]
		self.window_length = 1000
		self.downsample = 1
		self.channels = [1]
		self.interval = 1

		
		
		
		
		device_info =  sd.query_devices(self.device, 'input')
		
		self.samplerate = device_info['default_samplerate']
		length  = int(self.window_length*self.samplerate/(1000*self.downsample))
		sd.default.samplerate = self.samplerate
		
		self.plotdata =  np.zeros((length,len(self.channels)))
		self.timer = QtCore.QTimer()
		self.timer.setInterval(self.interval) #msec
		self.timer.timeout.connect(self.update_plot)
		self.timer.start()
		self.data=[0]
		self.lineEdit.textChanged['QString'].connect(self.update_window_length)
		self.lineEdit_2.textChanged['QString'].connect(self.update_sample_rate)
		self.lineEdit_3.textChanged['QString'].connect(self.update_down_sample)
		self.lineEdit_4.textChanged['QString'].connect(self.update_interval)
		self.pushButton.clicked.connect(self.start_worker)
		self.pushButton_2.clicked.connect(self.stop_worker)
		self.worker = None
		self.go_on = False
		
	
	def getAudio(self):

		QtWidgets.QApplication.processEvents()	
		CHUNK = self.CHUNK
		
		wf = wave.open(self.tmpfile, 'rb')
		p = pyaudio.PyAudio()
		stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
		                channels=wf.getnchannels(),
		                rate=wf.getframerate(),
		                output=True,
		                frames_per_buffer=CHUNK)
		self.samplerate = wf.getframerate()
		sd.default.samplerate = self.samplerate
		while(self.vid.isOpened()):
		    
			QtWidgets.QApplication.processEvents()    
			data = wf.readframes(CHUNK)
			audio_as_np_int16 = np.frombuffer(data, dtype=np.int16)
			audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
##		# Normalise float32 array                                                   
			max_int16 = 2**15
			audio_normalised = audio_as_np_float32 / max_int16
			
			self.q.put_nowait(audio_normalised)
			stream.write(data)
			
			if self.go_on:
				break
			
		self.pushButton.setEnabled(True)
		self.lineEdit.setEnabled(True)
		self.lineEdit_2.setEnabled(True)
		self.lineEdit_3.setEnabled(True)
		self.lineEdit_4.setEnabled(True)
		self.comboBox.setEnabled(True)	
			

	def start_worker(self):
		
		self.lineEdit.setEnabled(False)
		self.lineEdit_2.setEnabled(False)
		self.lineEdit_3.setEnabled(False)
		self.lineEdit_4.setEnabled(False)
		self.comboBox.setEnabled(False)
		self.pushButton.setEnabled(False)
		self.canvas.axes.clear()
		self.loadVideoPath()
		self.go_on = False
		self.vworker = Worker(self.start_vstream)
		
		self.worker = Worker(self.start_stream, )
		self.threadpool.start(self.vworker)	
		self.threadpool.start(self.worker)	

		self.reference_plot = None
		self.timer.setInterval(self.interval) #msec
		
	
	def stop_worker(self):
		
		self.go_on=True
		with self.q.mutex:
			self.q.queue.clear()
		
		
	def start_stream(self):
		self.getAudio()
		
		
	def start_vstream(self):
		self.loadImage()
		
		
	def update_now(self,value):
		self.device = self.devices_list.index(value)
		

	def update_window_length(self,value):
		self.window_length = int(value)
		length  = int(self.window_length*self.samplerate/(1000*self.downsample))
		self.plotdata =  np.zeros((length,len(self.channels)))
		

	def update_sample_rate(self,value):
		self.samplerate = int(value)
		sd.default.samplerate = self.samplerate
		length  = int(self.window_length*self.samplerate/(1000*self.downsample))
		self.plotdata =  np.zeros((length,len(self.channels)))
		
	
	def update_down_sample(self,value):
		self.downsample = int(value)
		length  = int(self.window_length*self.samplerate/(1000*self.downsample))
		self.plotdata =  np.zeros((length,len(self.channels)))
	

	def update_interval(self,value):
		self.interval = int(value)
		
		

	def update_plot(self):
		try:
			
			
			print('ACTIVE THREADS:',self.threadpool.activeThreadCount(),end=" \r")
			while  self.go_on is False:
				QtWidgets.QApplication.processEvents()	
				try: 
					self.data = self.q.get_nowait()
					
					
				except queue.Empty:
					break
				
				shift = len(self.data)
				self.plotdata = np.roll(self.plotdata, -shift,axis = 0)
				self.plotdata = self.data
				self.ydata = self.plotdata[:]
				self.canvas.axes.set_facecolor((0,0,0))
				
	  
				if self.reference_plot is None:
					plot_refs = self.canvas.axes.plot( self.ydata, color=(0,1,0.29))
					self.reference_plot = plot_refs[0]	
				else:
					self.reference_plot.set_ydata(self.ydata)
					

			
			self.canvas.axes.yaxis.grid(True,linestyle='--')
			start, end = self.canvas.axes.get_ylim()
			self.canvas.axes.yaxis.set_ticks(np.arange(start, end, 0.1))
			self.canvas.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
			self.canvas.axes.set_ylim( ymin=-1, ymax=1)		

			self.canvas.draw()
		except Exception as e:
			
			pass

	def setPhoto(self,image):
		""" This function will take image input and resize it 
			only for display purpose and convert it to QImage
			to set at the label.
		"""
		self.tmp = image
		
		frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
		self.ui.label.setPixmap(QtGui.QPixmap.fromImage(image))
		
		

	def update(self):
		""" This function will update the photo according to the 
			current values of blur and brightness and set it to photo label.
		"""
		self.setPhoto(self.image)	

	def loadVideoPath(self):
		""" This function will load the user selected video
			and set it to label using the setPhoto function
		"""
		try:
			os.remove(self.tmpfile)
		except:
			pass
		self.filename = QFileDialog.getOpenFileName(filter="Video (*.*)")[0]
		command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(self.filename,self.tmpfile)
		os.system(command)
		self.vid = cv2.VideoCapture(self.filename) # place path to your video file here
		
	
	def loadImage(self):
		""" This function will load the camera device, obtain the image
			and set it to label using the setPhoto function
		"""
		

		
		FPS = self.vid.get(cv2.CAP_PROP_FPS)

		TS = (1/FPS)
		BREAK=False
		fps,st,frames_to_count,cnt = (0,0,10,0)

		while(self.vid.isOpened()):
			QtWidgets.QApplication.processEvents()	
			img, self.image = self.vid.read()
			try:
				self.image  = imutils.resize(self.image ,width = 640 )
				
				if cnt == frames_to_count:
					try:
						fps = (frames_to_count/(time.time()-st))
						
						st=time.time()
						cnt=0
						if fps>FPS:
							TS+=0.001
						elif fps<FPS:
							TS-=0.001
						else:
							pass
					except:
						pass
				cnt+=1
				
				self.update()
				time.sleep(TS)
				if self.go_on:
					self.vid = cv2.VideoCapture(self.filename) # place path to your video file here	

					break
			except:
				self.stop_worker()
				break

## www.pyshine.com
class Worker(QtCore.QRunnable):

	def __init__(self, function, *args, **kwargs):
		super(Worker, self).__init__()
		self.function = function
		self.args = args
		self.kwargs = kwargs

	@pyqtSlot()
	def run(self):

		self.function(*self.args, **self.kwargs)			
	


app = QtWidgets.QApplication(sys.argv)
mainWindow = PyShine_LIVE_PLOT_APP()
mainWindow.show()
sys.exit(app.exec_())

```




