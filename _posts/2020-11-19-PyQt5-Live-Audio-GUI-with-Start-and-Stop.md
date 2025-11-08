---
categories:
- GUI tutorial series
description: How to stop a thread in managed by the QThreadPool in PyQt5
featured-img: audiolive2
keywords:
- PyQt5
- Live Audio
- QThread
- GUI
- Audio Stream
- Thread Management
layout: post
mathjax: true
tags:
- PyQt5
- Audio
- QThread
- GUI
- Thread Management
title: PytQt5 Live Audio GUI with start and stop buttons
---


<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/5vbIMWwWU5A" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>
Hello friends, this tutorial is an update of the previous tutorial No. 10. Like before, we will get data from the live audio stream through the computer's microphone. We have another button named "Stop" added to this GUI. This button will stop the current worker and let the user update the parameters in the GUI. The GUI remains responsive and acts according to the input parameters. In the upcoming tutorial, we will use a different approach to run several threads. The .ui and .py files are available below. Please make a new folder and save both of them as separate files.

Tested on Mac OS and Windows.

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
    <width>888</width>
    <height>470</height>
   </rect>
  </property>
  <property name="windowTitle">
   <string>MainWindow</string>
  </property>
  <widget class="QWidget" name="centralwidget">
   <layout class="QGridLayout" name="gridLayout_5">
    <item row="0" column="0">
     <layout class="QGridLayout" name="gridLayout_4">
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
              <string>Plot It!</string>
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
      <item row="2" column="1">
       <widget class="QWidget" name="widget" native="true">
        <property name="styleSheet">
         <string notr="true">background-color: rgb(0, 0, 0);</string>
        </property>
       </widget>
      </item>
      <item row="0" column="1">
       <widget class="QLabel" name="label">
        <property name="text">
         <string>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;&lt;span style=&quot; font-size:14pt; font-weight:600;&quot;&gt;PyShine Live Voice Plot GUI&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</string>
        </property>
        <property name="alignment">
         <set>Qt::AlignCenter</set>
        </property>
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
     </layout>
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
## This is part 11 of the PyQt5 learning series
## Based on parameters, the GUI will plot live voice data using Matplotlib in PyQt5
## We will use Qthreads to run the audio stream data.

import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import queue
import numpy as np
import sounddevice as sd

from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtMultimedia import QAudioDeviceInfo,QAudio,QCameraInfo
import time
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
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("PyShine.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.threadpool = QtCore.QThreadPool()	
        self.threadpool.setMaxThreadCount(1)
        self.devices_list= []
        for device in input_audio_deviceInfos:
            self.devices_list.append(device.deviceName())
        
        self.comboBox.addItems(self.devices_list)
        self.comboBox.currentIndexChanged['QString'].connect(self.update_now)
        self.comboBox.setCurrentIndex(0)
        
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.ui.gridLayout_4.addWidget(self.canvas, 2, 1, 1, 1)
        self.reference_plot = None
        self.q = queue.Queue(maxsize=20)

        self.device = self.devices_list[0]
        self.window_length = 1000
        self.downsample = 1
        self.channels = [1]
        self.interval = 30 
        
        ## device_info =  sd.query_devices(self.device, 'input')
        
        ## self.samplerate = device_info['default_samplerate']
        self.samplerate = 44100
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
        try:
            QtWidgets.QApplication.processEvents()	
            def audio_callback(indata,frames,time,status):
                self.q.put(indata[::self.downsample,[0]])
            stream  = sd.InputStream( device = self.device, channels = max(self.channels), samplerate =self.samplerate, callback  = audio_callback)
            with stream:
                
                while True:
                    QtWidgets.QApplication.processEvents()
                    if self.go_on:
                        
                        break
                        
                
            self.pushButton.setEnabled(True)
            self.lineEdit.setEnabled(True)
            self.lineEdit_2.setEnabled(True)
            self.lineEdit_3.setEnabled(True)
            self.lineEdit_4.setEnabled(True)
            self.comboBox.setEnabled(True)	
            
        except Exception as e:
            print("ERROR: ",e)
            pass

    def start_worker(self):
        
        self.lineEdit.setEnabled(False)
        self.lineEdit_2.setEnabled(False)
        self.lineEdit_3.setEnabled(False)
        self.lineEdit_4.setEnabled(False)
        self.comboBox.setEnabled(False)
        self.pushButton.setEnabled(False)
        self.canvas.axes.clear()
        
        self.go_on = False
        self.worker = Worker(self.start_stream, )
        self.threadpool.start(self.worker)	
        self.reference_plot = None
        self.timer.setInterval(self.interval) #msec
        


    def stop_worker(self):
        
        self.go_on=True
        with self.q.mutex:
            self.q.queue.clear()
        
        #self.timer.stop()
        
        
        
    def start_stream(self):
        
        self.getAudio()
        
        
        
    def update_now(self,value):
        print(value)
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
                self.plotdata[-shift:,:] = self.data
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
            self.canvas.axes.set_ylim( ymin=-0.5, ymax=0.5)		

            self.canvas.draw()
        except Exception as e:
            print("Error:",e)
            pass


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




