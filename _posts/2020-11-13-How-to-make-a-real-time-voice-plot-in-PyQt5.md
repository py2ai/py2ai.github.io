---
layout: post
title: PytQt5 GUI design to plot Live audio data from Microphone
categories: [GUI tutorial series]
mathjax: true
featured-img: audiolive
summary: How to make a GUI using PyQt5 and Matplotlib to plot real-time data
---

<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/Ng00Mj5Tt8o" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>

Hello friends, today we will design a simple but beneficial GUI in PyQt5. We will plot live audio data that is sampled from your computer's audio device.
We will use the Matplotlib figure and update the canvas according to the set interval in the GUI. This kind of application is useful in your projects; for example, 
you can plot temperature, counts of vehicles, peoples, and much more. The basic idea is to use the QTimer and QThread to run the Audio stream, put the stream data to a queue, and get the queue data to display it to the Matplotlib canvas. The functionality of the different modules used in the code below are explained
in the video above.

### Installation of sounddevice

Installing sounddevice in Linux/Ubuntu requires following two steps:

```
sudo apt-get install libportaudio2
pip3 install sounddevice
```
On Windows and Mac OS

```
pip3 install sounddevice
```

Other included libraries can also be installed using pip3 install.

Alright, to run the GUI, we actually require two files:

1) gui.py
2) main.ui

Please copy the code and paste in the empty Python files named as main.ui and gui.py. It is highly recommended to run this code (gui.py) using Power Shell, IDLE or terminal in
the MAC OS. Put both these files in the same directory and from that driectory run the code.

### Run the code as:
```python

python3 gui.py

```
Here is the main.ui file, you can open it in the PyQt5 designer as a main.ui file. To save time, you can add more changes to it according to your requirements.

### main.ui
{% include codeHeader.html %}
```xml
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
              <string>Window Length</string>
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
              <string>Sampling Rate (Hz)</string>
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
                <string>Down Sample</string>
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
                <string>Update Interval (ms)</string>
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



### gui.py
{% include codeHeader.html %}
```python
# Welcome to PyShine
# This is part 10 of the PyQt5 learning series
# Based on parameters, the GUI will plot live voice data using Matplotlib in PyQt5
# We will use Qthreads to run the audio stream data.

import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import queue
import numpy as np
import sounddevice as sd
import pdb
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtMultimedia import QAudioDeviceInfo,QAudio,QCameraInfo

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
        
        # device_info =  sd.query_devices(self.device, 'input')
        # self.samplerate = device_info['default_samplerate']

        self.samplerate = 44100
        length  = int(self.window_length*self.samplerate/(1000*self.downsample))
        sd.default.samplerate = self.samplerate
        
        self.plotdata =  np.zeros((length,len(self.channels)))
        
        self.update_plot()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.interval) #msec
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        self.lineEdit.textChanged['QString'].connect(self.update_window_length)
        self.lineEdit_2.textChanged['QString'].connect(self.update_sample_rate)
        self.lineEdit_3.textChanged['QString'].connect(self.update_down_sample)
        self.lineEdit_4.textChanged['QString'].connect(self.update_interval)
        self.pushButton.clicked.connect(self.start_worker)
        #self.update_now(self.device)
        
    def getAudio(self):
        try:
            def audio_callback(indata,frames,time,status):
                self.q.put(indata[::self.downsample,[0]])
            stream  = sd.InputStream( device = self.device, channels = max(self.channels), samplerate =self.samplerate, callback  = audio_callback)
            with stream:
                input()
        except Exception as e:
            print("ERROR: ",e)

    def start_worker(self):
        worker = Worker(self.start_stream, )
        self.threadpool.start(worker)	

    def start_stream(self):
        self.lineEdit.setEnabled(False)
        self.lineEdit_2.setEnabled(False)
        self.lineEdit_3.setEnabled(False)
        self.lineEdit_4.setEnabled(False)
        self.comboBox.setEnabled(False)
        self.pushButton.setEnabled(False)
        self.getAudio()
        
    def update_now(self,value):
        print(value)
        self.device = self.devices_list.index(value)
        print('Device:',self.devices_list.index(value))

    def update_window_length(self,value):
        self.window_length = int(value)
        length  = int(self.window_length*self.samplerate/(1000*self.downsample))
        self.plotdata =  np.zeros((length,len(self.channels)))
        self.update_plot()

    def update_sample_rate(self,value):
        self.samplerate = int(value)
        sd.default.samplerate = self.samplerate
        length  = int(self.window_length*self.samplerate/(1000*self.downsample))
        self.plotdata =  np.zeros((length,len(self.channels)))
        self.update_plot()

    def update_down_sample(self,value):
        self.downsample = int(value)
        length  = int(self.window_length*self.samplerate/(1000*self.downsample))
        self.plotdata =  np.zeros((length,len(self.channels)))
        self.update_plot()

    def update_interval(self,value):
        self.interval = int(value)
        self.timer.setInterval(self.interval) #msec
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        try:
            data=[0]
            
            while True:
                try: 
                    data = self.q.get_nowait()
                except queue.Empty:
                    break
                shift = len(data)
                self.plotdata = np.roll(self.plotdata, -shift,axis = 0)
                self.plotdata[-shift:,:] = data
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
        except:
            pass


# www.pyshine.com
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

Please do comment, and give your valuable suggestions. Have a nice day!
