---
layout: post
title: How to make a Matplotlib and PyQt5 based GUI to plot a CSV file data
categories: [GUI tutorial series]
mathjax: true
featured-img: pyqt15datatime
summary: Making a drag drop CSV file based matplotlib GUI with multiple themes and adding axis selection options
---


Hello friends! This is part 15 of the PyQt5 GUI learning series. Recently, some friends have given interesting suggestions related to part 07. In
that tutorial we designed a GUI to drag and drop the csv file to show data. Today's tutorial is an extension to the previous one. So lets have a look at the 
suggestions:

 1. It would be better to let user select x-axis and y-axis data from a dropdown list.
 2. Some CSV files especially time-series data have datetime stamps, which should be handled on the x-axis.

Based on these fantastic suggestions we have made two changes to the previous part 07 gui of the PyQt5 learning series. Thanks to the modular approach, all changes were made 
to the main.py and the drag_drop.py file remains intact.

<p align="center">
  <img src="https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/posts/pyqt515gui.gif" alt="animated" />
</p>

Enjoy, and do give your feedback and suggestions. Also please make sure that you are using Matplotlib version 3.2.1 or above. For more detail visit pyshine youtube channel.
Below is the tutorial of previous part 07 (in case you have missed it):
<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/1q3Z2clyIEQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>

Let's have a look at the project structure:

```
15-Drag Drop CSV matplotlib GUI with selection/
├── main.py
├── drag_drop.py
├── performance_metrics.csv
├── time_series-data_2.csv
└── time_series_data_1.csv

```
In a project directory (in our case ```15-Drag Drop CSV matplotlib GUI with selection```) put these files.



## main.py

```python

# -*- coding: utf-8 -*-
# Subscribe to PyShine Youtube channel for more detail! 
#
# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!
#
# WEBSITE: www.pyshine.com
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as Navi
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
import sip # can be installed : pip install sip
from datetime import datetime
# We require a canvas class


class MatplotlibCanvas(FigureCanvasQTAgg):
	def __init__(self,parent=None, dpi = 120):
		fig = Figure(dpi = dpi)
		self.axes = fig.add_subplot(111)
		super(MatplotlibCanvas,self).__init__(fig)
		fig.tight_layout()
		
		

class Ui_MainWindow(object):
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(1440, 1000)
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
		self.gridLayout.setObjectName("gridLayout")
		self.horizontalLayout = QtWidgets.QHBoxLayout()
		self.horizontalLayout.setObjectName("horizontalLayout")
		
		self.label = QtWidgets.QLabel(self.centralwidget)
		self.label.setObjectName("label")

		self.label_1 = QtWidgets.QLabel(self.centralwidget)
		self.label_1.setObjectName("label_1")
		
		self.label_2 = QtWidgets.QLabel(self.centralwidget)
		self.label_2.setObjectName("label_2")
	
		self.horizontalLayout.addWidget(self.label)

		self.comboBox = QtWidgets.QComboBox(self.centralwidget)
		self.comboBox.setObjectName("comboBox")
		self.horizontalLayout.addWidget(self.comboBox)
		
		self.comboBox_1 = QtWidgets.QComboBox(self.centralwidget)
		self.comboBox_1.setObjectName("comboBox_1")
		

		self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
		self.comboBox_2.setObjectName("comboBox_2")
		

		self.pushButton = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton.setObjectName("pushButton")
		self.horizontalLayout.addWidget(self.pushButton)
		spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
		self.horizontalLayout.addItem(spacerItem)
		self.horizontalLayout.addWidget(self.label_1)
		self.horizontalLayout.addWidget(self.comboBox_1)
		self.horizontalLayout.addWidget(self.label_2)
		self.horizontalLayout.addWidget(self.comboBox_2)	
		self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
		self.verticalLayout = QtWidgets.QVBoxLayout()
		self.verticalLayout.setObjectName("verticalLayout")
		self.spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
		self.verticalLayout.addItem(self.spacerItem1)

		self.gridLayout.addLayout(self.verticalLayout, 1, 0, 1, 1)	
		MainWindow.setCentralWidget(self.centralwidget)
		self.menubar = QtWidgets.QMenuBar(MainWindow)
		self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
		self.menubar.setObjectName("menubar")
		self.menuFile = QtWidgets.QMenu(self.menubar)
		self.menuFile.setObjectName("menuFile")
		MainWindow.setMenuBar(self.menubar)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setObjectName("statusbar")
		MainWindow.setStatusBar(self.statusbar)
		self.actionOpen_csv_file = QtWidgets.QAction(MainWindow)
		self.actionOpen_csv_file.setObjectName("actionOpen_csv_file")
		self.actionExit = QtWidgets.QAction(MainWindow)
		self.actionExit.setObjectName("actionExit")
		self.menuFile.addAction(self.actionOpen_csv_file)
		self.menuFile.addAction(self.actionExit)
		self.menubar.addAction(self.menuFile.menuAction())

		self.retranslateUi(MainWindow)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)
		
		self.filename = ''
		self.canv = MatplotlibCanvas(self)
		self.df = []
		
		self.toolbar = Navi(self.canv,self.centralwidget)
		self.horizontalLayout.addWidget(self.toolbar)
		
		self.themes = ['bmh', 'classic', 'dark_background', 'fast', 
		'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright',
		 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 
		 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook',
		 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk',
		 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn',
		 'Solarize_Light2', 'tableau-colorblind10']
		 
		self.comboBox.addItems(self.themes)
		self.comboBox_1.addItems(['Select horizontal axis here'])
		self.comboBox_2.addItems(['Select vertical axis here'])
		
		self.pushButton.clicked.connect(self.getFile)
		self.comboBox.currentIndexChanged['QString'].connect(self.Update)
		self.comboBox_1.currentIndexChanged['QString'].connect(self.selectXaxis)
		self.comboBox_2.currentIndexChanged['QString'].connect(self.selectYaxis)
		self.actionExit.triggered.connect(MainWindow.close)
		self.actionOpen_csv_file.triggered.connect(self.getFile)
		self.dataset={}
		self.x_axis_slt=None
		self.y_axis_slt=None

	def selectXaxis(self,value):
		"""
		This function will update the plot according to the data of x axis selected from combo box

		"""
		print('x-axis',value)
		self.x_axis_slt=value
		self.Update(self.themes[0])
		
	def selectYaxis(self,value):
		"""
		This function will update the plot according to the data of y axis selected from combo box

		"""
		print('y-axis',value)
		self.y_axis_slt=value
		self.Update(self.themes[0])

	def Update(self,value):

		"""
		This function will input the value of theme and accordingly plot the data, if the data is relative, i.e., x verus y-axis
		then the user can assign x and y axis from the combo box. If all data should be plotted in paraller then leave,
		the combo boxes of axis selections to their default starting location.
			
		"""
		print("Value from Combo Box:",value)
		plt.clf()
		plt.style.use(value)
		try:
			self.horizontalLayout.removeWidget(self.toolbar)
			self.verticalLayout.removeWidget(self.canv)
			
			sip.delete(self.toolbar)
			sip.delete(self.canv)
			self.toolbar = None
			self.canv = None
			self.verticalLayout.removeItem(self.spacerItem1)
		except Exception as e:
			print(e)
			pass
		self.canv = MatplotlibCanvas(self)
		self.toolbar = Navi(self.canv,self.centralwidget)
		
		self.horizontalLayout.addWidget(self.toolbar)
		self.verticalLayout.addWidget(self.canv)
		
		self.canv.axes.cla()
		ax = self.canv.axes
		try:
	
			ax.plot(self.dataset[self.x_axis_slt],self.dataset[self.y_axis_slt],label=self.y_axis_slt) 
			legend = ax.legend()
			legend.set_draggable(True)
			ax.set_xlabel(self.x_axis_slt)
			ax.set_ylabel(self.y_axis_slt)
			ax.set_title(self.Title)
			plt.setp(ax.xaxis.get_majorticklabels(), rotation=25)  # uncomment if you want the x-axis to tilt 25 degree
			
		except Exception as e:
			print('==>',e)
			self.df.plot(ax = self.canv.axes)
			
		
			legend = ax.legend()
			legend.set_draggable(True)
			
			ax.set_xlabel('X axis')
			ax.set_ylabel('Y axis')
			ax.set_title(self.Title)
			pass
		
		self.canv.draw()
		
		
		
		
	
	def getFile(self):
		""" This function will get the address of the csv file location
			also calls a readData function 
		"""
		
		self.filename = QFileDialog.getOpenFileName(filter = "csv (*.csv)")[0]
		print("File :", self.filename)
		self.readData()
	
	def getDataset(self,csvfilename):
		"""
		This function will convert csv file to a dictionary of dataset, with keys as the columns' names and
		values as values. The datatime format should be one of the standard datatime formats. Before plottting
		we need to convert the string of data time in the csv file values to datatime format.
		"""
		df = pd.read_csv(csvfilename,encoding='utf-8').fillna(0)
		LIST_OF_COLUMNS = df.columns.tolist()
		dataset={}
		#time_format = '%Y-%m-%d %H:%M:%S.%f' # Please use this format for time_series-data_2.csv kind of time stamp
		time_format = '%d/%m/%Y %H:%M%f'     # Please use this format for time_series-data_1.csv kind of time stamp
		
		for col in LIST_OF_COLUMNS:
			dataset[col]  =  df[col].iloc[0:].values
			try:
				dataset[col] = [datetime.strptime(i, time_format) for i in df[col].iloc[0:].values]
			except Exception as e:
				pass
				print(e)
		return dataset,LIST_OF_COLUMNS

	def readData(self):
		""" This function will read the data using pandas and call the update
			function to plot
		"""
		import os
		self.dataset={}
		self.x_axis_slt=None
		self.y_axis_slt=None

		base_name = os.path.basename(self.filename)
		self.Title = os.path.splitext(base_name)[0]
		print('FILE',self.Title )
		self.dataset, LIST_OF_COLUMNS = self.getDataset(self.filename)
		
		self.df = pd.read_csv(self.filename,encoding = 'utf-8').fillna(0)
		
		self.Update(self.themes[0]) # lets 0th theme be the default : bmh
		self.comboBox_1.clear()
		self.comboBox_2.clear()
		self.comboBox_1.addItems(['Select horizontal axis here'])
		self.comboBox_2.addItems(['Select vertical axis here'])
		self.comboBox_1.addItems(LIST_OF_COLUMNS)
		self.comboBox_2.addItems(LIST_OF_COLUMNS)

	

	
	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
		self.label.setText(_translate("MainWindow", "Select Theme"))
		self.label_1.setText(_translate("MainWindow", "X-axis"))
		self.label_2.setText(_translate("MainWindow", "Y-axis"))
		self.pushButton.setText(_translate("MainWindow", "Open"))
		self.menuFile.setTitle(_translate("MainWindow", "File"))
		self.actionOpen_csv_file.setText(_translate("MainWindow", "Open csv file"))
		self.actionExit.setText(_translate("MainWindow", "Exit"))

# Subscribe to PyShine Youtube channel for more detail! 

# WEBSITE: www.pyshine.com

if __name__ == "__main__":
	import sys
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow)
	
	MainWindow.show()
	sys.exit(app.exec_())


```

## drag_drop.py

```python

# Lets make the main window class
# Subscribe to PyShine Youtube channel for more detail! 
# WEBSITE: www.pyshine.com
from main import Ui_MainWindow
import sys
from PyQt5 import QtCore, QtWidgets
import platform


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
	def __init__(self):
		"""
		This function initializes our main window from the main.py, set its title 
		and also allow the drops on it.
		"""
		super().__init__()
		self.setupUi(self)
		self.setWindowTitle('PyShine drag drop plot')
		self.setAcceptDrops(True)
	def dragEnterEvent(self, e):
		"""
		This function will detect the drag enter event from the mouse on the main window
		"""
		if e.mimeData().hasUrls:
			e.accept()
		else:
			e.ignore()
	def dragMoveEvent(self,e):
		"""
		This function will detect the drag move event on the main window
		"""
		if e.mimeData().hasUrls:
			e.accept()
		else:
			e.ignore()
	def dropEvent(self,e):
		"""
		This function will enable the drop file directly on to the 
		main window. The file location will be stored in the self.filename
		"""
		if e.mimeData().hasUrls:
			e.setDropAction(QtCore.Qt.CopyAction)
			e.accept()
			for url in e.mimeData().urls():
				fname = str(url.toLocalFile())
			self.filename = fname
			print("GOT ADDRESS:",self.filename)
			self.readData()
		else:
			e.ignore() # just like above functions	
# Subscribe to PyShine Youtube channel for more detail! 
# WEBSITE: www.pyshine.com		
if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())
	
	
```

### performance_metrics.csv
```csv
Variance score,Mean Absolute Error,Mean Square Error,Root Mean Square Error,R2
0.174653649,0.10403642,0.032201797,0.179448591,0.055943322
0.154518962,0.10784932,0.03478812,0.186515738,0.027057328
0.123427689,0.11292368,0.037969694,0.194858137,-0.006755184
0.122108102,0.11730553,0.041560065,0.203862858,-0.00646342
0.121421337,0.12173681,0.045150366,0.212486154,-0.007069634
0.119434297,0.12569132,0.045021992,0.212183864,-0.012261498
0.117850363,0.12962678,0.044915833,0.211933557,-0.017949929
0.11405772,0.13792177,0.051269993,0.226428782,-0.027874308
0.111305952,0.14622809,0.05761102,0.240022954,-0.037620495
0.100067019,0.14161506,0.0548042,0.234102965,-0.045218857
0.087266326,0.1368897,0.052055642,0.228157056,-0.055802489
0.114817798,0.12368377,0.043832093,0.209361156,0.0062935
0.151865721,0.11043203,0.035611518,0.188710142,0.074067622
0.154146492,0.106421374,0.0320584,0.179048594,0.077624776
0.156415761,0.10234559,0.0285632,0.169006507,0.079329505
0.136851907,0.1102752,0.035450008,0.188281724,0.047548182
0.121788442,0.11816314,0.04243706,0.206002577,0.019450136
0.133582175,0.11679362,0.04048665,0.201212944,0.02434273
0.145928741,0.11554903,0.038469866,0.196137365,0.030796711
0.14290756,0.11104645,0.03395744,0.184275447,0.03080231


```
Important in the main.py: 
      ```
      time_format = '%Y-%m-%d %H:%M:%S.%f' # Please use this format for time_series-data_2.csv kind of time stamp
      time_format = '%d/%m/%Y %H:%M%f'     # Please use this format for time_series-data_1.csv kind of time stamp
      ```
### Date Time formats

To understand the various formats that can be used for the date time formatting, please have a look at this code:

```python
import datetime

formats=["%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%d-%m-%Y",
        "%H:%M:%S",
        "%M:%S"
        ] 


for ft in formats:
    time = datetime.datetime.now()
    time = time.strftime(ft)
    print("Format",ft,": ", time)
```
You can try the above code: 
<br>
<form action="https://pyshine.com/sww/configure/date_time_formats.html" method="get" target="_blank"><button type="submit">Try code Yourself!</button></form>
<br>



### time_series-data_2.csv

```csv
timestamp, light, orange, mango, banana, pear
2020-08-12 22:45:12.826871, 65, 244, 213, 196, 21.625
2020-08-12 22:50:14.151601, 66, 246, 208, 196, 21.312
2020-08-12 22:55:15.399692, 15, 247, 208, 196, 21.375
2020-08-12 23:00:16.717546, 15, 248, 209, 195, 21.5
2020-08-12 23:05:18.041433, 15, 249, 212, 195, 21.625
2020-08-12 23:10:19.372733, 16, 248, 216, 195, 21.687

```
### time_series_data_1.csv
```csv
time,data,# Lane Points,% Observed
04/03/2016 0:00,16,1,100
04/03/2016 0:05,10,1,100
04/03/2016 0:10,11,1,100
04/03/2016 0:15,11,1,100
04/03/2016 0:20,6,1,100
04/03/2016 0:25,13,1,100
04/03/2016 0:30,7,1,100
04/03/2016 0:35,2,1,100
04/03/2016 0:40,6,1,100
04/03/2016 0:45,7,1,100
04/03/2016 0:50,4,1,100
04/03/2016 0:55,7,1,100
04/03/2016 1:00,12,1,100
04/03/2016 1:05,5,1,100
04/03/2016 1:10,10,1,100
04/03/2016 1:15,10,1,100
04/03/2016 1:20,1,1,100
04/03/2016 1:25,6,1,100
04/03/2016 1:30,6,1,100
04/03/2016 1:35,5,1,100
04/03/2016 1:40,3,1,100
04/03/2016 1:45,3,1,100
04/03/2016 1:50,7,1,100
04/03/2016 1:55,5,1,100
04/03/2016 2:00,2,1,100
04/03/2016 2:05,1,1,100
04/03/2016 2:10,1,1,100
04/03/2016 2:15,4,1,100
04/03/2016 2:20,7,1,100
04/03/2016 2:25,4,1,100
04/03/2016 2:30,5,1,100
04/03/2016 2:35,4,1,100
04/03/2016 2:40,4,1,100
04/03/2016 2:45,6,1,100
04/03/2016 2:50,1,1,100
04/03/2016 2:55,5,1,100
04/03/2016 3:00,6,1,100
04/03/2016 3:05,3,1,100
04/03/2016 3:10,3,1,100
04/03/2016 3:15,5,1,100
04/03/2016 3:20,4,1,100
04/03/2016 3:25,2,1,100
04/03/2016 3:30,8,1,100
04/03/2016 3:35,8,1,100
04/03/2016 3:40,5,1,100
04/03/2016 3:45,7,1,100
04/03/2016 3:50,10,1,100
04/03/2016 3:55,5,1,100
04/03/2016 4:00,10,1,100
04/03/2016 4:05,8,1,100
04/03/2016 4:10,14,1,100
04/03/2016 4:15,6,1,100
04/03/2016 4:20,18,1,100
04/03/2016 4:25,18,1,100
04/03/2016 4:30,15,1,100
04/03/2016 4:35,26,1,100
04/03/2016 4:40,33,1,100
04/03/2016 4:45,27,1,100
04/03/2016 4:50,37,1,100
04/03/2016 4:55,42,1,100
04/03/2016 5:00,42,1,100
04/03/2016 5:05,48,1,100
04/03/2016 5:10,46,1,100
04/03/2016 5:15,61,1,100
04/03/2016 5:20,57,1,100
04/03/2016 5:25,58,1,100
04/03/2016 5:30,66,1,100
04/03/2016 5:35,67,1,100
04/03/2016 5:40,105,1,100
04/03/2016 5:45,89,1,100
04/03/2016 5:50,93,1,100
04/03/2016 5:55,89,1,100
04/03/2016 6:00,102,1,100
04/03/2016 6:05,107,1,100
04/03/2016 6:10,113,1,100
04/03/2016 6:15,132,1,100
04/03/2016 6:20,142,1,100
04/03/2016 6:25,133,1,100
04/03/2016 6:30,149,1,100
04/03/2016 6:35,145,1,100
04/03/2016 6:40,152,1,100
04/03/2016 6:45,161,1,100
04/03/2016 6:50,148,1,100
04/03/2016 6:55,134,1,100
04/03/2016 7:00,181,1,100
04/03/2016 7:05,127,1,100
04/03/2016 7:10,145,1,100
04/03/2016 7:15,120,1,100
04/03/2016 7:20,122,1,100
04/03/2016 7:25,119,1,100
04/03/2016 7:30,101,1,100
04/03/2016 7:35,81,1,100
04/03/2016 7:40,93,1,100
04/03/2016 7:45,84,1,100
04/03/2016 7:50,66,1,100
04/03/2016 7:55,78,1,100
04/03/2016 8:00,90,1,100
04/03/2016 8:05,89,1,100
04/03/2016 8:10,99,1,100
04/03/2016 8:15,96,1,100
04/03/2016 8:20,94,1,100
04/03/2016 8:25,80,1,100
04/03/2016 8:30,73,1,100
04/03/2016 8:35,79,1,100
04/03/2016 8:40,94,1,100
04/03/2016 8:45,94,1,100
04/03/2016 8:50,80,1,100
04/03/2016 8:55,77,1,100
04/03/2016 9:00,95,1,100
04/03/2016 9:05,103,1,100
04/03/2016 9:10,93,1,100
04/03/2016 9:15,100,1,100
04/03/2016 9:20,93,1,100
04/03/2016 9:25,104,1,100
04/03/2016 9:30,102,1,100
04/03/2016 9:35,105,1,100
04/03/2016 9:40,107,1,100
04/03/2016 9:45,135,1,100
04/03/2016 9:50,97,1,100
04/03/2016 9:55,105,1,100
04/03/2016 10:00,115,1,100
04/03/2016 10:05,100,1,100
04/03/2016 10:10,111,1,100
04/03/2016 10:15,123,1,100
04/03/2016 10:20,104,1,100
04/03/2016 10:25,94,1,100
04/03/2016 10:30,95,1,100
04/03/2016 10:35,110,1,100
04/03/2016 10:40,113,1,100
04/03/2016 10:45,124,1,100
04/03/2016 10:50,115,1,100
04/03/2016 10:55,120,1,100
04/03/2016 11:00,120,1,100
04/03/2016 11:05,95,1,100
04/03/2016 11:10,109,1,100
04/03/2016 11:15,110,1,100
04/03/2016 11:20,117,1,100
04/03/2016 11:25,114,1,100
04/03/2016 11:30,109,1,100
04/03/2016 11:35,110,1,100
04/03/2016 11:40,104,1,100
04/03/2016 11:45,119,1,100
04/03/2016 11:50,105,1,100
04/03/2016 11:55,105,1,100
04/03/2016 12:00,116,1,100
04/03/2016 12:05,97,1,100
04/03/2016 12:10,98,1,100
04/03/2016 12:15,91,1,100
04/03/2016 12:20,104,1,100
04/03/2016 12:25,113,1,100
04/03/2016 12:30,101,1,100
04/03/2016 12:35,120,1,100
04/03/2016 12:40,115,1,100
04/03/2016 12:45,125,1,100
04/03/2016 12:50,114,1,100
04/03/2016 12:55,98,1,100
04/03/2016 13:00,115,1,100
04/03/2016 13:05,120,1,100
04/03/2016 13:10,109,1,100
04/03/2016 13:15,112,1,100
04/03/2016 13:20,111,1,100
04/03/2016 13:25,112,1,100
04/03/2016 13:30,100,1,100
04/03/2016 13:35,96,1,100
04/03/2016 13:40,109,1,100
04/03/2016 13:45,115,1,100
04/03/2016 13:50,117,1,100
04/03/2016 13:55,98,1,100
04/03/2016 14:00,113,1,100
04/03/2016 14:05,92,1,100
04/03/2016 14:10,107,1,100
04/03/2016 14:15,97,1,100
04/03/2016 14:20,104,1,100
04/03/2016 14:25,91,1,100
04/03/2016 14:30,109,1,100
04/03/2016 14:35,91,1,100
04/03/2016 14:40,105,1,100
04/03/2016 14:45,100,1,100
04/03/2016 14:50,104,1,100
04/03/2016 14:55,106,1,100
04/03/2016 15:00,89,1,100
04/03/2016 15:05,89,1,100
04/03/2016 15:10,100,1,100
04/03/2016 15:15,113,1,100
04/03/2016 15:20,104,1,100
04/03/2016 15:25,105,1,100
04/03/2016 15:30,93,1,100
04/03/2016 15:35,92,1,100
04/03/2016 15:40,113,1,100
04/03/2016 15:45,126,1,100
04/03/2016 15:50,120,1,100
04/03/2016 15:55,91,1,100
04/03/2016 16:00,92,1,100
04/03/2016 16:05,89,1,100
04/03/2016 16:10,101,1,100
04/03/2016 16:15,109,1,100
04/03/2016 16:20,95,1,100
04/03/2016 16:25,87,1,100
04/03/2016 16:30,84,1,100
04/03/2016 16:35,87,1,100
04/03/2016 16:40,89,1,100
04/03/2016 16:45,98,1,100
04/03/2016 16:50,104,1,100
04/03/2016 16:55,87,1,100
04/03/2016 17:00,79,1,100
04/03/2016 17:05,83,1,100
04/03/2016 17:10,97,1,100
04/03/2016 17:15,86,1,100
04/03/2016 17:20,99,1,100
04/03/2016 17:25,98,1,100
04/03/2016 17:30,89,1,100
04/03/2016 17:35,99,1,100
04/03/2016 17:40,85,1,100
04/03/2016 17:45,76,1,100
04/03/2016 17:50,77,1,100
04/03/2016 17:55,83,1,100
04/03/2016 18:00,93,1,100
04/03/2016 18:05,71,1,100
04/03/2016 18:10,71,1,100
04/03/2016 18:15,74,1,100
04/03/2016 18:20,102,1,100
04/03/2016 18:25,89,1,100
04/03/2016 18:30,82,1,100
04/03/2016 18:35,76,1,100
04/03/2016 18:40,64,1,100
04/03/2016 18:45,77,1,100
04/03/2016 18:50,77,1,100
04/03/2016 18:55,85,1,100
04/03/2016 19:00,86,1,100
04/03/2016 19:05,83,1,100
04/03/2016 19:10,93,1,100
04/03/2016 19:15,83,1,100
04/03/2016 19:20,96,1,100
04/03/2016 19:25,97,1,100
04/03/2016 19:30,86,1,100

```


To run the project simply:

```
python3 drag_drop.py

or 

python drag_drop.py
```

[Download sample csv data files](https://drive.google.com/file/d/10gvk-A0orWWktIaHkw7WMAGzYcoMkB9t/view?usp=sharing)
	
	
	
	
	
	

