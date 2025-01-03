---
layout: post
title: How to make a Matplotlib and PyQt5 based GUI with drag and drop the CSV file
categories: [GUI tutorial series]
mathjax: true
featured-img: pyqt7
description: Making a drag drop CSV file based matplotlib GUI with multiple themes
---


Hello friends, here is the code for the drag and drop enabled matplotlib GUI in PyQt5. Save the first main.py and the second drag_drop.py and run it. Enjoy, and do 
give your feedback and suggestions. Also please make sure that you are using Matplotlib version 3.2.1 or above. For more detail visit pyshine youtube channel.

<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/1q3Z2clyIEQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>

## main.py
{% include codeHeader.html %}
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

# We require a canvas class
import platform

# Use NSURL as a workaround to pyside/Qt4 behaviour for dragging and dropping on OSx
op_sys = platform.system()
if op_sys == 'Darwin':
    from Foundation import NSURL

class MatplotlibCanvas(FigureCanvasQTAgg):
	def __init__(self,parent=None, dpi = 120):
		fig = Figure(dpi = dpi)
		self.axes = fig.add_subplot(111)
		super(MatplotlibCanvas,self).__init__(fig)
		#fig.tight_layout() #  uncomment for tight layout
		
		

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
		self.horizontalLayout.addWidget(self.label)
		self.comboBox = QtWidgets.QComboBox(self.centralwidget)
		self.comboBox.setObjectName("comboBox")
		self.horizontalLayout.addWidget(self.comboBox)
		self.pushButton = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton.setObjectName("pushButton")
		self.horizontalLayout.addWidget(self.pushButton)
		spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
		self.horizontalLayout.addItem(spacerItem)
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
		
		self.pushButton.clicked.connect(self.getFile)
		self.comboBox.currentIndexChanged['QString'].connect(self.Update)
		self.actionExit.triggered.connect(MainWindow.close)
		self.actionOpen_csv_file.triggered.connect(self.getFile)
		
	def Update(self,value):
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
		self.df.plot(ax = self.canv.axes)
		legend = ax.legend()
		legend.set_draggable(True)
		
		ax.set_xlabel('X axis')
		ax.set_ylabel('Y axis')
		ax.set_title(self.Title)
		
		self.canv.draw()
		
		
		
		
	
	def getFile(self):
		""" This function will get the address of the csv file location
			also calls a readData function 
		"""
		self.filename = QFileDialog.getOpenFileName(filter = "csv (*.csv)")[0]
		print("File :", self.filename)
		self.readData()
	
	def readData(self):
		""" This function will read the data using pandas and call the update
			function to plot
		"""
		import os
		base_name = os.path.basename(self.filename)
		self.Title = os.path.splitext(base_name)[0]
		print('FILE',self.Title )
		self.df = pd.read_csv(self.filename,encoding = 'utf-8').fillna(0)
		self.Update(self.themes[0]) # lets 0th theme be the default : bmh
	

	
	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "PyShine simple plot"))
		self.label.setText(_translate("MainWindow", "Select Theme"))
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
{% include codeHeader.html %}
```python

# Lets make the main window class
# Subscribe to PyShine Youtube channel for more detail! 
# WEBSITE: www.pyshine.com
from main import Ui_MainWindow
import sys
from PyQt5 import QtCore, QtWidgets
import platform

op_sys = platform.system()
if op_sys == 'Darwin':
	from Foundation import  NSURL
# We use NSURL as a workaround to PySide/ Qt4 for drag/drop
# on OSx


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
				#if op_sys == 'Darwin':
				#	fname = str(NSURL.URLWithString_(str(url.toString())).filePathURL().path())
				#else:
				#	fname = str(url.toLocalFile())
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


[Download sample csv data files](https://drive.google.com/file/d/10gvk-A0orWWktIaHkw7WMAGzYcoMkB9t/view?usp=sharing)
	
	
	
	
	
	

