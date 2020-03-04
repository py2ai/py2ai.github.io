---
layout: post
title: Making Python GUI for sine and cosine
author: Hussain A.
categories: [Keras tutorial series]
mathjax: true
summary: A quick tutorial on pyqtgraph GUI
---






## A quick and easy Classification model

Hi there! 
![]({{ "assets/img/posts/lab5_keras_model.png" | absolute_url }}). Lets import the necessary components. 
```# PyShine GUI series Lab-1
# Lets import the required libraries
import sys
from PyQt5 import QtGui, QtGui
import numpy as np
import pyqtgraph as pg

# Lets make the initialization variables

region = pg.LinearRegionItem()
minX=0
maxX=0
vb=[]
data1=0
data2=0
dataPosX=0

# Cross hair generation in terms of verticle and horizontal lines

vLine=pg.InfiniteLine(angle=90,movable=False)
hLine=pg.InfiniteLine(angle=0,movable=False)

# Ok so lets put some colors R,G,B values to a List

Colors_Set = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),(44, 160, 44), (152, 223, 138)]

# Lets name the curves some in majors Legend

majors = ["Sine","Cosine"]

# It would be awesome to let us control the background colors
# Lets choose as white w

pg.setConfigOption("background","w")

# Its time to make the major Class , lets call it pyshine_plot

class pyshine_plot(QtGui.QWidget):
	def __init__(self):
		global dataPosX
		super(pyshine_plot,self).__init__()
		self.amplitude=10
		self.init_ui()
		self.t = 0
		self.qt_connections()
		self.num_of_curves=2 # since we are plotting only two curves
		
		plotCurveIds = ["%d" % x for x in np.arange(self.num_of_curves)]
		curvePointsIds = ["%d" % x for x in np.arange(self.num_of_curves)]
		textIds = ["%d" % x for x in np.arange(self.num_of_curves)]
		arrowIds = ["%d" % x for x in np.arange(self.num_of_curves)]
		dataIds = ["%d" % x for x in np.arange(self.num_of_curves)]
		
		self.plotcurves = plotCurveIds
		self.curvePoints = curvePointsIds
		self.texts = textIds
		self.arrows = arrowIds
		self.datas = dataIds
		
		# Lets iterate over the number of cuvers to assign each plot curvePointsIds
		for k in range (self.num_of_curves):
			 self.plotcurves[k] = pg.PlotCurveItem()
		# Here we can call an update Plot functions
		self.updateplot()
		
		# Here we can again use the for loop for the rest of items
		for k in range (self.num_of_curves):
			self.plotwidget.addItem(self.plotcurves[k])
			self.curvePoints[k] = pg.CurvePoint(self.plotcurves[k])
			self.plotwidget.addItem(self.curvePoints[k])
			self.texts[k] = pg.TextItem(str(k),color=Colors_Set[k+2],anchor=(0.5,-1.0))
			# Here we require setParent on the TextItem
			self.texts[k].setParentItem(self.curvePoints[k])
			self.arrows[k] = pg.ArrowItem(angle=60,pen=Colors_Set[k+2],brush=Colors_Set[k])
			self.arrows[k].setParentItem(self.curvePoints[k])
		
		# Its time to make a proxy signal 
		
		self.proxy = pg.SignalProxy (self.plotwidget.scene().sigMouseMoved,rateLimit = 60, slot=self.mouseMoved)
		self.timer = pg.QtCore.QTimer()
		self.timer.timeout.connect(self.moveplot)
		self.timer.start(1000)
		
	# Alright so lets make the init_ui functions
	def init_ui(self):
		global region
		global minX
		global maxX
		global vLine
		global hLine
		global vb
		self.setWindowTitle("PyShine")
		self.label = pg.LabelItem(justify="left")
		hbox = QtGui.QVBoxLayout()
		
		self.setLayout(hbox)
		self.plotwidget = pg.PlotWidget()
		self.plotwidget.addItem(vLine,ignoreBounds=True)
		self.vb = self.plotwidget.plotItem.vb
		self.plotwidget.addItem(self.label)
		hbox.addWidget(self.plotwidget)
		self.increasebutton = QtGui.QPushButton("Increase Amplitude")
		self.decreasebutton = QtGui.QPushButton("Decrease Amplitude")
		# And now we add this buttons to the horizontal box hbox
		
		hbox.addWidget(self.increasebutton)
		hbox.addWidget(self.decreasebutton)
		self.show()
	# Ok so now we make the mouseMoved function, it will indicate the data1
	# as we move the mouse on the plot curves
	
	def mouseMoved(self,evt): # Here evt means event
		global hLine
		global vLine
		global data1
		global data2
		global dataPosX
		pos = evt[0] # Remember that using proxy signal we get the original arguments in a tuple
		
		if self.plotwidget.sceneBoundingRect().contains(pos):
			mousePoint = self.vb.mapSceneToView(pos)
			index = int(mousePoint.x())
			if index >=0 and index < len(self.datas[0]):
				dataPosX = mousePoint.x()
				# Here we have obtained the mouse x point
				# SO lets use a for loop for each curve to set the Pos 
				
				for m in range (self.num_of_curves):
					self.curvePoints[m].setPos(float(index)/(len(self.datas[m])-1))
					T = majors[m] # Get the respective text string of the Legend
					self.texts[m].setText("[%0.1f,%0.1f]:"%(dataPosX,self.datas[m][index])+str(T))
			# Now we can set Pos of the vLine and hLine as the mousePoint
			vLine.setPos(mousePoint.x())
			hLine.setPos(mousePoint.y())
	
	# Lets make the qt_connections function for the buttons
	def qt_connections(self):
		self.increasebutton.clicked.connect(self.on_increasebutton_clicked)
		self.decreasebutton.clicked.connect(self.on_decreasebutton_clicked)
		
	# Ok so another function to update plot
	def moveplot(self):
		self.updateplot() # Rather we can call the function
		
	# Update the data on plot 
	def updateplot(self):
		global data1
		global data2
		
		# Now comes the time to make the plot functions such as cos and sine
		# Lets increase the samples in both to 20
		# As we increase the samples the plot becomes smoother
		self.datas[0] = self.amplitude*np.sin(np.linspace(-2*np.pi,2*np.pi,1000)+self.t) # A single sine wave from 0 to 2pi consisting of 201 points
		self.datas[1] = self.amplitude*np.cos(np.linspace(0,2*np.pi,1000)+self.t) # A single cosine wave from 0 to 2pi consisting of 201 points
		for j in range(self.num_of_curves):
			pen = pg.mkPen(color=Colors_Set[j+2],width=5)
			# Here it should be color not colors 
			self.plotcurves[j].setData(self.datas[j],pen=pen,clickable=True)
			# pen is for the color and plot curves get the data
	
	# Alright so lets make the button functions to increment or decrement the amplitude
	def on_increasebutton_clicked (self):
		self.amplitude+=1
		self.updateplot()
		
	def on_decreasebutton_clicked (self):
		self.amplitude-=1
		self.updateplot()

# Alright so the functions are done and now we can make the main function to run the app

def main():
	import sys
	app = QtGui.QApplication(sys.argv)
	app.setApplicationName("Sinwave")
	ex = pyshine_plot()
	if (sys.flags.interactive!=1)or not hasattr(QtCore,"PYQT_VERSION"):
		sys.exit(app.exec_())
if __name__ == "__main__":
	main()
	
# That ends the required coding, lets run it 
# As we can see that currently there is no color or even text at the cross hair
# Lets fix this first
# Lets run again!
# Ok so the text is appearing now however the colors of plots need a fix
# Lets try again!
# Ok so that is working now
# Lets change the colors its now taking colors from the index positions 2,3
# Lets change the sine wave x axis from -2pi to +2pi
# Lets reduce the number of samples in the sine wave to 8

# Thats all for today, I hope you will learn and enjoy from this video
# Thanks and have a nice day!
	
```




