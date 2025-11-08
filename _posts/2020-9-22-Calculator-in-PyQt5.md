---
categories:
- GUI tutorial series
description: Hi friends! this tutorial is about making a calculator application in PyQt5. The calc.ui file contains information about the gui and main.py is all about the...
featured-img: calc
keywords:
- Python
- PyQt5
- Calculator
- GUI Design
layout: post
mathjax: true
tags:
- Python
- PyQt5
- Calculator
- GUI
title: How to make a calculator GUI in python with PyQt5
---


Hi friends! this tutorial is about making a calculator application in PyQt5. The calc.ui file contains information about the gui and main.py is all about the calculator functionality. Copy both the calc.ui and main.py files in a directory and open the terminal and cd to the directory containing both files.
Run `python3 main.py` and calc application will start.

<br>
<div align="center">
	<iframe width="560" height="315" src="https://www.youtube.com/embed/c29ejbtzM3c" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>

# calc.ui

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
    <width>352</width>
    <height>310</height>
   </rect>
  </property>
  <property name="windowTitle">
   <string>MainWindow</string>
  </property>
  <widget class="QWidget" name="centralwidget">
   <layout class="QGridLayout" name="gridLayout_3">
    <item row="0" column="0">
     <layout class="QGridLayout" name="gridLayout_2">
      <item row="0" column="0">
       <widget class="QTextEdit" name="textEdit"/>
      </item>
      <item row="1" column="0">
       <layout class="QGridLayout" name="gridLayout">
        <item row="0" column="0">
         <layout class="QHBoxLayout" name="horizontalLayout">
          <item>
           <widget class="QPushButton" name="pushButton">
            <property name="text">
             <string>%</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_2">
            <property name="text">
             <string>x^3</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_3">
            <property name="text">
             <string>AC</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_4">
            <property name="text">
             <string>Backspace</string>
            </property>
           </widget>
          </item>
         </layout>
        </item>
        <item row="1" column="0">
         <layout class="QHBoxLayout" name="horizontalLayout_2">
          <item>
           <widget class="QPushButton" name="pushButton_5">
            <property name="text">
             <string>1/x</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_6">
            <property name="text">
             <string>x^2</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_7">
            <property name="text">
             <string>sqrt</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_8">
            <property name="text">
             <string>/</string>
            </property>
           </widget>
          </item>
         </layout>
        </item>
        <item row="2" column="0">
         <layout class="QHBoxLayout" name="horizontalLayout_4">
          <item>
           <widget class="QPushButton" name="pushButton_13">
            <property name="text">
             <string>7</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_14">
            <property name="text">
             <string>8</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_15">
            <property name="text">
             <string>9</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_16">
            <property name="text">
             <string>*</string>
            </property>
           </widget>
          </item>
         </layout>
        </item>
        <item row="3" column="0">
         <layout class="QHBoxLayout" name="horizontalLayout_3">
          <item>
           <widget class="QPushButton" name="pushButton_9">
            <property name="text">
             <string>4</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_10">
            <property name="text">
             <string>5</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_11">
            <property name="text">
             <string>6</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_12">
            <property name="text">
             <string>-</string>
            </property>
           </widget>
          </item>
         </layout>
        </item>
        <item row="4" column="0">
         <layout class="QHBoxLayout" name="horizontalLayout_6">
          <item>
           <widget class="QPushButton" name="pushButton_21">
            <property name="text">
             <string>1</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_22">
            <property name="text">
             <string>2</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_23">
            <property name="text">
             <string>3</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_24">
            <property name="text">
             <string>+</string>
            </property>
           </widget>
          </item>
         </layout>
        </item>
        <item row="5" column="0">
         <layout class="QHBoxLayout" name="horizontalLayout_5">
          <item>
           <widget class="QPushButton" name="pushButton_17">
            <property name="text">
             <string>+/-</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_18">
            <property name="text">
             <string>0</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_19">
            <property name="text">
             <string>.</string>
            </property>
           </widget>
          </item>
          <item>
           <widget class="QPushButton" name="pushButton_20">
            <property name="text">
             <string>=</string>
            </property>
           </widget>
          </item>
         </layout>
        </item>
       </layout>
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
````
## # calc.py

{% include codeHeader.html %}
```python

## # -*- coding: utf-8 -*-

## # Form implementation generated from reading ui file 'calc.ui'
##
## # Created by: PyQt5 UI code generator 5.11.3
##
## # WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import re
class Ui_MainWindow(object):
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(338, 395)
		icon = QtGui.QIcon()
		icon.addPixmap(QtGui.QPixmap("pyshine.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
		MainWindow.setWindowIcon(icon)
		self.mw  = MainWindow
		
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
		self.verticalLayout.setObjectName("verticalLayout")
		self.gridLayout = QtWidgets.QGridLayout()
		self.gridLayout.setObjectName("gridLayout")
		self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
		self.textEdit.setObjectName("textEdit")
		
		self.gridLayout.addWidget(self.textEdit, 0, 0, 1, 4)
		self.pushButton_22 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_22.setObjectName("pushButton_22")
		self.gridLayout.addWidget(self.pushButton_22, 1, 0, 1, 1)
		self.pushButton_19 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_19.setObjectName("pushButton_19")
		self.gridLayout.addWidget(self.pushButton_19, 1, 1, 1, 1)
		self.pushButton_23 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_23.setObjectName("pushButton_23")
		self.gridLayout.addWidget(self.pushButton_23, 1, 2, 1, 1)
		self.pushButton_21 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_21.setObjectName("pushButton_21")
		self.gridLayout.addWidget(self.pushButton_21, 2, 0, 1, 1)
		self.pushButton_24 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_24.setObjectName("pushButton_24")
		self.gridLayout.addWidget(self.pushButton_24, 2, 1, 1, 1)
		self.pushButton_17 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_17.setObjectName("pushButton_17")
		self.gridLayout.addWidget(self.pushButton_17, 2, 2, 1, 1)
		self.pushButton_20 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_20.setObjectName("pushButton_20")
		self.gridLayout.addWidget(self.pushButton_20, 2, 3, 1, 1)
		self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_8.setObjectName("pushButton_8")
		self.gridLayout.addWidget(self.pushButton_8, 3, 0, 1, 1)
		self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_10.setObjectName("pushButton_10")
		self.gridLayout.addWidget(self.pushButton_10, 3, 1, 1, 1)
		self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_9.setObjectName("pushButton_9")
		self.gridLayout.addWidget(self.pushButton_9, 3, 2, 1, 1)
		self.pushButton_16 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_16.setObjectName("pushButton_16")
		self.gridLayout.addWidget(self.pushButton_16, 3, 3, 1, 1)
		self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_5.setObjectName("pushButton_5")
		self.gridLayout.addWidget(self.pushButton_5, 4, 0, 1, 1)
		self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_7.setObjectName("pushButton_7")
		self.gridLayout.addWidget(self.pushButton_7, 4, 1, 1, 1)
		self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_6.setObjectName("pushButton_6")
		self.gridLayout.addWidget(self.pushButton_6, 4, 2, 1, 1)
		self.pushButton_13 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_13.setObjectName("pushButton_13")
		self.gridLayout.addWidget(self.pushButton_13, 4, 3, 1, 1)
		self.pushButton = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton.setObjectName("pushButton")
		self.gridLayout.addWidget(self.pushButton, 5, 0, 1, 1)
		self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_3.setObjectName("pushButton_3")
		self.gridLayout.addWidget(self.pushButton_3, 5, 1, 1, 1)
		self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_4.setObjectName("pushButton_4")
		self.gridLayout.addWidget(self.pushButton_4, 5, 2, 1, 1)
		self.pushButton_14 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_14.setObjectName("pushButton_14")
		self.gridLayout.addWidget(self.pushButton_14, 5, 3, 1, 1)
		self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_2.setObjectName("pushButton_2")
		self.gridLayout.addWidget(self.pushButton_2, 6, 0, 1, 1)
		self.pushButton_12 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_12.setObjectName("pushButton_12")
		self.gridLayout.addWidget(self.pushButton_12, 6, 1, 1, 1)
		self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_11.setObjectName("pushButton_11")
		self.gridLayout.addWidget(self.pushButton_11, 6, 2, 1, 1)
		self.pushButton_15 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_15.setObjectName("pushButton_15")
		self.gridLayout.addWidget(self.pushButton_15, 6, 3, 1, 1)
		self.pushButton_18 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_18.setObjectName("pushButton_18")
		self.gridLayout.addWidget(self.pushButton_18, 1, 3, 1, 1)
		self.verticalLayout.addLayout(self.gridLayout)
		MainWindow.setCentralWidget(self.centralwidget)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setObjectName("statusbar")
		MainWindow.setStatusBar(self.statusbar)
		self.actionNew = QtWidgets.QAction(MainWindow)
		icon = QtGui.QIcon()
		icon.addPixmap(QtGui.QPixmap("pyshine.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
		self.actionNew.setIcon(icon)
		self.actionNew.setObjectName("actionNew")
		self.actionExit = QtWidgets.QAction(MainWindow)
		self.actionExit.setObjectName("actionExit")
		self.actionitem1 = QtWidgets.QAction(MainWindow)
		self.actionitem1.setObjectName("actionitem1")
		self.actionitem2 = QtWidgets.QAction(MainWindow)
		self.actionitem2.setObjectName("actionitem2")
		self.actionaa = QtWidgets.QAction(MainWindow)
		self.actionaa.setObjectName("actionaa")

		self.retranslateUi(MainWindow)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)
		self.pushButton_12.clicked.connect(self.show) #0
		self.pushButton.clicked.connect(self.show) #1
		self.pushButton_3.clicked.connect(self.show)#2
		self.pushButton_4.clicked.connect(self.show)#3
		self.pushButton_5.clicked.connect(self.show)#4
		self.pushButton_7.clicked.connect(self.show)#5
		self.pushButton_6.clicked.connect(self.show)#6
		self.pushButton_8.clicked.connect(self.show)#7
		self.pushButton_10.clicked.connect(self.show)#8
		self.pushButton_9.clicked.connect(self.show)#9
		
		self.pushButton_14.clicked.connect(self.show)#+
		self.pushButton_13.clicked.connect(self.show)#-
		self.pushButton_16.clicked.connect(self.show)#x
		self.pushButton_20.clicked.connect(self.show)#/
		
		self.pushButton_11.clicked.connect(self.show)#.
		self.pushButton_18.clicked.connect(self.show)#<=
		self.pushButton_15.clicked.connect(self.show)#=
		self.pushButton_2.clicked.connect(self.show)#+/-
		
		self.pushButton_23.clicked.connect(self.show)#AC
		self.pushButton_19.clicked.connect(self.show)#x^3
		self.pushButton_22.clicked.connect(self.show)#%
		
		self.pushButton_21.clicked.connect(self.show)#1/x
		self.pushButton_24.clicked.connect(self.show)#x^2
		self.pushButton_17.clicked.connect(self.show)#sqrt
		
		self.text=''
		self.textEdit.setFontPointSize(24)
		self.processed=False
		
		
		
		 
		
		
		
	def process(self):
		try:
			inp=self.text
			inp = re.sub(r"\.(?![0-9])","", inp)
			val = eval(inp, {'__builtins__':None})
			self.text=str(val)
		except Exception as e:
			self.text = str(e)
		self.processed = True
			
		
		
	def show(self):
		self.textEdit.setFontPointSize(24)
		self.text = self.textEdit.toPlainText()
		
		num_list = [str(num) for num in [0,1,2,3,4,5,6,7,8,9,'.']]
		op_list = ['+','-','*','/','%']
		
		c_or_ce_list = ['AC']
		func_list=['1/x','x^2','sqrt','+/-','x^3']
		
		if self.mw.sender().text()!='Backspace':
			if self.mw.sender().text() in num_list :
				if self.processed == True:
					self.text=''
				self.text+=self.mw.sender().text()
				self.processed = False
			
			if self.mw.sender().text() in op_list :
				
				self.text+=self.mw.sender().text()
				self.processed = False
			
			if self.mw.sender().text() =='=':
				self.process()
			if self.mw.sender().text() in c_or_ce_list:
				self.text=''
				self.processed = False
			if self.mw.sender().text() in func_list:
				if self.mw.sender().text() == func_list[0]:
					try:
						self.text= str(1/eval(self.text))
					except Exception as e: 
						self.text=str(e)
					self.processed = False
				if self.mw.sender().text() == func_list[1]:
					self.text= str(eval(self.text)**2)
					self.processed = False
				if self.mw.sender().text() == func_list[2]:
					self.text= str(eval(self.text)**0.5)
					self.processed = False
				if self.mw.sender().text() == func_list[3]:
					self.text= str(-1*eval(self.text))	
					self.processed = False
				if self.mw.sender().text() == func_list[4]:
					self.text= str(eval(self.text)**3)
					self.processed = False
			
				
		else:
			self.text = self.text[0:len(self.text)-1]
			
		self.textEdit.setText(self.text)

	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
		self.pushButton_22.setText(_translate("PyShine Calculator", "%"))
		self.pushButton_22.setShortcut(_translate("MainWindow", "%"))
		self.pushButton_19.setText(_translate("MainWindow", "x^3"))
		self.pushButton_23.setText(_translate("MainWindow", "AC"))
		self.pushButton_23.setShortcut(_translate("MainWindow", "AC"))
		self.pushButton_21.setText(_translate("MainWindow", "1/x"))
		self.pushButton_24.setText(_translate("MainWindow", "x^2"))
		self.pushButton_17.setText(_translate("MainWindow", "sqrt"))
		self.pushButton_17.setShortcut(_translate("MainWindow", "S"))
		self.pushButton_20.setText(_translate("MainWindow", "/"))
		self.pushButton_20.setShortcut(_translate("MainWindow", "/"))
		self.pushButton_8.setText(_translate("MainWindow", "7"))
		self.pushButton_8.setShortcut(_translate("MainWindow", "7"))
		self.pushButton_10.setText(_translate("MainWindow", "8"))
		self.pushButton_10.setShortcut(_translate("MainWindow", "8"))
		self.pushButton_9.setText(_translate("MainWindow", "9"))
		self.pushButton_9.setShortcut(_translate("MainWindow", "9"))
		self.pushButton_16.setText(_translate("MainWindow", "*"))
		self.pushButton_16.setShortcut(_translate("MainWindow", "*"))
		self.pushButton_5.setText(_translate("MainWindow", "4"))
		self.pushButton_5.setShortcut(_translate("MainWindow", "4"))
		self.pushButton_7.setText(_translate("MainWindow", "5"))
		self.pushButton_7.setShortcut(_translate("MainWindow", "5"))
		self.pushButton_6.setText(_translate("MainWindow", "6"))
		self.pushButton_6.setShortcut(_translate("MainWindow", "6"))
		self.pushButton_13.setText(_translate("MainWindow", "-"))
		self.pushButton_13.setShortcut(_translate("MainWindow", "-"))
		self.pushButton.setText(_translate("MainWindow", "1"))
		self.pushButton.setShortcut(_translate("MainWindow", "1"))
		self.pushButton_3.setText(_translate("MainWindow", "2"))
		self.pushButton_3.setShortcut(_translate("MainWindow", "2"))
		self.pushButton_4.setText(_translate("MainWindow", "3"))
		self.pushButton_4.setShortcut(_translate("MainWindow", "3"))
		self.pushButton_14.setText(_translate("MainWindow", "+"))
		self.pushButton_14.setShortcut(_translate("MainWindow", "+"))
		self.pushButton_2.setText(_translate("MainWindow", "+/-"))
		self.pushButton_2.setShortcut(_translate("MainWindow", "#"))
		self.pushButton_12.setText(_translate("MainWindow", "0"))
		self.pushButton_12.setShortcut(_translate("MainWindow", "0"))
		self.pushButton_11.setText(_translate("MainWindow", "."))
		self.pushButton_11.setShortcut(_translate("MainWindow", "."))
		self.pushButton_15.setText(_translate("MainWindow", "="))
		self.pushButton_15.setShortcut(_translate("MainWindow", "Return"))
		self.pushButton_18.setText(_translate("MainWindow", "Backspace"))
		self.pushButton_18.setShortcut(_translate("MainWindow", "Backspace"))
		self.statusbar.setStatusTip(_translate("MainWindow", "Pyshine"))
		self.actionNew.setText(_translate("MainWindow", "New"))
		self.actionExit.setText(_translate("MainWindow", "Exit"))
		self.actionitem1.setText(_translate("MainWindow", "item1"))
		self.actionitem2.setText(_translate("MainWindow", "item2"))
		self.actionaa.setText(_translate("MainWindow", "aa"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


```
