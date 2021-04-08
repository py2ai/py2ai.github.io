---
layout: post
title: How to open and show the SQL database file in a PyQt5 GUI
categories: [GUI tutorial series]
mathjax: true
featured-img: ssqlgui
summary:  This tutorial is about using sqlite3 to open, and show the data inside db file
---

Hi friends! Hope you are doing great. Today's tutorial is simple but interesting because we will see how to use PyQt5 based GUI to display the data. In a previous tutorial
about the basics of SQLite3 we have learnt how to generate the ```.db``` database file in Python. We can use the same file here to open up and populate the table of
the PyQt5 GUI.

The GUI will have two buttons ```LOAD``` and ```SHOW```, the first one will be used to open up a file dialog so that you can select a `.db` file and then click
on the second button to visualize all the data inside this database file. Let's have a look at the overall code as shown below


### main.py

```python
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import sqlite3
import time
import datetime
from PyQt5.QtCore import pyqtSlot
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.threadpool = QtCore.QThreadPool()
        MainWindow.setObjectName("MainWindow")
        """ Add logo in main window """
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        """ LOGO end """
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setMinimumSize(QtCore.QSize(56, 17))
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setMinimumSize(QtCore.QSize(56, 17))
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 1, 1, 1, 1)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.gridLayout.addWidget(self.widget, 2, 1, 1, 1)
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.gridLayout.addWidget(self.tableWidget, 0, 0, 3, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 426, 18))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuForm = QtWidgets.QMenu(self.menubar)
        self.menuForm.setObjectName("menuForm")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionRecent = QtWidgets.QAction(MainWindow)
        self.actionRecent.setObjectName("actionRecent")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionRecent)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuForm.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.projectName =  ''
        self.pushButton.clicked.connect(self.openFile)
        self.pushButton_2.clicked.connect(self.workerStart)
        self.MainWindow = MainWindow
        self.col_num = 7
        self.id = 10020
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "LOGO"))
        self.pushButton.setText(_translate("MainWindow", "LOAD"))
        self.pushButton_2.setText(_translate("MainWindow", "SHOW"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuForm.setTitle(_translate("MainWindow", "Form"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionRecent.setText(_translate("MainWindow", "Recent"))
        self.conn=''

    def openFile(self):
        projectName = QFileDialog.getOpenFileName(filter="Data (*.db)")
        print("File name: ",projectName[0])
        self.projectName = projectName[0]
        
    def workerStart(self):
        worker = Worker(self.loadDataBase)
        self.threadpool.start(worker)

    def loadDataBase(self):
        
        self.conn  = sqlite3.connect(self.projectName)
        query = "SELECT * FROM COMPANY"			
        cursor = self.conn.execute(query)
        row_len = []
        for i in cursor:
            row_len.append(len(i))
        self.col_num = max(row_len)
        self.tableWidget.setRowCount(0)	
        self.tableWidget.setColumnCount(int(self.col_num))
        
        cursor = self.conn.execute(query)
        for row, row_data in enumerate(cursor):
            self.tableWidget.insertRow(row)
            for col, col_data in enumerate(row_data):
                self.tableWidget.setItem(row, col, QtWidgets.QTableWidgetItem(str(col_data)))

            
            
        self.conn.close()
        

# visit www.pyshine.com for more details
class Worker(QtCore.QRunnable):  

    def __init__(self, fnc, *args, **kwargs):
        super(Worker, self).__init__()        
        self.fnc = fnc
        self.args = args
        self.kwargs = kwargs
    @pyqtSlot()
    def run(self):
        
        self.fnc(*self.args, **self.kwargs)	
	
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    """ DYNAMICALLY ARRANGE SIZE of MAIN WINDOW """
    screen = app.primaryScreen()
    size = screen.size()
    print('Size: %d x %d' % (size.width(), size.height()))
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.resize(size.width()//2, size.height()//2)
    MainWindow.show()
    sys.exit(app.exec_())

```

### Details of above code

We will have two functions in the above code connected to the PUSH buttons

```python
  self.pushButton.clicked.connect(self.openFile)
  self.pushButton_2.clicked.connect(self.workerStart)
```

The `openFile` function will get us the path of the `.db` file. The second push button will call `workerStart` function that will start a thread to `loadDataBase`.
This will load everything from the `COMPANY` database in a query as shown below:

```python
 def loadDataBase(self):
        
        self.conn  = sqlite3.connect(self.projectName)
        query = "SELECT * FROM COMPANY"		
```

Then we will start the cursor for this query and populate a list as shown:

```python
cursor = self.conn.execute(query)
row_len = []
for i in cursor:
    row_len.append(len(i))
```
Finally the table widget is populated in the rest of the code:

```python
self.col_num = max(row_len)
self.tableWidget.setRowCount(0)	
self.tableWidget.setColumnCount(int(self.col_num))

cursor = self.conn.execute(query)
for row, row_data in enumerate(cursor):
    self.tableWidget.insertRow(row)
    for col, col_data in enumerate(row_data):
	self.tableWidget.setItem(row, col, QtWidgets.QTableWidgetItem(str(col_data)))

```

