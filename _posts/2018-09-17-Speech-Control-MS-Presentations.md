---
layout: post
title: With Speech, control the MS Power Point Presentation
categories: [tutorial]
mathjax: true
summary: A simple AI application tutorial to control PPTX slides with speech using python
---

Hello friends, today i am going to present a very simple AI application to voice control the Microsoft PowerPoint Presentation.
The basic steps in making this application are:
1) Installation of the required libararies.
2) PySide based GUI to get the .pptx file and run it.
3) Listening to the voice command for two specific words such as; 'next' and 'bingo'. This is just for the demonstration purpose.
You can definetly choose your own magic words to move the slide. In my case saying the word 'next' will move forward the presentation
and `bingo` will move back the presentation.
4) For the speech recognition i am using SpeechRecognition API. 

Ok now lets start the first step, for this we need PySide. Go to [Gohlke](https://www.lfd.uci.edu/~gohlke/pythonlibs/#p=PySide) and download the relevant .whl file that suits your Python version.
Here is the list of available .whl files 

   `PySide‑1.2.4‑cp27‑cp27m‑win32.whl
    PySide‑1.2.4‑cp27‑cp27m‑win_amd64.whl
    PySide‑1.2.4‑cp34‑cp34m‑win32.whl
    PySide‑1.2.4‑cp34‑cp34m‑win_amd64.whl
    PySide‑1.2.4‑cp35‑cp35m‑win32.whl
    PySide‑1.2.4‑cp35‑cp35m‑win_amd64.whl
    PySide‑1.2.4‑cp36‑cp36m‑win32.whl
    PySide‑1.2.4‑cp36‑cp36m‑win_amd64.whl
    PySide‑1.2.4‑vc14‑x64.zip`

Lets take the example if you have Python 3.6 installed then download `PySide‑1.2.4‑cp36‑cp36m‑win_amd64.whl` and then go the download directory of .whl file and hold the shift button and right click to choose open command window here option. And in command prompt write the following and hit enter.
`pip install PySide‑1.2.4‑cp36‑cp36m‑win_amd64.whl`
Next we also need SpeechRecognition, so you can easily install it by 
`pip install SpeechRecognition`. 

I hope rest of the libraries in the code are already installed in your pc, otherwise you can easily install them using `pip instal`.

The import section of the Python code contains the PySide QtCore and Gui classes which will be used for making the Dialog
to open the .pptx file.
```python 
from PySide import QtCore, QtGui
from PySide.QtGui import (QApplication, QMainWindow, QAction, QWidget,
    QGraphicsScene, QGraphicsView,QFileDialog)
import ntpath
import os
import win32com.client
import time
import ctypes.wintypes
CSIDL_PERSONAL = 5       
SHGFP_TYPE_CURRENT = 0   
import speech_recognition as sr 
```

Next we need to point the OS towards the Documents folder by default for this we will use the code below:

```python
buf= ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf)
if not os.path.exists((buf.value)+'\\'):
   os.makedirs((buf.value)+'\\')
   pathProjects = (buf.value)+'\\'
```
Now lets define our first function to get the name of the .pptx file from the full path.

```python
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
```
After this we need to define the Class for the Dialog, buttons and the functions.

```python
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1086, 206)
        self.verticalLayout_3 = QtGui.QVBoxLayout(Dialog)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
      
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.lineEdit_4 = ""
        self.FlagAccept = False
       
        self.lineEdit_2 = QtGui.QLineEdit(Dialog)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.verticalLayout.addWidget(self.lineEdit_2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        self.pushButton = QtGui.QPushButton(Dialog)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_3.addWidget(self.buttonBox)
        
        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.accepted.connect(self.checkOK)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.pushButton.clicked.connect(self.getProjectName)
        self.projectName = pathProjects
        self.lineEdit_2.setText(self.projectName)
        self.lineEdit_2.textEdited.connect(self.textEdited)
        

        self.Dialog =Dialog
    def checkOK(self):
    
        self.FlagAccept = True
        app = win32com.client.Dispatch("PowerPoint.Application")
        presentation = app.Presentations.Open(FileName=self.projectName, ReadOnly=1)
        presentation.SlideShowSettings.Run()
        while(1):
            print('rr')
            r = sr.Recognizer()
            with sr.Microphone()as source:
                print('Say Something')
                audio = r.listen(source)
                
                print('Done')
                try:
                    text = r.recognize_sphinx(audio)
                    print(text)
                    if text.find('next')!=-1:
                    
                        presentation.SlideShowWindow.View.Next()
                        print('sliding....')
                    elif text.find('bingo')!=-1:
                        presentation.SlideShowWindow.View.Previous()
                    else:
                        pass
                    
                except:
                    print('error')

            
         

            # presentation.SlideShowWindow.View.Exit()
            # app.Quit()
                    
        
    def getProjectName(self):
       
   
        projectName = QFileDialog.getOpenFileName(filter="Data (*.pptx)")
        
        print("project name: ",projectName[0])
        self.projectName = projectName[0]
        self.lineEdit_2.setText(self.projectName)
      
        self.Dialog.setWindowTitle((str(path_leaf(projectName[0]))))
        textIs=(self.lineEdit_2.text())
        self.Dialog.setWindowTitle(path_leaf(projectName[0]))
        if textIs !="":
            self.buttonBox.setEnabled(True)
            
        else:
            self.buttonBox.setEnabled(False)
        
    
        
        
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Open Presentation"))
        self.pushButton.setText(_translate("Dialog", "Browse..."))
        self.label_2.setText(_translate("Dialog", "Location: "))
    def textEdited(self):
        self.Dialog.setWindowTitle(path_leaf(self.lineEdit_2.text()))
```

And in the last section we need to call this class and run the app
```python
if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Dialog = QtGui.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
```
    
## ScreenShot:
Save all the code in a app.py file and run using `python app.py`. 

Now chose the .pptx file and hit OK button. After this the ppt will launch in full screen mode. You can say next slide, or next, in this portion of the code the text is scanned for next word and if it is detected correctly, the slide will turn to 
the next page and similarly, if the word is bingo then it will go back one page. 
```python
if text.find('next')!=-1:
    presentation.SlideShowWindow.View.Next()
    print('sliding....')
elif text.find('bingo')!=-1:
    presentation.SlideShowWindow.View.Previous()
```
Even though it is a simple app but it can be enchance with different options. Have a nice day! Leave comments if you have question regarding this post.


