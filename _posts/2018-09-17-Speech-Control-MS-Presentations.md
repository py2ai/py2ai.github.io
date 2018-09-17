---
layout: post
title: Speech controling the MS Power Point Presentation
author: Hussain A.
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

The import section of the Python code is shown here:
```python 
from PySide import QtCore, QtGui, QtGui
from PySide.QtGui import (QApplication, QMainWindow, QAction, QWidget,
    QGraphicsScene, QGraphicsView,QFileDialog)
import ntpath
import os
import win32com.client
import time
import ctypes.wintypes
CSIDL_PERSONAL = 5       
SHGFP_TYPE_CURRENT = 0   
import threading

import speech_recognition as sr


