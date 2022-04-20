---
layout: post
title: FAQs about PyQt5
categories: [GUI tutorial series]
mathjax: true
summary: You can find important issues and their solutions related to PyQt5 here
---

Hi friends, we are going to start a Q and A about PyQt5 here. This page will be dedicated to only Questions that are frequently asked by you and their answers.
We will continue to update this page accordingly.

### Question: I can't find the Qt designer app in windows? I just typed designer in cmd but have this error:
```
designer : The term 'designer' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or if a path was included, verify 
that the path is correct and try again.
At line:1 char:1
+ designer
+ ~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (designer:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
```
### Answer:
Unfortunatley, in newer versions, the PyQt5 wheels do not provide tools such as Qt Designer that were included in the old binary installers. But the good news is that you can install it via 

`pip install pyqt5-tools~=5.15`

Once installed you can simple type the following to run the Qt designer

```
pyqt5-tools designer
```

### Question: I installed PyQt5 via pip and can't find the "3" folder or the designer executable. How did you install it?

### Answer
```
pip3 install pyqt5
pip3 install pyqt5-tools
After that simply run the command below in Terminal:

qt5-tools designer
```
### Question: Thank you, these are great tutorials. I do have a problem with getting the main window contents to resize with the main window when I run the app. Can you please tell me what I am doing wrong?

### Answer: 
Try Python3.6.5 and pip3 install PyQt5==5.15.2

### Question: How did you align the second row of buttons (keyboard shortcut?)

### Answer: 
Select and Click on Grid layout icon

### Question: Is it possible to store all the new functions in a separately .py-file? I've tried this, but afterwards the GUI does not run properly... What can I do, if I want to change something in the Qt-Designer but also to keep my written functions?

### Answer:
Hi! You're welcome and thanks for asking this nice question. Yes there is a way to keep the working gui with functions intact and also upgrade GUI. For this purpose you need 

from PyQt5 import uic
self.ui = uic.loadUi('main.ui',self)
and simply go on use the once created main.py as it is with more functions to add. All you need is open and edit the .ui file and save it without changing it to . py file. You can see the example    
here https://pyshine.com/PyQt5-Live-Audio-GUI-with-Start-and-Stop/

### Question: Thank you very much for your video but I have a problem in my project where I use grubCut function for image in label, the problem is exactly when using 'cv2.setMouseCallback', we are forced to create an exterior window (cv2.nameWindow) while the modifications must be done directly on the label. Please help me

### Answer:
Hi! You're welcome and thanks for reaching out. Please follow this tutorial where mouse events are used in the image in label https://youtu.be/lGeM3lSdwRM

### Question: Hey, you have done a great job! Can you provide a link to your source code? I'm also building a pyqt5 application and want to use interactive map for timeseries data. Your video inspired me a lot :)

### Answer: 
Hello! So glad to hear it. You can find the source code in the description under video. https://pyshine.com/Make-GUI-With-Matplotlib-And-PyQt5/

### Question: Hello, this is a great job! This video helped me a lot to do my project! Thank You so much. Can you please plot a csv file data using scatter as well in pyqt5?

### Answer: 
You're welcome and thanks for pointing out, we can use plt.scatter function

### Question: Hi, thanks for the continued support! The lessons learned from this videos are very useful. This video shows how to integrate matplotlib into a PyQt5 environment.. Is is possible in addition to integrate matplotplib widgets? Specifically, the RangeSlider widget:   from matplotlib.widgets import RangeSlider.  I almost got it done using your code as a template, however, the mouse events aren't passed on to the RangeSlider. Any insights appreciated.

### Answer:
Hi, Yes its possible, Multiple Matplotlib widgets can be added in the same way as in the above video. For slider input the widget can be added in a way similar to the video related to Video processing GUI at pyshine.com

### Question: Thanks for this great tutorial. You have done a good job. I have only one problem. When I open a CSV file, it plots alright in the window but it pops out another window titled Figure 1, but it has no plot in it. What can be the problem?

### Answer: 
Hi! most welcome! Please use Matplotlib version 3.2.1 and Python 3.6.5 . Also which OS you are using? You can install the matplotlib as: pip3 install matplotlib==3.2.1

If your using spyder IDE: then it's a spyder IDE problem just add  plt.ioff() in the code will solve it.

### Question: Nice video, I actually have a problem. After clicking a csv file, I'm not getting the csv file imported, what did I miss, I wrote whole code by watching it

### Answer:
Hi! Please copy the code from https://pyshine.com/Make-GUI-With-Matplotlib-And-PyQt5/

Also it is highly recommended to install matplotlib version 3.2.1

### Question: ImportError: cannot import name 'Qtcore' from 'PyQt5' I am getting this error.  I've already install pyqt5
```Traceback (most recent call last):
  File "C:\Users\Ketan\Desktop\Kedarnath\maingui.py", line 17, in <module>
    from PyQt5 import Qtcore, QtWidgets
ImportError: cannot import name 'Qtcore' from 'PyQt5' (C:\Users\Ketan\AppData\Local\Programs\Python\Python39\lib\site-packages\PyQt5\__init__.py)
```
### Answer:
You are using Python3.9, there might be some issue with it. Try using Python 3.6.5 with PyQt5.  `pip3 install PyQt5==5.15.2`
