---
layout: post
title: How to install OpenCV and Python in windows 
categories: [tutorial]
mathjax: true
featured-img: opencvandpythoninstall
description: A quick tutorial to install python and opencv in windows7
---




These steps are tested on windows 7.
1. Please download following Python packages and install them to their default locations
[Python 2.7]. Also make the environment variable as name `Path` with value set to `C:\Python27;%PYTHON_HOME%\;%PYTHON_HOME%\Scripts\;`

2. Please find pip installer python file named `get-pip.py` from here [get-pip.py].

3. Press shift button and right click anywhere in the folder and select “Open command window here” and enter:
`python get-pip.py`.

4. The pip installer will be installed and the go to `C:\Python27\Scripts` and again while holding the shift button in the Scripts folder right click to open the command window here. Enter the command `pip install dateutils`.

5. Stay in the same command prompt and also install `matplotlib` by using the command `pip install matplotlib`.

6. In addition we also need imutils , so again enter the command `pip install imutils`.
 
7. Perform the command `pip install pytesseract==0.1.8`.

8. The [Tesseract-OCR] setup is also required. Go ahead and install using the defaults values.

9. An additional command has to be added in the python `.py` file, such as:
`pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'`.

10. Now, we need to install [opencv].

11. Extract the opencv to C: drive and after the extraction finishes , Go to the location `C:\opencv\build\python\2.7\x86`.

12. And copy the file `cv2.pyd` from the location `C:\opencv34\opencv\build\python\2.7\x86` and paste it to the location `C:/Python27/Lib/site-packages`.

13. Now check the installation by 

`python`

`import cv2`

`print(cv2.__version__)`

The output will be 2.4.9. Congratulations! you have now successfully setup opencv and python.



[get-pip.py]:https://github.com/py2ai/py2ai.github.io/raw/master/assets/files/get-pip.py
[Python 2.7]:http://python.org/ftp/python/2.7.5/python-2.7.5.msi
[Tesseract-OCR]:https://sourceforge.net/projects/tesseract-ocr-alt/files/tesseract-ocr-setup-3.02.02.exe/download
[opencv]:http://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.9/opencv-2.4.9.exe/download
