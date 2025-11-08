---
categories:
- Tutorial Series
description: This tutorial is about streaming multiple videos on a webpage using PyShine server
featured-img: livehtmlpagetwo
keywords:
- stream multiple videos on HTML
- PyShine video streaming
- OpenCV video streaming
- Python multiprocessing
- HTML video streaming tutorial
- web development with video
- PyShine server tutorial
layout: post
mathjax: true
tags:
- Streaming
- HTML
- PyShine
- OpenCV
- Multiprocessing
- Video Streaming
- Web Development
- Tutorial
title: How to stream multiple videos on an HTML webpage
---

<br>
Hi friends, hope your are doing great. This tutorial is about streaming multiple videos on a webpage. As an example, we will use two .mp4 files. We will use two processes to stream each video feed. In the following code, we will use two functions: 

1) First video feed (either webcam or mp4 file), 
2) Second video feed, either webcam or mp4 file).


[![GIF](https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/posts/multiwebpagevideos.gif?raw=true)](https://youtu.be/vt6Fu-Rp-h0 "GIF")

You will need to install pyshine and opencv libraries. Pyshine can be installed as as:
pip3 install pyshine==0.0.9

In Python 3, the multiprocessing is a package that supports spawning processes using an API similar to the threading module. 
The multiprocessing package offers both local and remote concurrency, effectively side-stepping the Global Interpreter Lock by using subprocesses 
instead of threads. The multiprocessing module allows us to fully leverage multiple processors on a given machine. 
It runs on multiple Operating systems including Windows. Our PyShine streamer will server each video on a given IP address and ports. 
The key idea here is to route the video feeds on the same server's same IP but with different port numbers. 

So let say if you want to stream four videos from four webcams connected to your server PC, you need to run four functions, each on a different port number. 
To further explain this concept, let's look at the complete code for two video case.

# run.py
{% include codeHeader.html %}
```python

import cv2
import  pyshine as ps #  pip3 install pyshine==0.0.9
from multiprocessing import Process
import sys
HTML="""
<html>
<head>
<title>PyShine Live Streaming</title>
</head>

<body>
<center><h1> PyShine Live Streaming Multiple videos </h1></center>
<center><img src="http://10.211.55.27:9000/stream.mjpg" width='360' height='240' autoplay playsinline></center>
<br>
<center><img src="http://10.211.55.27:9001/stream.mjpg" width='360' height='240' autoplay playsinline></center>
</body>
</html>
"""
def main1():
    StreamProps = ps.StreamProps
    StreamProps.set_Page(StreamProps,HTML)
    address = ('10.211.55.27',9000) # Enter your IP address 
    try:
        StreamProps.set_Mode(StreamProps,'cv2')
        capture = cv2.VideoCapture(0) # replace 0 (webcam id) with the path of your .mp4 video file
        capture.set(cv2.CAP_PROP_BUFFERSIZE,4)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,320)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
        capture.set(cv2.CAP_PROP_FPS,30)
        StreamProps.set_Capture(StreamProps,capture)
        StreamProps.set_Quality(StreamProps,90)
        server = ps.Streamer(address,StreamProps)
        print('Server started at','http://'+address[0]+':'+str(address[1]))
        server.serve_forever()
        print('done')
        
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()

def main2():
    StreamProps = ps.StreamProps
    StreamProps.set_Page(StreamProps,HTML)
    address = ('10.211.55.27',9001) # Enter your IP address 
    try:
        StreamProps.set_Mode(StreamProps,'cv2')
        capture = cv2.VideoCapture(1) # replace 1 (webcam id) with the path of your .mp4 for video file
        capture.set(cv2.CAP_PROP_BUFFERSIZE,4)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,320)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
        capture.set(cv2.CAP_PROP_FPS,30)
        StreamProps.set_Capture(StreamProps,capture)
        StreamProps.set_Quality(StreamProps,90)
        server = ps.Streamer(address,StreamProps)
        print('Server started at','http://'+address[0]+':'+str(address[1]))
        server.serve_forever()
        print('done')
        
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()        
        
if __name__=='__main__':
    p1 = Process(target=main1)
    p1.start()
    p2 = Process(target=main2)
    p2.start()
```    

In the above code, please change your PC/server IP's address:

1. In the HTML docstring section change the first source link ```src="http://10.211.55.27:9000/stream.mjpg"``` to ```src="http://your.ip.address.please:port_number/stream.mjpg"```
2. Again for the second video feed, you need to change the HTML docstring address ```src="http://10.211.55.27:9001/stream.mjpg"``` to ```src="http://your.ip.address.please:port_number+1/stream.mjpg"```
3. In the ```main1()``` and ```main2()``` function replace your path of video file or capturing device id accordingly, and of course the IP address.
4. Thats it! now execute the code as ```python3 run.py``` and after that in a browser enter the address as ```http://10.211.55.27:9000``` (again in your case this address will be printed differently so please paste that in the browser), you will see the output webpage showing both streams at the same time.




