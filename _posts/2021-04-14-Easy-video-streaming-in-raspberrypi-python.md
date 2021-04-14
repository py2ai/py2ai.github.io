---
layout: post
title: How to easily stream webcam video over wifi with Raspberry Pi
categories: [GUI tutorial series]
mathjax: true
featured-img: jeep01
summary:  This tutorial is about streaming webcam video to an HTML page without Flask
---

Hi friends! Today's tutorial is Part 01 of the Raspberry Pi learning series. You will learn how to transmit video that from OpenCV library over the wifi. This video stream can be received on any mobile device connected to wifi and can open up an HTML webpage. Interestingly, in this tutorial, we do not require Flask or Django. All we need are two libraries; 1) cv2 and 2) pyshine.

### Installation 
1) cv2 
You can easily install opencv as cv2 by following the tutorial here:
`https://pyshine.com/How-to-install-OpenCV-in-Rasspberry-Pi/`
2) pyshine
pyshine is required to enable the streaming features in this tutorial. So you can use the latest version 0.0.9 as:
```pip3 install pyshine==0.0.9```

Please note that you can easily observe the current IP address of the Raspberry Pi. Although the code below will work on any PC connected to the same wifi router, we recommend using RPi for its numerous advantages in daily life applications. 

We will use raspberry pi zero board in this tutorial, but theoretically, you can use any other boards as long as they have the Python3 installed and access wifi. The wifi standard used in RPi zero is IEEE 802.11n, which relatively supports a longer distance of about 200 m in Line of sight theoretically. However, the range also depends on antenna characteristics and, of course, the transmission power. We will cover these details in upcoming tutorials so let us get back to the streaming part.

For this purpose, we need a `main.py` code as shown below, which will use the docstring of an HTML page and pass it to the streaming server through pyshine library. The HTML page will get the mjpeg stream plays it in an inline fashion.

From our previous tutorials, you may already have known the idea to get the IP address of your device. Let's say your IP address is `192.168.1.1`, then the following code is all we need to observe the video stream in another device's webbrowser.

### main.py
```python
# Part 01 using opencv access webcam and transmit the video in HTML
import cv2
import  pyshine as ps #  pip3 install pyshine==0.0.9
HTML="""
<html>
<head>
<title>PyShine Live Streaming</title>
</head>

<body>
<center><h1> PyShine Live Streaming using OpenCV </h1></center>
<center><img src="stream.mjpg" width='640' height='480' autoplay playsinline></center>
</body>
</html>
"""
def main():
    StreamProps = ps.StreamProps
    StreamProps.set_Page(StreamProps,HTML)
    address = ('192.168.1.1',9000) # Enter your IP address 
    try:
        StreamProps.set_Mode(StreamProps,'cv2')
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_BUFFERSIZE,4)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,320)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
        capture.set(cv2.CAP_PROP_FPS,30)
        StreamProps.set_Capture(StreamProps,capture)
        StreamProps.set_Quality(StreamProps,90)
        server = ps.Streamer(address,StreamProps)
        print('Server started at','http://'+address[0]+':'+str(address[1]))
        server.serve_forever()
        
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()
        
if __name__=='__main__':
    main()
    
```

Copy this code to a main.py file and paste that file in the Raspberry pi that already has camera enabled.
To run the code:

```python3 main.py```

After that you will see these messages in terminal window:

```
Warning! PortAudio library not found
Warning! No module named 'matplotlib'
Warning! No module named 'keras'
Server started at http://192.168.1.1:9000

```
Ignore the warnings and copy the address `http://192.168.1.1:9000` and 
paste it to the browser of your pc or mobile device that is on the same wifi network. And there you go, now you can view in the browser what the camera of 
your RPi is providing to you.

But wait a minute, what if you don't have a wifi router and you want to connect RPi in Ad Hoc mode. The whole idea is to configure RPI's wifi interface in Adhoc mode, so that it can be connected to a PC/Mobile device without any wifi router. Yes, in this mode we can enjoy First Person View (FPV) capabilities of RPi.  By simply writing an auto run script in Python that will be called once the RPi is ever booted up, we can run the above code in it. In this way, you can also enjoy point to point FPV link, where the video from RPi is transmitted straight to your mobile device.
The RPi zero is only 9g in weight, which means you can use it with drones as well. The frame size is kept 320x240 to reduce the latency, which is essential for FPV based vehicles/drones. To configure the Raspberry pi in Adhoc mode please follow our upcoming tutorial.

Have a nice day!







