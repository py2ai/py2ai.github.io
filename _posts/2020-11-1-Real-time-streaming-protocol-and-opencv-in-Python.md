---
layout: post
title: OpenCV and Real time streaming protocol (RTSP)
categories: [Networking tutorials]
mathjax: true
featured-img: rtsp
summary: How to obtain video frames from an RTSP stream of video
---

Hello friends, this tutorial is about RTSP stream basics, how to process it, and obtain frames in Python. In general, OpenCV is used with webcams connected to computers or also embedded inside them. However, for the surveillance purpose, we commonly use IP cameras that generate video streams using
RTSP protocol. We can use such IP cameras in projects of video processing, like motion detection, etc. So let's start by knowing what RTSP is.

### RTSP

It is a network control protocol to deliver multimedia content across the desired networks. With TCP, it provides a reliable stream, and its structure is similar to the HTTP. It was designed mainly for entertainment and communications systems to access and control streaming media servers. The clients can issue commands
to the media server like play, record, pause, etc. to enable real-time control of the media servers. RTSP allows for video-on-demand solutions.

Lets try to open some rtsp and http links:

1. "rtsp://freja.hiof.no:1935/rtplive/definst/hessdalen03.stream"
2. "http://wmccpinetop.axiscam.net/mjpg/video.mjpg"

Here is the code to obtain frames of video from the rtsp link. 

```python

import cv2
# vid = cv2.VideoCapture(0) # For webcam
vid = cv2.VideoCapture("http://wmccpinetop.axiscam.net/mjpg/video.mjpg") # For streaming links
while True:
  _,frame = vid.read()
  print(frame)
  cv2.imshow('Video Live IP cam',frame)
  key = cv2.waitKey(1) & 0xFF
  if key ==ord('q'):
    break

vid.release()
cv2.destroyAllWindows()

```

The general syntax of rtsp stream for IP cameras are like:

rtsp://UserName:Password@IpAdress:Port/Streaming/Channels/ChannelID

According to your settings, give the parameters in the link and use in your opencv projects. 
