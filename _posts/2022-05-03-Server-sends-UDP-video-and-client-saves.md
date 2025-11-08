---
categories:
- Socket Programming Series
description: This tutorial is about using OpenCv and UDP sockets for server-client video transfer and saving MP4 at client
featured-img: pythonudp1
keywords:
- UDP
- Video Streaming
- OpenCV
- Python
- Socket Programming
layout: post
mathjax: true
tags:
- UDP
- Video Streaming
- Python
- OpenCV
title: How to send video over UDP socket and save it as MP4 at...
---




Hello friends! Previously, we have seen how TCP and UDP sockets work. This tutorial is about sending the video frames over UDP from server and then saving them 
at the client side as .MP4 file. 

# server.py
{% include codeHeader.html %}
```python

## This is server code to send video frames over UDP so that client can save it
import cv2, imutils, socket
import numpy as np
import time
import base64

BUFF_SIZE = 65536
server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()
host_ip = '192.168.10.113'#  socket.gethostbyname(host_name)
print(host_ip)
port = 9999
socket_address = (host_ip,port)
server_socket.bind(socket_address)
print('Listening at:',socket_address)

vid = cv2.VideoCapture(0) #  replace 'rocket.mp4' with 0 for webcam
fps,st,frames_to_count,cnt = (0,0,20,0)

while True:
	msg,client_addr = server_socket.recvfrom(BUFF_SIZE)
	print('GOT connection from ',client_addr)
	WIDTH=400
	while(vid.isOpened()):
		_,frame = vid.read()
		frame = imutils.resize(frame,width=WIDTH)
		encoded,buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
		message = base64.b64encode(buffer)
		server_socket.sendto(message,client_addr)
		frame = cv2.putText(frame,'FPS: '+str(fps),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
		cv2.imshow('TRANSMITTING VIDEO',frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			server_socket.close()
			break
		if cnt == frames_to_count:
			try:
				fps = round(frames_to_count/(time.time()-st))
				st=time.time()
				cnt=0
			except:
				pass
		cnt+=1
```

## client.py
{% include codeHeader.html %}
```python

## This is client code to receive video frames over UDP and save as .MP4 file
import cv2, imutils, socket
import numpy as np
import time
import base64

from datetime import datetime
fourcc =0x7634706d 
now = datetime.now()
time_str = now.strftime("%d%m%Y%H%M%S")
time_name = '_Rec_'+time_str+'.mp4'
FPS = 30
frame_shape = False


BUFF_SIZE = 65536
client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()
host_ip = '192.168.10.113'#  socket.gethostbyname(host_name)
print(host_ip)
port = 9999
message = b'Hello'

client_socket.sendto(message,(host_ip,port))
fps,st,frames_to_count,cnt = (0,0,20,0)
while True:
    packet,_ = client_socket.recvfrom(BUFF_SIZE)
    data = base64.b64decode(packet,' /')
    npdata = np.fromstring(data,dtype=np.uint8)
    frame = cv2.imdecode(npdata,1)
    frame = cv2.putText(frame,'FPS: '+str(fps),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    
    if cnt == frames_to_count:
        try:
            fps = round(frames_to_count/(time.time()-st))
            st=time.time()
            cnt=0
        except:
            pass
    cnt+=1
    if not frame_shape:
        video_file_name  = str(host_ip) + time_name
        out = cv2.VideoWriter(video_file_name, fourcc, FPS, (frame.shape[1], frame.shape[0]), True)
        frame_shape = True
    out.write(frame)
    cv2.imshow("RECEIVING VIDEO",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        
        break
client_socket.close()
out.release()
```
Please note that for MacOS we require:
```
sudo sysctl -w net.inet.udp.maxdgram=65535
```
If restart computer the above UDP max buffer size will shrink to 9216. So please run the above command again if required.
