---
layout: post
title: How to publish-subscribe video using socket programming in Python
categories: [Socket programming tutorial series]
mathjax: true
featured-img: pubsub
description: This tutorial is short and simple implementation of server client modules in Publish/Subscribe mode to transfer video frames
keywords: [Python, Socket Programming, Publish-Subscribe, ZeroMQ, Video Streaming]
tags: [Python, Socket Programming, Publish-Subscribe, ZeroMQ, Video Streaming]
---
Hi friends HAPPY NEW YEAR 2021! In a previous tutorial we used opencv to obtain video frames of webcam and send them over wifi to server/client. Below is the video about basics of socket 
programming.

<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/7-O7yeO3hNQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
</iframe>
</div>
<br>

Today, we will use a rather simple way to transfer video over wifi using Publish/Subscribe mode by leveraging sockets from the ZMQ library. The frame rate
will be displayed on the client window, and a server will transmit (publish) the video. Similar to our previous videos, we assume that you 
will find your computer's IP address for the wifi. Both server and client computers should be connected to the same wifi router.

### ZMQ
ZeroMQ (also known as ØMQ, 0MQ, or zmq) is like an embeddable networking library but it acts like a concurrency framework. It gives us sockets
that carry atomic messages across various transports like in-process, inter-process, TCP, and multicast. We can connect sockets N-to-N with patterns 
like fan-out, pub-sub, task distribution, and request-reply. It's fast enough to be the fabric for clustered products. Its asynchronous I/O model gives 
us scalable multicore applications, built as asynchronous message-processing tasks. 

The new thing about this tutorial is that you will observe faster frame rate (Above 30 FPS) especially in video transfer mode. In the camera mode, the
inherent frame rate of webcam will decide the bottleneck of the FPS.

So let's install the essentials! (Please use Python 3)

```
pip3 install pyshine==0.0.6
pip3 install zmq
```

In the server.py code, please change the address in server_socket.bind("tcp://192.168.1.105:5555") to server_socket.bind("tcp://your_computer_ip:5555").
The default mode is set to Camera. To transmit video place a folder named videos in the same directory as the server.py and give the path
of the .mp4 file accordingly.

We will use pyshine_video_queue(vid) function to obtain a queue of size 10. This queue will continue to acquire the frames from the webcam
in a separate thread. Then a separate while loop is used to obtain and send each frame from this queue. 

### server.py
{% include codeHeader.html %}
```python
import cv2,imutils
import zmq
import base64,time
import queue,threading
# www.pyshine.com
context = zmq.Context()
server_socket = context.socket(zmq.PUB)
server_socket.bind("tcp://192.168.1.105:5555")
camera = True
if camera == True:
	vid = cv2.VideoCapture(0)
else:
	vid = cv2.VideoCapture('videos/mario.mp4')

def pyshine_video_queue(vid):
	
	frame = [0]
	q = queue.Queue(maxsize=10)
	def getAudio():
		while (vid.isOpened()):
			try:
				img, frame = vid.read()
				frame = imutils.resize(frame,width=640)
				q.put(frame)
			except:
				pass
			
	thread = threading.Thread(target=getAudio, args=())
	thread.start()
	return q

q = pyshine_video_queue(vid)

while True:
	frame = q.get()
	encoded, buffer = cv2.imencode('.jpg', frame,[cv2.IMWRITE_JPEG_QUALITY,80])
	data = base64.b64encode(buffer)
	print(server_socket.send(data))
	cv2.imshow("server image", frame)
	key = cv2.waitKey(1) & 0xFF
	time.sleep(0.01)
	if key  == ord('q'):
		break

vid.release()
cv2.destroyAllWindows()
  
		

```
Here is code for the client. Please give the IP address of your server accordingly in client_socket.connect("tcp://192.168.1.105:5555"), 
following the similar way as in server.py code.

### client.py
{% include codeHeader.html %}
```python
import cv2
import zmq
import base64
import numpy as np,time
import pyshine as ps
# www.pyshine.com
context = zmq.Context()
client_socket = context.socket(zmq.SUB)
client_socket.connect("tcp://192.168.1.105:5555")
client_socket.setsockopt_string(zmq.SUBSCRIBE,optval='')
fps=0
st=0
frames_to_count=20
cnt=0
while True:
    if cnt == frames_to_count:
        try:

            fps = round(frames_to_count/(time.time()-st))
            st = time.time()
            cnt=0
        except:
            pass
    cnt+=1
    frame = client_socket.recv()
    img = base64.b64decode(frame)
    npimg = np.fromstring(img, dtype=np.uint8)
    source = cv2.imdecode(npimg, 1)
    text  =  'FPS: '+str(fps)
    source = ps.putBText(source,text,text_offset_x=20,text_offset_y=30,background_RGB=(10,20,222))
    time.sleep(0.01)
    cv2.imshow("client image", source)
    key = cv2.waitKey(1) & 0xFF
    if key  == ord('q'):
        break
cv2.destroyAllWindows()



```
Unlike our previous tutorials about socket programming, in this tutorial you can run either of the server and client.py as the first code. A server will send frames to non-existent peers; no errors will be generated; instead, they'll queue up in socket buffers based on the configuration.

To run the code, on server side:
```
python3 server.py
```
The client starts like this:

```
python3 client.py

```

Thats it! If you have questions, suggestions please do comment. Have a nice day!
