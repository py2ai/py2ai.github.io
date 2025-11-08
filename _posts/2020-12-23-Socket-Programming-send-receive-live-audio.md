---
categories:
- GUI tutorial series
description: This tutorial is about sending and receiving audio data over wifi between server and client.
featured-img: audio_socket
keywords:
- Socket Programming
- Audio Streaming
- Pyshine
- Python
- WiFi Communication
layout: post
mathjax: true
tags:
- Socket Programming
- Audio Streaming
- Pyshine
- Python
- WiFi Communication
title: How to send and receive live audio using socket...
---



Hi friends! In a previous tutorial we used opencv to obtain video frames of webcam and send them over wifi to server/client. Below is the video about basics of socket 
programming.

<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/7-O7yeO3hNQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
</iframe>
</div>
<br>

Today, we will move one step further, instead of transmitting video frames over wifi from one computer to another, we will use pyshine to send audio frames. The
default audio frame will be of 1024 data samples for each audio channel from the microphone device of a computer. The mechanism is almost similar to that of video 
transmitting and receiving. So without any delay, let's install the essentials.

Install pyshine version 0.0.6 in Windows OS as:

```
pip3 install pyshine==0.0.6
```

Both server and client computers should be on the same wifi router. The required IP address will be for Wifi LAN (inet)

# Windows OS users

From the cmd window run this command:

```
ipconfig
```

The required IP address will be shown against IPv4 Address

Here is the server side code. First, please change the IP address: '192.168.1.105' to yours, otherwise your server will not start.


## # server.py

```python
import socket, cv2, pickle,struct,time
import pyshine as ps

mode =  'send'
name = 'SERVER TRANSMITTING AUDIO'
audio,context= ps.audioCapture(mode=mode)
## # ps.showPlot(context,name)

## # Socket Create
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '192.168.1.105'
port = 4982
backlog = 5
socket_address = (host_ip,port)
print('STARTING SERVER AT',socket_address,'...')
server_socket.bind(socket_address)
server_socket.listen(backlog)

while True:
	client_socket,addr = server_socket.accept()
	print('GOT CONNECTION FROM:',addr)
	if client_socket:

		while(True):
			frame = audio.get()
			
			a = pickle.dumps(frame)
			message = struct.pack("Q",len(a))+a
			client_socket.sendall(message)
			
	else:
		break

client_socket.close()			

```
Here is code for the client. Please give the IP address of your server accordingly.

## # client.py
```python
import socket,cv2, pickle,struct
import pyshine as ps

mode =  'get'
name = 'CLIENT RECEIVING AUDIO'
audio,context = ps.audioCapture(mode=mode)
ps.showPlot(context,name)

## # create socket
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '192.168.1.105'
port = 4982

socket_address = (host_ip,port)
client_socket.connect(socket_address) 
print("CLIENT CONNECTED TO",socket_address)
data = b""
payload_size = struct.calcsize("Q")
while True:
	while len(data) < payload_size:
		packet = client_socket.recv(4*1024) # 4K
		if not packet: break
		data+=packet
	packed_msg_size = data[:payload_size]
	data = data[payload_size:]
	msg_size = struct.unpack("Q",packed_msg_size)[0]
	
	while len(data) < msg_size:
		data += client_socket.recv(4*1024)
	frame_data = data[:msg_size]
	data  = data[msg_size:]
	frame = pickle.loads(frame_data)
	audio.put(frame)

client_socket.close()


```
To run the code, on server side:
```
python3 server.py
```
Once the server starts listening

```
python3 client.py

```

Thats it! If you have questions, suggestions please do comment. Have a nice day!
