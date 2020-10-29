---
layout: post
title: A simple cache-server to broadcast video to clients
categories: [GUI tutorial series]
mathjax: true
featured-img: drone
summary: How to broadcast drone video to multiple clients using a cache-server in Python
---

[![GIF](https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/posts/ydrone.png?raw=true)](https://youtu.be/ZyFYBawiA24 "GIF")

Hello friends, today we will write three codes as described in the above video:
1. drone.py
2. cache-server.py
3. client.py

You can test them on a single or multiple computers. One for the drone, such as Raspberry Pi, or any platform with Python. The cache-server.py code will do the 
heavy lifting here, because increasing the number of threads, demands more resources at the CPU end. So its better to use a multicore PC for the cache server implementation.
The client side pc will get the video from the cache server. It is assumed in this tutorial that all devices have access to the local Wifi network. To find out the 
IP address of each pc use the one against the Wifi adapter, the procedure for various platforms is here:

### MAC OS users

Go to the terminal window and run this command:

```
ipconfig getifaddr en0

```
That will show your LAN IP address. Note that en0 is commonly used for ethernet interface, and en1 is for the Airport interface. Make sure that your IP address is not starting from 127.x.x.x because that is your local host, and if you only want to check server client for the same pc then it is fine. Otherwise, consider use the command above and write the correct ip address for video transfer over different machines. 


### Linux/Ubuntu OS users

From the terminal window run this command:

```
ifconfig
```
The require IP address will be for Wifi LAN (inet)

### Windows OS users

From the cmd window run this command:

```
ipconfig
```

The require IP address will show against IPv4 Address.

Here are the respective codes:

### drone.py

```python
# This code will run the drone side, it will send video to cache server
# Lets import the libraries
# Welcome to PyShine
# www.pyshine.com
import socket, cv2, pickle, struct
import imutils
import cv2


server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_name  = socket.gethostname()
host_ip = '192.168.79.102' # Enter the Drone IP address
print('HOST IP:',host_ip)
port = 9999
socket_address = (host_ip,port)
server_socket.bind(socket_address)
server_socket.listen()
print("Listening at",socket_address)

def start_video_stream():
	client_socket,addr = server_socket.accept()
	camera = True
	if camera == True:
		vid = cv2.VideoCapture(0)
	else:
		vid = cv2.VideoCapture('videos/boat.mp4')
	try:
		print('CLIENT {} CONNECTED!'.format(addr))
		if client_socket:
			while(vid.isOpened()):
				img,frame = vid.read()

				frame  = imutils.resize(frame,width=320)
				a = pickle.dumps(frame)
				message = struct.pack("Q",len(a))+a
				client_socket.sendall(message)
				cv2.imshow("TRANSMITTING TO CACHE SERVER",frame)
				key = cv2.waitKey(1) & 0xFF
				if key ==ord('q'):
					client_socket.close()
					break

	except Exception as e:
		print(f"CACHE SERVER {addr} DISCONNECTED")
		pass

while True:
	start_video_stream()


```

### cache-server.py

```python
# Cache server will recieve video stream from the the drone camera
# Also it will serve this video stream to multiple clients 
# Lets import the libraries
# Welcome to PyShine
# www.pyshine.com
import socket, cv2, pickle, struct
import imutils # pip install imutils
import threading
import cv2


server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_name  = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('HOST IP:',host_ip)
port = 9999
socket_address = (host_ip,port)
server_socket.bind(socket_address)
server_socket.listen()
print("Listening at",socket_address)

global frame
frame = None

def start_video_stream():
	global frame
	client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	host_ip = '192.168.79.102' # Here provide Drone IP 
	port = 9999
	client_socket.connect((host_ip,port))
	data = b""
	payload_size = struct.calcsize("Q")
	while True:
		while len(data) < payload_size:
			packet = client_socket.recv(4*1024) 
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
		cv2.imshow("RECEIVING VIDEO FROM DRONE",frame)
		key = cv2.waitKey(1) & 0xFF
		print(data)
		if key  == ord('q'):
			break
	client_socket.close()
	

thread = threading.Thread(target=start_video_stream, args=())
thread.start()

def serve_client(addr,client_socket):
	global frame
	try:
		print('CLIENT {} CONNECTED!'.format(addr))
		if client_socket:
			while True:
				a = pickle.dumps(frame)
				message = struct.pack("Q",len(a))+a
				client_socket.sendall(message)
				
	except Exception as e:
		print(f"CLINET {addr} DISCONNECTED")
		pass

   
while True:
	client_socket,addr = server_socket.accept()
	print(addr)
	thread = threading.Thread(target=serve_client, args=(addr,client_socket))
	thread.start()
	print("TOTAL CLIENTS ",threading.activeCount() - 2) # edited here because one thread is already started before


```

### client.py

```python
# Welcome to PyShine
# lets make the client code
# Welcome to PyShine
# www.pyshine.com
import socket,cv2, pickle,struct

# create socket
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '192.168.124.15' # Here Require CACHE Server IP
port = 9999
client_socket.connect((host_ip,port)) # a tuple
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
	cv2.imshow("RECEIVING VIDEO FROM CACHE SERVER",frame)
	key = cv2.waitKey(1) & 0xFF
	if key  == ord('q'):
		break
client_socket.close()
	

```

