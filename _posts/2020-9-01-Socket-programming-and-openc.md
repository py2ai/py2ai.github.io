---
layout: post
title: Socket programming to send and receive webcam video
categories: [tutorial series]
mathjax: true
featured-img: server
summary: This code will demonstrate the server client modules to transmit and receive video over wifi
---

[![Everything Is AWESOME](https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/posts/j.png?raw=true)](https://youtu.be/7-O7yeO3hNQ "Everything Is AWESOME")


Depending on the operating system, you can easily find the IP address of your machine as follows:

### MAC OS users

Go to the terminal window and run this command:

```
ipconfig getifaddr en0

```
That will show your LAN IP address. Note that en0 is commonly used for ethernet interface, and en1 is for the Airport interface. Make sure that your IP address is not starting from 127.x.x.x because that is your local host, and if you only want to check server client for the same pc then it is fine. Otherwise, consider using the command above and write the correct ip address for video transfer over different machines. 

Depending on different OS settings:

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

The require IP address will show against IPv4 Address




# server.py
```python
# Welcome to PyShine

# This code is for the server 
# Lets import the libraries
import socket, cv2, pickle,struct

# Socket Create
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_name  = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('HOST IP:',host_ip)
port = 9999
socket_address = (host_ip,port)

# Socket Bind
server_socket.bind(socket_address)

# Socket Listen
server_socket.listen(5)
print("LISTENING AT:",socket_address)

# Socket Accept
while True:
	client_socket,addr = server_socket.accept()
	print('GOT CONNECTION FROM:',addr)
	if client_socket:
		vid = cv2.VideoCapture(0)
		
		while(vid.isOpened()):
			img,frame = vid.read()
			a = pickle.dumps(frame)
			message = struct.pack("Q",len(a))+a
			client_socket.sendall(message)
			
			cv2.imshow('TRANSMITTING VIDEO',frame)
			key = cv2.waitKey(1) & 0xFF
			if key ==ord('q'):
				client_socket.close()
				
```
After running the server.py, copy paste the host_ip of that to client.py

# client.py

```python
# lets make the client code
import socket,cv2, pickle,struct

# create socket
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '192.168.1.20' # paste your server ip address here
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
	cv2.imshow("RECEIVING VIDEO",frame)
	key = cv2.waitKey(1) & 0xFF
	if key  == ord('q'):
		break
client_socket.close()
	
	
	
```
