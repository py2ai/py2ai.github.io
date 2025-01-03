---
layout: post
title: Transfer video over sockets from multiple clients
categories: [GUI tutorial series]
mathjax: true
featured-img: mclients
description: Socket programming with multiple clients and OpenCV in Python
---

<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/1skHb3IjOr4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>


Hello friends, today we will do socket programming for multiple clients and a single server. It's about creating multiple client sockets and transmitting their 
videos to a server in Python. The client.py utilizes OpenCv to access the video frames either from the live webcam or through the MP4 video. The server side code 
runs multi-threading to display video frame of each connected client. 

## Requirements

```
pip3 install opencv-contrib-python
pip3 install pyshine
pip3 install numpy
pip3 install imutils
```

The client side must know the IP of the server. Both the server and client should be connected to the same wifi router. Depending on the operating system, you can easily find the IP address of your machine as follows:

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
The required IP address will be for Wifi LAN (inet)

### Windows OS users

From the cmd window run this command:

```
ipconfig
```

The required IP address will be shown against IPv4 Address




### client.py
{% include codeHeader.html %}
```python

# Welcome to PyShine
# lets make the client code
# In this code client is sending video to server
import socket,cv2, pickle,struct
import pyshine as ps # pip install pyshine
import imutils # pip install imutils
camera = True
if camera == True:
	vid = cv2.VideoCapture(0)
else:
	vid = cv2.VideoCapture('videos/mario.mp4')
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '192.168.1.11' # Here according to your server ip write the address

port = 9999
client_socket.connect((host_ip,port))

if client_socket: 
	while (vid.isOpened()):
		try:
			img, frame = vid.read()
			frame = imutils.resize(frame,width=380)
			a = pickle.dumps(frame)
			message = struct.pack("Q",len(a))+a
			client_socket.sendall(message)
			cv2.imshow(f"TO: {host_ip}",frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				client_socket.close()
		except:
			print('VIDEO FINISHED!')
			break

```

And the server.py is available here

### server.py
{% include codeHeader.html %}
```python

# Welcome to PyShine
# In this video server is receiving video from clients.
# Lets import the libraries
import socket, cv2, pickle, struct
import imutils
import threading
import pyshine as ps # pip install pyshine
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

def show_client(addr,client_socket):
	try:
		print('CLIENT {} CONNECTED!'.format(addr))
		if client_socket: # if a client socket exists
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
				text  =  f"CLIENT: {addr}"
				frame =  ps.putBText(frame,text,10,10,vspace=10,hspace=1,font_scale=0.7, 						background_RGB=(255,0,0),text_RGB=(255,250,250))
				cv2.imshow(f"FROM {addr}",frame)
				key = cv2.waitKey(1) & 0xFF
				if key  == ord('q'):
					break
			client_socket.close()
	except Exception as e:
		print(f"CLINET {addr} DISCONNECTED")
		pass
		
while True:
	client_socket,addr = server_socket.accept()
	thread = threading.Thread(target=show_client, args=(addr,client_socket))
	thread.start()
	print("TOTAL CLIENTS ",threading.activeCount() - 1)
	
				

```




