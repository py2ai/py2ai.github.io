---
layout: post
title: UDP Single server to multiple clients 
featured-img: udp_single_s_multiple_c
mathjax: true
toc: true
summary:  This tutorial is about streaming real-time video to multiple clients over UDP
---

Hi friends! Here is the UDP based single server and multiple clients. For more details visit UDP basic server client tutorial at pyshine.com

### server.py
{% include codeHeader.html %}
```python

# This is server code to send video frames over UDP
import cv2,  socket
import time
import base64, threading
import os
global RUNF, frame
RUNF = {}
frame = None
# For details visit pyshine.com

BUFF_SIZE = 65536
server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
server_socket.settimeout(0.2)
host_name = socket.gethostname()
host_ip = '192.168.10.113'
print('Server IP:',socket.gethostbyname(host_name))
print('Selected IP:',host_ip )
port = 9999
socket_address = (host_ip,port)
server_socket.bind(socket_address)
print('Listening at:',socket_address)

vid = cv2.VideoCapture(0) #  replace 'rocket.mp4' with 0 for videos
RUNF[str(socket_address)] = False

def video_stream_gen():
	global RUNF, frame  
	width, height = 400, 300
	dsize = (width, height)
	
	fps,st,frames_to_count,cnt = (0,0,20,0)
	while(vid.isOpened()):
		try:
			_,_frame = vid.read()
			_frame = cv2.resize(_frame, dsize)
			frame = cv2.putText(_frame,'FPS: '+str(fps),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
			cv2.imshow('TRANSMITTING VIDEO',frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				RUNF[str(socket_address)] = True
				break
			if cnt == frames_to_count:
				try:
					fps = round(frames_to_count/(time.time()-st))
					st=time.time()
					cnt=0
				except:
					pass
			cnt+=1
		except:
			os._exit(1)
	print('Stream closed')
	vid.release()
	
thread = threading.Thread(target=video_stream_gen, args=())
thread.start()


def serve_client(client_addr,client_msg):
	global RUNF,frame
	if client_msg:
		while True:
			encoded,buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
			message = base64.b64encode(buffer)
			server_socket.sendto(message,client_addr)
			if str(client_addr) in RUNF:
				if  RUNF[str(client_addr)] or RUNF[str(socket_address)]:
					break
	RUNF[str(client_addr)] = False
	
thread_run = {}
if __name__ == '__main__':
	
	while True:
		if RUNF[str(socket_address)]:
			break
		try:
			client_msg,addr = server_socket.recvfrom(BUFF_SIZE)
		except:
			client_msg = False
		if client_msg == b'bye':
			RUNF[str(addr)] = True
		if client_msg == b'Hello' :
			print('Got request from: ',addr,client_msg)
			thread = threading.Thread(target=serve_client, args=(addr,client_msg))
			thread.start()
		print("TOTAL CLIENTS ",threading.activeCount()-2, end='\r') # edited here because one thread is already started before		
		time.sleep(0.1)
	os._exit(1)
```

Following is the index.html containing javascript

### client.py
{% include codeHeader.html %}
```python

# This is client code to receive video frames over UDP
import cv2, socket
import numpy as np
import base64

BUFF_SIZE = 65536
client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()
host_ip = '192.168.10.113'
print('Server IP:',socket.gethostbyname(host_name))
print('Selected IP:',host_ip )
port = 9999
message = b'Hello'

client_socket.sendto(message,(host_ip,port))

while True:
    packet,_ = client_socket.recvfrom(BUFF_SIZE)
    data = base64.b64decode(packet,' /')
    npdata = np.fromstring(data,dtype=np.uint8)
    frame = cv2.imdecode(npdata,1)
    cv2.imshow("RECEIVING VIDEO",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        message = b'bye'
        client_socket.sendto(message,(host_ip,port))
        client_socket.close()
        break
```
First run the server.py in a terminal window according to your IP address and then client.py accordingly
