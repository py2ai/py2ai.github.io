---
layout: post
title: How to send audio and video using socket programming in Python
categories: [GUI tutorial series]
mathjax: true
featured-img: matthiasgroeneveld
summary: This tutorial is about using OpenCV, UDP and TCP sockets for server-client transfer of audio-video streams
---

Hi! Let's say we have an audio file (.wav), and we want to send it to the client so that the client can listen
the stream as a playback in real-time. For this purpose we require PyAudio and socket programming. PyAudio enriches
Python bindings for PortAudio, the cross-platform audio I/O library. With PyAudio, we can easily use Python to 
play and record audio on a variety of platforms, such as GNU/Linux, Microsoft Windows, and Apple Mac OS X / macOS.
The audio data normally consists of 2 channels, which means the data array will be of shape (CHUNK,2), where CHUNK is
the number of data samples representing the digital sound. We commonly put CHUNK as 1024. Below are the TCP based
server client codes to provide these data CHUNKS from a server to a client. For more details on socket programming
please visit our previous tutorials. 

Here is the server side code, we assume that you already have wave (.wav) audio file in the same directory as this 
server.py file. Please run the server.py at one computer and accordingly provide your host_ip to it. 
```
python server.py
```

### server.py

```python
# This is server code to send video and audio frames over UDP/TCP

import socket
import threading, wave, pyaudio,pickle,struct

host_name = socket.gethostname()
host_ip = '192.168.1.102'#  socket.gethostbyname(host_name)
print(host_ip)
port = 9611

def audio_stream():
    server_socket = socket.socket()
    server_socket.bind((host_ip, (port-1)))

    server_socket.listen(5)
    CHUNK = 1024
    wf = wave.open("temp.wav", 'rb')
    
    p = pyaudio.PyAudio()
    print('server listening at',(host_ip, (port-1)))
   
    
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    input=True,
                    frames_per_buffer=CHUNK)

             

    client_socket,addr = server_socket.accept()
 
    data = None
    while True:
        if client_socket:
            while True:
              
                data = wf.readframes(CHUNK)
                a = pickle.dumps(data)
                message = struct.pack("Q",len(a))+a
                client_socket.sendall(message)
                
t1 = threading.Thread(target=audio_stream, args=())
t1.start()


```

On the same or second computer please run the code below as:
```
python client.py
```

### client.py

```python
# Welcome to PyShine
# This is client code to receive video and audio frames over UDP/TCP

import socket,os
import threading, wave, pyaudio, pickle,struct
host_name = socket.gethostname()
host_ip = '192.168.1.102'#  socket.gethostbyname(host_name)
print(host_ip)
port = 9611
def audio_stream():
	
	p = pyaudio.PyAudio()
	CHUNK = 1024
	stream = p.open(format=p.get_format_from_width(2),
					channels=2,
					rate=44100,
					output=True,
					frames_per_buffer=CHUNK)
					
	# create socket
	client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	socket_address = (host_ip,port-1)
	print('server listening at',socket_address)
	client_socket.connect(socket_address) 
	print("CLIENT CONNECTED TO",socket_address)
	data = b""
	payload_size = struct.calcsize("Q")
	while True:
		try:
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
			stream.write(frame)

		except:
			
			break

	client_socket.close()
	print('Audio closed')
	os._exit(1)
	
t1 = threading.Thread(target=audio_stream, args=())
t1.start()


```

If everything goes well you will listen the good quality sound at the client side.


