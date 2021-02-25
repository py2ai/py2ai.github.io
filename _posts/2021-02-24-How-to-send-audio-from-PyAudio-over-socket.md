---
layout: post
title: How to send audio data using socket programming in Python
categories: [Socket Programming Series]
mathjax: true
featured-img: matthiasgroeneveld
summary: This tutorial is about using PyAudio and TCP sockets for server-client transfer of audio stream
---

Hi! Let's say we have an audio file (.wav), and we want to send it to the client so that the client can listen
the stream as a playback in real-time. For this purpose we require PyAudio and socket programming. PyAudio enriches
Python bindings for PortAudio, the cross-platform audio I/O library. We will first make codes for the TCP and then go on with the UDP. But before that, please
install PyAudio. If you can't install it using pip installer, then please go this [link](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and download the ```.whl``` according to your Python version. For instance if you are
using Python 3.6 then you need to download this ```PyAudio‑0.2.11‑cp36‑cp36m‑win_amd64.whl ```. After that go to the location of this download
and open up the power shell or terminal and use the command below:

```
pip3.6 install PyAudio‑0.2.11‑cp36‑cp36m‑win_amd64.whl
```

The audio data normally consists of 2 channels, which means the data array will be of shape (CHUNK,2), where CHUNK is
the number of data samples representing the digital sound. We commonly put CHUNK as 1024. Below are the TCP based
server client codes to provide these data CHUNKS from a server to a client. For more details on socket programming
please visit our previous tutorials. 

Here is the server side code, we assume that you already have wave (.wav) audio file in the same directory as this 
server.py file. Please run the server.py at one computer and accordingly provide your host_ip to it. 

## TCP SOCKET VERSION 

### server.py

```python
# This is server code to send video and audio frames over TCP

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

Usage:

```
python server.py
```

On the same or second computer please run the code below as:

```
python client.py
```

### client.py

```python
# Welcome to PyShine
# This is client code to receive video and audio frames over TCP

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

## UDP SOCKET VERSION 

Alright, lets do the same things as above but this time using UDP socket. The process of reading audio data should be streamed smoothly, in case of UDP. The ```wf.readframes(CHUNK)``` returns at most CHUNK frames of audio, as a bytes object. These bytes are then sent to the client address using ```server_socket.sendto(data,client_addr) ```. If we don't put enough wait, the receiver will overload and the reliability that all samples are properly sent, will be highly compromised and could even result in no sound or exceptin raised. We need to put a wait using ```time.sleep()```, before sending the next CHUNK of audio data. Since we know that the sample rate of audio is 44100 samples per second, so it means that the time for one sample to send is ``` 1/sample_rate``` . In this context, all the samples in a CHUNK would require a time of ```CHUNK/sample_rate ```. Therefore, after sending a CHUNK we will put a ```time.sleep(CHUNK/sample_rate)```. Here is the server side code:

### server.py

```python

# This is server code to send video and audio frames over UDP

import socket
import threading, wave, pyaudio, time

host_name = socket.gethostname()
host_ip = '192.168.1.102'#  socket.gethostbyname(host_name)
print(host_ip)
port = 9633
# For details visit: www.pyshine.com

def audio_stream_UDP():

    BUFF_SIZE = 65536
    server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)

    server_socket.bind((host_ip, (port)))
    CHUNK = 10*1024
    wf = wave.open("temp.wav")
    p = pyaudio.PyAudio()
    print('server listening at',(host_ip, (port)),wf.getframerate())
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    input=True,
                    frames_per_buffer=CHUNK)

    data = None
    sample_rate = wf.getframerate()
    while True:
        msg,client_addr = server_socket.recvfrom(BUFF_SIZE)
        print('GOT connection from ',client_addr,msg)
        
        while True:
            data = wf.readframes(CHUNK)
            server_socket.sendto(data,client_addr)
            time.sleep(CHUNK/sample_rate)
            
           
                

t1 = threading.Thread(target=audio_stream_UDP, args=())
t1.start()

```
Since in UDP there is no handshake, so at the receiver side, we have to store each received datagram in a queue. For this purpose our queue will serve
as a buffer of some size (let's say 100). On a thread we will continue to receive the UDP datagram of audio data. On the other hand, in a while loop
we will playback the sound. A time.sleep(5), will provide 5 second delay at the receiver side to fill the buffer, you can change it according to your requirements.

### client.py

```python
# Welcome to PyShine
# This is client code to receive video and audio frames over UDP

import socket
import threading, wave, pyaudio, time, queue

host_name = socket.gethostname()
host_ip = '192.168.1.102'#  socket.gethostbyname(host_name)
print(host_ip)
port = 9633
# For details visit: www.pyshine.com
q = queue.Queue(maxsize=100)

def audio_stream_UDP():
	BUFF_SIZE = 65536
	client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
	client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
	p = pyaudio.PyAudio()
	CHUNK = 10*1024
	stream = p.open(format=p.get_format_from_width(2),
					channels=2,
					rate=44100,
					output=True,
					frames_per_buffer=CHUNK)
					
	# create socket
	message = b'Hello'
	client_socket.sendto(message,(host_ip,port))
	socket_address = (host_ip,port)
	
	def getAudioData():
		while True:
			frame,_= client_socket.recvfrom(BUFF_SIZE)
			q.put(frame)
			print('Queue size...',q.qsize())
	t1 = threading.Thread(target=getAudioData, args=())
	t1.start()
	time.sleep(5)
	print('Now Playing...')
	while True:
		frame = q.get()
		stream.write(frame)

	client_socket.close()
	print('Audio closed')
	os._exit(1)



t1 = threading.Thread(target=audio_stream_UDP, args=())
t1.start()

```

Usage: on the server side run:
```
python server.py
```
On the client side please run:
```
python client.py
```
