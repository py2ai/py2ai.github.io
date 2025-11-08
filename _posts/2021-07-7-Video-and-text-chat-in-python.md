---
categories:
- tutorial
description: A quick tutorial to make server-client video and text chat
featured-img: 2021-12-24-chat-text
keywords:
- Python video chat
- Python text chat
- server-client chat
- UDP video transfer
- Python socket programming
- real-time chat tutorial
layout: post
mathjax: true
tags:
- Python
- video chat
- text chat
- UDP
- socket programming
- tutorial
title: Video and Text chat in Python
---



Hi friends, today's tutorial is rather more interesting than the previous ones. We will use a UDP socket connection for the real-time video transfer between the server and the client. So it will be a bidirectional video transfer, which means the server can watch the client's video and vice versa. 
At the same time, it will enable users at the server and client to chat via messages to each other. Two important things to consider:

First, to test the server.py and client.py on the same computer, you can use two USB cameras. One will have the id of 0 as in `vid = cv2.VideoCapture(0)` in the server.py. The other cam will have the id of 1 in the client.py, but you can change them according to available cams on your pc.

Second, to test both Python codes on separate computers, you can keep the id the same as 0 for both. 



# server.py
{% include codeHeader.html %}
```python
## This is server code to send video (over UDP) and message frames (over TCP)

import cv2, imutils, socket
import numpy as np
import time
import base64
import threading, wave, pyaudio,pickle,struct
import sys
import queue
import os
## For details visit pyshine.com
q = queue.Queue(maxsize=10)



BUFF_SIZE = 65536
server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()
host_ip = '10.211.55.27'#  socket.gethostbyname(host_name)
print(host_ip)
port = 9699
socket_address = (host_ip,port)
server_socket.bind(socket_address)
print('Listening at:',socket_address)

vid = cv2.VideoCapture(0)


def generate_video():
    
    WIDTH=400
    while(vid.isOpened()):
        try:
            _,frame = vid.read()
            frame = imutils.resize(frame,width=WIDTH)
            q.put(frame)
        except:
            os._exit(1)
        time.sleep(0.001)
    print('Player closed')
    BREAK=True
    vid.release()
	
def send_video():
    fps,st,frames_to_count,cnt = (0,0,20,0)
    cv2.namedWindow('SERVER TRANSMITTING VIDEO')        
    cv2.moveWindow('SERVER TRANSMITTING VIDEO', 400,30) 
    ## while True:
    msg,client_addr = server_socket.recvfrom(BUFF_SIZE)
    print('GOT connection from ',client_addr)
    WIDTH=400
    while(True):
        
        frame = q.get()
        encoded,buffer = cv2.imencode('.jpeg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
        message = base64.b64encode(buffer)
        server_socket.sendto(message,client_addr)
        frame = cv2.putText(frame,'FPS: '+str(round(fps,1)),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        
        if cnt == frames_to_count:
            try:
                fps = round(frames_to_count/(time.time()-st),1)
                st=time.time()
                cnt=0
            except:
                pass
        cnt+=1
    
        cv2.imshow('SERVER TRANSMITTING VIDEO', frame)
        key = cv2.waitKey(1) & 0xFF	
        if key == ord('q'):
            os._exit(1)
        time.sleep(0.01)   

def send_message():
    s = socket.socket()
    s.bind((host_ip, (port-1)))
    s.listen(5)
    client_socket,addr = s.accept()
    cnt=0
    while True:
        if client_socket:
            while True:
                print('SERVER TEXT ENTER BELOW:')
                data = input ()
                a = pickle.dumps(data)
                message = struct.pack("Q",len(a))+a
                client_socket.sendall(message)
           
                cnt+=1
                time.sleep(0.01)
                
        

def get_message():
    s = socket.socket()
    s.bind((host_ip, (port-2)))
    s.listen(5)
    client_socket,addr = s.accept()
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
            print('',end='\n')
            print('CLIENT TEXT RECEIVED:',frame,end='\n')
            print('SERVER TEXT ENTER BELOW:')
            time.sleep(0.001)

        except Exception as e:
            print('Dropped...')
            pass

    client_socket.close()
    print('Audio closed')
    
        


def get_video():

    cv2.namedWindow('SERVER RECEIVING VIDEO')        
    cv2.moveWindow('SERVER RECEIVING VIDEO', 400,360) 
    fps,st,frames_to_count,cnt = (0,0,20,0)
    BUFF_SIZE = 65536
    server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
    socket_address = (host_ip,port-3)
    server_socket.bind(socket_address)
    while True:
        
        packet,_ = server_socket.recvfrom(BUFF_SIZE)
        data = base64.b64decode(packet,' /')
        npdata = np.fromstring(data,dtype=np.uint8)

        frame = cv2.imdecode(npdata,1)
        frame = cv2.putText(frame,'FPS: '+str(fps),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.imshow("SERVER RECEIVING VIDEO",frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            client_socket.close()
            break
            

        if cnt == frames_to_count:
            try:
                fps = round(frames_to_count/(time.time()-st),1)
                st=time.time()
                cnt=0
            except:
                pass
        cnt+=1
        time.sleep(0.001)
            
    client_socket.close()
    cv2.destroyAllWindows() 


                
t1 = threading.Thread(target=send_message, args=())
t2 = threading.Thread(target=get_message, args=())
t3 = threading.Thread(target=generate_video, args=())
t4 = threading.Thread(target=send_video, args=())
t5 = threading.Thread(target=get_video, args=())
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()

```

### client.py
{% include codeHeader.html %}
```python
## Welcome to PyShine
## This is client code to receive video (over UDP) and message frames (over TCP)

import cv2, imutils, socket
import numpy as np
import time, os
import base64
import queue
import threading, wave, pyaudio,pickle,struct
## For details visit pyshine.com
BUFF_SIZE = 65536

BREAK = False
client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
host_name = socket.gethostname()
host_ip = '10.211.55.27'#  socket.gethostbyname(host_name)
print(host_ip)
port = 9699
message = b'Hello'

client_socket.sendto(message,(host_ip,port))


q = queue.Queue(maxsize=10)
vid = cv2.VideoCapture(1)


def generate_video():
    
    WIDTH=400
    while(vid.isOpened()):
        try:
            _,frame = vid.read()
            frame = imutils.resize(frame,width=WIDTH)
            q.put(frame)
            
        except:
            os._exit(1)
        time.sleep(0.001)
    print('Player closed')
    BREAK=True
    vid.release()

def get_message():
	
    ## TCP socket
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
            print('',end='\n')
            print('SERVER TEXT RECEIVED:',frame,end='\n')
            print('CLIENT TEXT ENTER BELOW:')
            time.sleep(0.01)
        except:
            
            break

    client_socket.close()
    print('closed')
    os._exit(1)


def send_message():

    ## create socket
    client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    socket_address = (host_ip,port-2)
    print('server listening at',socket_address)
    client_socket.connect(socket_address) 
    print("msg send CLIENT CONNECTED TO",socket_address)
    while True:
        if client_socket: 
            while (True):
                print('CLIENT TEXT ENTER BELOW:')
                data = input ()
                a = pickle.dumps(data)
                message = struct.pack("Q",len(a))+a
                client_socket.sendall(message)
                time.sleep(0.01)
                
                    
            

def get_video():

    cv2.namedWindow('CLIENT RECEIVING VIDEO')        
    cv2.moveWindow('CLIENT RECEIVING VIDEO', 10,360) 
    fps,st,frames_to_count,cnt = (0,0,20,0)
    message = b'Hello'
    client_socket.sendto(message,(host_ip,port))
    while True:
        packet,_ = client_socket.recvfrom(BUFF_SIZE)
        data = base64.b64decode(packet,' /')
        npdata = np.fromstring(data,dtype=np.uint8)

        frame = cv2.imdecode(npdata,1)
        frame = cv2.putText(frame,'FPS: '+str(fps),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.imshow("CLIENT RECEIVING VIDEO",frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            client_socket.close()
            os._exit(1)
            

        if cnt == frames_to_count:
            try:
                fps = round(frames_to_count/(time.time()-st),1)
                st=time.time()
                cnt=0
            except:
                pass
        cnt+=1
        time.sleep(0.001)
        
            
    client_socket.close()
    cv2.destroyAllWindows() 




def send_video():
    socket_address = (host_ip,port-3)
    print('server listening at',socket_address)

    fps,st,frames_to_count,cnt = (0,0,20,0)
    cv2.namedWindow('CLIENT TRANSMITTING VIDEO')        
    cv2.moveWindow('CLIENT TRANSMITTING VIDEO', 10,30) 
    while True:
        
        WIDTH=400
        while(True):
            frame = q.get()
            encoded,buffer = cv2.imencode('.jpeg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
            message = base64.b64encode(buffer)
            client_socket.sendto(message,socket_address)
            frame = cv2.putText(frame,'FPS: '+str(round(fps,1)),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            
            if cnt == frames_to_count:
                try:
                    fps = round(frames_to_count/(time.time()-st),1)
                    st=time.time()
                    cnt=0
                except:
                    pass
            cnt+=1
        
            cv2.imshow('CLIENT TRANSMITTING VIDEO', frame)
            key = cv2.waitKey(1) & 0xFF	
            if key == ord('q'):
                os._exit(1)
            time.sleep(0.001)
                

t1 = threading.Thread(target=get_message, args=())
t2= threading.Thread(target=send_message, args=())
t3 = threading.Thread(target=get_video, args=())
t4 = threading.Thread(target=send_video, args=())
t5 = threading.Thread(target=generate_video, args=())
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()

```
### Important:
Please note that you have to change the IP address only according to your computers. For example, in above codes we have `host_ip = '10.211.55.27'`, but in your case 
it might be different let say `host_ip = '192.168.1.10'` or else. But the question is how to find your computer's IP. The answer is to follow this tutorial :https://pyshine.com/Socket-programming-and-openc/

### Usage:
```python3 server.py```
```python3 client.py```

Below are outputs observed on our side, let's see how it runs on your side :)

### Output on server.py side

```
PS C:\PyShine> python .\server.py
10.211.55.27
Listening at: ('10.211.55.27', 9699)
GOT connection from  ('10.211.55.27', 64774)
SERVER TEXT ENTER BELOW:
Hello client I am server!
SERVER TEXT ENTER BELOW:

CLIENT TEXT RECEIVED: Hi server, nice to meet you!
SERVER TEXT ENTER BELOW:

CLIENT TEXT RECEIVED: Can you receive my video?
SERVER TEXT ENTER BELOW:
Yes
SERVER TEXT ENTER BELOW:
See you next time
SERVER TEXT ENTER BELOW:

CLIENT TEXT RECEIVED: see you
SERVER TEXT ENTER BELOW:

CLIENT TEXT RECEIVED: and have a nice day
SERVER TEXT ENTER BELOW:
you too
SERVER TEXT ENTER BELOW:
Bye Bye !
SERVER TEXT ENTER BELOW:

```

### Output on client.py side

```
PS C:\PyShine> python client.py
10.211.55.27
server listening at ('10.211.55.27', 9698)
CLIENT CONNECTED TO ('10.211.55.27', 9698)
server listening at ('10.211.55.27', 9697)
msg send CLIENT CONNECTED TO ('10.211.55.27', 9697)
CLIENT TEXT ENTER BELOW:
server listening at ('10.211.55.27', 9696)

SERVER TEXT RECEIVED: Hello client I am server!
CLIENT TEXT ENTER BELOW:
Hi server, nice to meet you!
CLIENT TEXT ENTER BELOW:
Can you receive my video?
CLIENT TEXT ENTER BELOW:

SERVER TEXT RECEIVED: Yes
CLIENT TEXT ENTER BELOW:

SERVER TEXT RECEIVED: See you next time
CLIENT TEXT ENTER BELOW:
see you
CLIENT TEXT ENTER BELOW:
and have a nice day
CLIENT TEXT ENTER BELOW:

SERVER TEXT RECEIVED: you too
CLIENT TEXT ENTER BELOW:

SERVER TEXT RECEIVED: Bye Bye !
CLIENT TEXT ENTER BELOW:

```
