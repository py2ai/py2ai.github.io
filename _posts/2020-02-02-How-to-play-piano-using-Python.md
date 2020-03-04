---
layout: post
title: Playing piano with Python
author: Hussain A.
categories: [Python tutorial series]
mathjax: true
summary: A quick tutorial on multi threading in python to play twinkle twinkle little start poem
---




Hi there! 

Hello everyone 
Welcome to Pyshine and today we will play a poem
By using the wav files for each note of the Piano
I have a folder of wave files 
You may easily find the folder from the link below
https://freesound.org/people/TEDAgame/packs/25405/
I will also put them on www.pyshine.com

Ok so lets start the coding
First we require the list of notes so
```python

Twinkle_List = ['c4','c4','g4','g4','a4','a4','g4',\
				'f4','f4','e4','e4','d4','d4','c4',\
				'g5','g5','f4','f4','e4','e4','d4',\
				'g5','g5','f4','f4','e4','e4','d4',\
				'c4','c4','g4','g4','a4','a4','g4',\
				'f4','f4','e4','e4','d4','d4','c4',\
				]
```

This list has a total of 48 notes, and each line has 7 notes
We will play each note one by one and when we reach 
the end of a line , we simply give a pause of 1 second
Between each note we will give a 0.3 second pause


Lets import the required Libraries

```python
from threading import Thread
import pygame as pg 
import time 
```
First we initialize the mixer of pygame pg
```python
pg.mixer.init()
pg.init()
```
To be on safe side we will make the number of mixer 
channels to be equal to the number of elements in the 
Twinkle_List	
```python
pg.mixer.set_num_channels(len(Twinkle_List))
```
Now its time to make the function 
```python
def play_notes(notePath,duration):
	time.sleep(duration) # make a pause 
	pg.mixer.Sound(notePath).play()
	time.sleep(duration) # Let the sound play 
	print(notePath) # To see which note is now playing
```
Next is to make the path , all wav files are in Sounds folder
A total of 88 wav files
```python
path  = 'Sounds/'

cnt =1	# A counter to delay once a line is finished as there
# are 6 total lines
```
We now make a dictionary for each thread th
```python
th = {}
```
Lets iterate the Twinkle_List and launch each thread to
Play the note
```python
for t in Twinkle_List:
	th[t] = Thread(target = play_notes,args = (path+'{}.wav'.format(t),0.3))
	# These are arguments (path+'{}.wav'.format(t),0.3)
	# Lets start the thread
	th[t].start()
	th[t].join()
	if cnt%7==0:
		print("---Long Pause---")
		time.sleep(1) # Let the sound play for the last note of each line
		
	cnt+=1
```
Thats all lets play the code
This time we also visualize the waveform
That visualizer will be explained in next lab
Just have fun 
Thanks for watching please like and subscribe for the 
upcoming videos and tutorials
Have a nice and healthy day!



	



	
	







		
				
				
				
				
				
				
				
				
