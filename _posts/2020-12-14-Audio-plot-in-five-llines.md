---
layout: post
title: How to get audio frames from the microphone device
categories: [GUI tutorial series]
mathjax: true
featured-img: audio_in_five_lines
summary: How to get audio data from the microphone 
---
Hi friends! Install pyshine version 0.0.6 in Windows OS as:

```
pip3 install pyshine==0.0.6
```

### audio.py

```python

import pyshine as ps
audio,context = ps.audioCapture(mode='send')
ps.showPlot(context,name='pyshine.com')
while True:
	frame = audio.get()

```
save audio.py file and run it:

```
python3 audio.py
```
