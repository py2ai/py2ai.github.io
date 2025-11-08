---
categories:
- GUI tutorial series
description: 'Hi friends! Install pyshine version 0.0.6 in Windows OS as:'
featured-img: audio_in_five_lines
keywords:
- Audio Frames
- Microphone
- Pyshine
- Python
- Audio Capture
layout: post
mathjax: true
tags:
- Audio Frames
- Microphone
- Pyshine
- Python
- Audio Capture
title: How to get audio frames from the microphone device
---

Hi friends! Install pyshine version 0.0.6 in Windows OS as:

```
pip3 install pyshine==0.0.6
```

# audio.py

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
