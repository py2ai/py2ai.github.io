---
categories:
- GUI tutorial series
- Audio Processing
description: "Get audio frames from microphone in Python with just 5 lines of code using PyShine library. Simple audio capture tutorial for beginners with step-by-step examples."
featured-img: audio_in_five_lines
keywords:
- python audio capture
- microphone audio python
- pyaudio alternative
- python audio frames
- simple audio recording
- python microphone input
- audio capture tutorial
- pyshine audio
layout: post
mathjax: true
tags:
- Audio Frames
- Microphone
- Pyshine
- Python
- Audio Capture
- Audio Recording
title: Get Audio Frames from Microphone in Python - 5 Lines of Code
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
