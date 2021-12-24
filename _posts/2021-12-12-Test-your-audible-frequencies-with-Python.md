---
layout: post
title: Test your audible frequency range in Python
categories: [tutorial]
featured-img: 2021-12-24-20000Hz
mathjax: true
summary: A quick tutorial to generate audio tones of various frequencies and plot FFT
---

Hi friends, this tutorial is about generating audio tones of various frequencies and saving the FFT plots for each frequency. We will use `pygame` library to generate our
sound data and then we will use `wave` module to dump the data to .wav files. In PyGame, we have a very useful module named `pygame.sndarray` for accessing sound sample data functions 
to convert between numPy arrays and the sound objects. It will only be functional when pygame can use the external numpy package. If numpy can't be imported, `surfarray` becomes 
a MissingModule object. So numpy is very important in this case. Sound data is made of thousands of samples per second, and each sample is the amplitude of the wave 
at a particular moment in time. 

For example, in 22 kHz format, element number 5 of the array is the amplitude of the wave after 5/22000 seconds. The arrays are indexed by the X axis first, followed by the Y axis. 
Each sample is an 8-bit or 16-bit integer, depending on the data format. A stereo sound file has two values per sample, while a mono sound file only has one. We require `int16` array
of data which is a sine wave and will be generted using `numpy.array([4096 * numpy.sin(2.0 * numpy.pi * freq * x / sampleRate) for x in range(0, duration_sec*sampleRate)]).astype(numpy.int16)`.

We will initialize the pygame mixer with these parameters `pygame.mixer.init(sampling frequency, size, channels, buffer)` as `pygame.mixer.init(44100,-16,2,512)`. A list of
frequencies will be `[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]` in Hz.
We can plot the FFT of generated array using `fft` function from `from scipy.fftpack import fft`. By default the duration of each tone .wav file is set to 1 second as `duration_sec`.
The use of f-string will help us to dynamically generate name (string) as `plt.savefig(f'tone_{freq}_Hz.png')`. Once the png image is saved, we will proceed to use `pygame.sndarray.make_sound(arr2)` and get
the sound object and give a delay of 300ms (which will pause for a given number of milliseconds). The `pygame.time.delay(300)` function will use the processor (rather than sleeping) in order to make the delay
more accurate than `pygame.time.wait()`. The final portion of our code is to save the sound raw data to a .wav file for each frequency.
### main.py

```python
import numpy
import pygame
import wave, os
import matplotlib.pyplot as plt
from scipy.fftpack import fft
sampleRate = 44100
pygame.mixer.init(44100,-16,2,512)
duration_sec = 1
frequency_list=[x*100 for x in range(10)]+[x*1000 for x in range(1,21,1)]
# for details visis: www.pyshine.com
for freq in frequency_list:
	
    arr = numpy.array([4096 * numpy.sin(2.0 * numpy.pi * freq * x / sampleRate) for x in range(0, duration_sec*sampleRate)]).astype(numpy.int16)
    ys = arr
    T= ys.shape[0]
    ys = ys[0:T//1] 
    
    X = fft(ys,sampleRate)
    X_m = abs(X[0:sampleRate//2])
    plt.plot(X_m, 'o-', color = (0.25,1,0))
    x_text = numpy.argmax(X_m)
    y_text = X_m[x_text]
    plt.text(x_text, y_text-1e7, f"{x_text} Hz", size=10, rotation=0,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(0, 1, 1),
                   )
         )
    ax = plt.axes()
    ax.set_facecolor((0,0,0)) 
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    ax.set_title("Fast Fourier Transform plot")
    ax.yaxis.grid(True)
    plt.savefig(f'tone_{freq}_Hz.png')
    ax.clear()
    arr2 = numpy.c_[arr,arr]
    sound = pygame.sndarray.make_sound(arr2)
    sound.play(-1)
    pygame.time.delay(300)
    sound_raw = sound.get_raw()
    sfile = wave.open(f'pure_tone_{freq}_Hz.wav', 'w')
    print(f'Generating pure_tone_{freq}_Hz.wav')
    sfile.setframerate(sampleRate)
    sfile.setnchannels(2)
    sfile.setsampwidth(2)
    sfile.writeframesraw(sound_raw)
    sfile.close()
    sound.stop()
   
```

Usage:

```
python3 main.py
```
