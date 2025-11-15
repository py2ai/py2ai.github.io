---
description: Step-by-step beginner-friendly guide to create a voice-enabled PyGame wall clock with tick sound, date display, TTS time announcement, and STT voice recognition using Vosk.
featured-img: 20251114-wallclock-greeting
keywords:
- Python
- PyGame
- clock
- wall-clock
- tick sound
- real-time
- TTS
- text-to-speech
- STT
- speech-to-text
- Vosk
- beginner
- tutorial
layout: post
mathjax: false
tags:
- python
- pygame
- clock
- sound
- tts
- stt
- text-to-speech
- speech-to-text
- vosk
- beginner
- tutorial
title: Voice-Enabled Wall Clock with Greetings
---
# Wall Clock with Voice Time and Greeting

This tutorial shows how to create a **Python-based wall clock** using `pygame`, with **text-to-speech** and **speech-to-text** using **pyttsx3** and **Vosk**. The app listens for the word "time" and responds with the current time and a greeting based on the current hour.

---

# Table of Contents

1. [Introduction](#introduction)
2. [Features Overview](#features-overview)
3. [Prerequisites](#prerequisites)
4. [Installing Dependencies](#installing-dependencies)
   - [Windows](#windows)
   - [macOS](#macos)
   - [Linux](#linux)
5. [Understanding Speech-to-Text (Vosk)](#understanding-speech-to-text-vosk)
   - [Why Speech-to-Text Is Important](#why-speech-to-text-is-important)
   - [How Vosk Works — The Theory (Simplified)](#how-vosk-works--the-theory-simplified)
   - [Vosk Model Types](#vosk-model-types)
   - [Where to get language models](#where-to-get-language-models)
   - [Supported languages](#supported-languages)
   - [Which model should beginners use?](#which-model-should-beginners-use)
6. [Understanding Text-to-Speech (pyttsx3)](#understanding-text-to-speech-pyttsx3)
   - [Changing voice](#changing-voice)
   - [Changing speaking speed](#changing-speaking-speed)
7. [Code Breakdown](#code-breakdown)
   - [Clock rendering](#clock-rendering)
   - [Tick sound generation](#tick-sound-generation)
   - [Typing animation](#typing-animation)
   - [Listen button behavior](#listen-button-behavior)
   - [STT callback logic](#stt-callback-logic)
8. [Running the App](#running-the-app)
9. [Troubleshooting](#troubleshooting)
10. [Full Source Code](#full-source-code)

---

## Introduction

This project builds a **beautiful wall clock GUI** using `pygame` — but with a twist:

It can **speak the time aloud**
…and it can **hear you ask for the time** using speech recognition.

When you say **“time”**, the app will detect your speech using **Vosk**, speak the current time using **pyttsx3**, and display a smooth **typing animation** at the bottom of the screen.

---

## Features Overview

### Analog Wall Clock

- Smooth second, minute, and hour hands
- Date and weekday display
- Optional dark theme compatible

### Built‑in Tick Sound

- Generated artificially using NumPy
- No external audio files required

### Voice Detection (STT)

- Uses Vosk offline speech recognition
- Works without internet
- Detects simple keywords (“time”)

### Text‑to‑Speech (TTS)

- Uses pyttsx3 (offline)
- Automatically speaks:
  *“Good afternoon. It's 03:25 PM now!”*

### Typing Animation

- Displays greeting and time
- Smooth blinking cursor
- Auto-clears after a few seconds

### Listen Button

- Toggles continuous microphone listening
- Runs recognition in background thread

---

## Prerequisites

You only need:

- Python 3.8+ (Better use Python 3.12)
- A microphone
- Basic terminal usage
- Ability to install packages

---

## Installing Dependencies

### Windows

```bash
python -m venv py312
py312\Scripts\activate

pip install pygame pyttsx3 sounddevice vosk numpy
```

Download a Vosk model:
https://alphacephei.com/vosk/models

Get the model for example [this ](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")

Extract and rename:

```
vosk-model-small-en-us-0.15
```

---

### macOS

```bash
python3 -m venv py312
source py312/bin/activate
brew install portaudio
pip install pygame pyttsx3 sounddevice vosk numpy
```

Download the same English model as above.

---

### Linux

```bash
python3 -m venv py312
source py312/bin/activate

pip install pygame pyttsx3 sounddevice vosk numpy
sudo apt update
sudo apt install -y libportaudio2 libportaudiocpp0 portaudio19-dev
```

---

## Understanding Speech‑to‑Text (Vosk)

Speech-to-Text (STT) is the process of converting spoken language into written text. Vosk is one of the most popular offline STT engines, known for being lightweight, accurate, and easy to use in Python projects.

Below is a detailed explanation suitable for tutorials, documentation, or learning purposes.

### Why Speech-to-Text Is Important

Speech-to-Text technology has become essential in modern software because:

#### Hands-Free Interaction

Users can control apps using voice, useful for clocks, assistants, and any hands-busy scenario (cooking, driving, etc.).

#### Accessibility

STT helps users with motor disabilities or those who cannot type easily.

#### Real-Time Automation

Voice commands can trigger events instantly — e.g.,
“start timer”, “stop music”, “what’s the time”.

#### Works Without a Screen

Useful for IoT devices, Raspberry Pi systems, or embedded gadgets.

#### Offline Security

Vosk works completely offline, so no voice data is sent to the cloud, enhancing privacy.

### How Vosk Works — The Theory (Simplified)

Although Vosk feels simple to use, under the hood it uses serious speech-processing theory. Here’s a digestible, beginner-friendly explanation:

1. Audio Capture

   * Your microphone records raw audio waves.
   * These waves are just numbers representing air pressure changes over time.
2. Feature Extraction (MFCC)

   * Raw audio is too detailed and noisy for machine learning models.
   * Vosk converts the raw audio into MFCC features (Mel-Frequency Cepstral Coefficients).

#### MFCCs represent:

- frequency distribution
- loudness
- tone
- patterns that humans perceive as speech

*Think of MFCCs as a fingerprint of sound that neural networks can understand.*

3. Acoustic Model (Neural Network)

   This model takes the MFCC features and predicts phonemes —
   the smallest units of sound like:

   `k    a    t    ( = "cat" )`

   The acoustic model is trained on thousands of hours of speech recordings.
4. Language Model

   Humans don’t speak in random phoneme sequences.
   So the language model helps predict what words make sense.

   For example:
   If the acoustic model detects something like:

   `d   t   a   m   p`

   The language model guides it to:

   `→ "time"`

   instead of gibberish.
5. Decoder

   The decoder combines:

   - predictions from the acoustic model
   - probabilities from the language model

   and chooses the most likely final text output.

   Result: clear, readable text.

### Why Developers Love Vosk

* 100% Offline
* No internet means:

✔ privacy
✔ reliability
✔ great for IoT or field environments

* Low CPU Usage

Runs on:

- Raspberry Pi
- Old laptops
- Mid-range PCs
- Small Models Available
- Some models are <50MB.
- Fast & Real-Time
- Even on modest hardware, it transcribes instantly.
- Multi-language Support

### Vosk Model Types

You can choose based on your device:

#### Small models

- <40MB
- Fastest
- Lower accuracy
- Ideal for Raspberry Pi or simple commands
- Perfect for this “voice clock project”

#### Medium models

- Balanced accuracy + speed
- Good for desktops or laptops

#### Large models

- Best accuracy
- Heavier CPU load
- Overkill for simple voice commands

### Where to get language models

All official models here:
https://alphacephei.com/vosk/models

### Supported languages

Vosk supports:

| Language | Model                           |
| -------- | ------------------------------- |
| English  | `vosk-model-small-en-us-0.15` |
| Japanese | `vosk-model-small-ja-0.22`    |
| Chinese  | `vosk-model-small-cn-0.22`    |
| Spanish  | `vosk-model-small-es-0.42`    |
| French   | `vosk-model-small-fr-0.22`    |
| Hindi    | `vosk-model-small-hi-0.22`    |

…and many more.

### Which model should beginners use?

Use a **small model**:

- Fast
- Low CPU usage
- Perfect for Raspberry Pi
- Accurate enough for single-word commands

Example small model names:

- `vosk-model-small-en-us-0.15`
- `vosk-model-small-es-0.42`
- `vosk-model-small-fr-0.22`

---

## Understanding Text‑to‑Speech (pyttsx3)

### Changing voice

In the code:

```python
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
```

### Changing speaking speed

```python
engine.setProperty('rate', 150)
```

Common values:

- 120 (slow)
- 150 (default)
- 180 (fast)

---

## Code Breakdown

### Clock Rendering

The clock is drawn manually:

- Outer circle
- Hour numbers
- Minute ticks
- Rotating hands based on time

### Tick Sound Generation

Instead of loading `.wav`, we generate audio:

- 1500 Hz click
- 50 ms duration
- Exponential fade

Thanks to NumPy, the clock always ticks without importing external files.

### Typing Animation

The greeting appears like real typing:

- Characters appear gradually
- Cursor blinks
- After 4 seconds, text clears automatically

### Listen Button Behavior

- Toggle on/off
- Blue → idle
- Green → listening
- Runs Vosk microphone stream in background

### STT Callback Logic

When Vosk decodes speech:

- Print detected text
- If it contains “time”, call `speak_time()`

---

## Running the App

Once everything is installed:

```bash
python main.py
```

Steps:

1. Clock appears
2. Click **LISTEN**
3. Speak:**“time”**
4. Clock will speak the current time
5. Text animation appears at bottom

---

## Troubleshooting

### ❗ No microphone detected

Try:

```bash
pip install sounddevice
```

Or select input device:

```python
sd.default.device = 1
```

---

### ❗ No speech detected

Use a **small** model; large ones need more CPU.
Speak clearly and wait 1–2 seconds after clicking “LISTEN”.

---

### ❗ TTS works only once

Ensure each TTS call creates a **new engine** (already done in provided code).

---

## Full Source Code

### 1. Windows DPI Awareness

```python
import ctypes
try:
    ctypes.windll.user32.SetProcessDPIAware()
except:
    pass
```

- Ensures the application displays correctly on **high-DPI screens** in Windows.
- Wrapped in a `try` block for compatibility with other OS.

---

### 2. Imports

```python
import pygame, math, datetime, sys, numpy as np, pyttsx3, threading, time
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json, os
```

- **pygame**: GUI and graphics.
- **math**: Trigonometry for clock hands.
- **datetime**: Current time for clock and greetings.
- **numpy**: Generating artificial tick sounds.
- **pyttsx3**: Text-to-speech engine.
- **threading**: Run TTS/STT in background.
- **sounddevice & vosk**: Speech-to-text recognition.
- **json & os**: Parse Vosk output and handle files.

---

### 3. Pygame Initialization

```python
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyShine Wall Clock")
```

- Initializes **Pygame** and **audio mixer** for sound playback.
- Sets **screen size** and window **title**.

---

### 4. Constants and Colors

```python
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (150, 150, 150)
DARK_GRAY = (50, 50, 50)
BUTTON_COLOR = (0, 128, 255)
BUTTON_HOVER = (0, 180, 255)
BUTTON_ACTIVE = (0, 200, 0)
LIME = (0, 255, 0)
```

- Defines **colors** used for clock face, hands, button, and text.

---

### 5. Clock Parameters & Fonts

```python
center_x, center_y = WIDTH // 2, HEIGHT // 2
clock_radius = 150

font = pygame.font.SysFont('Arial', 24, bold=True)
date_font = pygame.font.SysFont('Arial', 20)
button_font = pygame.font.SysFont('Arial', 20, bold=True)
time_str_font = pygame.font.SysFont('Arial', 28, bold=True)
```

- `center_x, center_y`: Center of clock.
- `clock_radius`: Size of clock face.
- Fonts for **numbers, date, button text, and TTS text display**.

---

### 6. Tick Sound

```python
def create_tick_sound():
    ...
    tick_sound = pygame.sndarray.make_sound(sound_array)
    tick_sound.set_volume(0.5)
    return tick_sound
```

- Generates a **short 1500Hz click** using NumPy.
- No external audio file needed.
- Used to play **tick every second**.

---

### 7. Listen Button

```python
button_rect = pygame.Rect(WIDTH // 2 - 80, 80, 160, 50)
listening_active = False
def draw_button(mouse_pos):
    ...
```

- Draws **button on screen**.
- Changes color when **hovered** or **active**.
- Controls **microphone listening state**.

---

### 8. Text Typing & TTS

```python
def speak_time():
    ...
    threading.Thread(target=tts_func, args=(spoken_time_str,), daemon=True).start()
```

- Determines **greeting** based on current hour.
- Formats **spoken text**: e.g., `"Good afternoon\nIt's 03:25 PM now!"`.
- Starts **text-to-speech in a background thread**.
- Updates **typing animation** variables.

---

### 9. Vosk Speech-to-Text Setup

```python
MODEL_PATH = "vosk-model-small-en-us-0.15"
vosk_model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)
```

- Loads **offline Vosk model**.
- Recognizer converts **audio bytes to text**.
- Ensures **offline speech recognition**.

---

#### STT Callbacks

```python
def stt_callback(indata, frames, time_data, status):
    ...
    if "time" in result_text.lower():
        speak_time()
```

- Processes audio from microphone.
- Converts it into text.
- Triggers `speak_time()` when **keyword “time”** is detected.

---

### 10. Clock Drawing Functions

#### Clock Face

```python
def draw_clock_face():
    ...
```

- Draws **outer circle, hour numbers, minute ticks**.
- Differentiates **hour ticks** (thicker) and **minute ticks** (thin).

#### Clock Hands

```python
def draw_clock_hands():
    ...
```

- Draws **hour, minute, second hands** based on current time.
- Plays **tick sound every second**.
- Draws **center pivot** circle.

#### Date Display

```python
def draw_date_display(now):
    ...
```

- Displays **current date** and **day of the week**.

#### Typing Animation

```python
def draw_spoken_time():
    ...
```

- Shows **greeting and time gradually** like typing.
- Cursor **blinks**.
- Auto clears after **4 seconds**.

---

### 11. Main Loop

```python
def main():
    ...
```

- Handles **events**:
  - Quit
  - ESC key
  - Mouse click on **listen button**
- Updates:
  - **Clock face**
  - **Hands**
  - **Date**
  - **Typed greeting**
  - **Listen button**
- Runs at **30 FPS**.
- Ensures **smooth animation and interaction**.

---

### 12. Entry Point

```python
if __name__ == "__main__":
    main()
```

- Starts the **main loop** when the script is executed directly.

---

### main.py

The full working source code here:

```python
# Tutorial and Source Code available: www.pyshine.com

import ctypes
try:
    ctypes.windll.user32.SetProcessDPIAware()
except:
    pass
import pygame
import math
import datetime
import sys
import numpy as np
import pyttsx3
import threading
import time

#  VOSK STT IMPORTS 
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json
import os

# Initialize Pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# Screen dimensions
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyShine Wall Clock")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (150, 150, 150)
DARK_GRAY = (50, 50, 50)
BUTTON_COLOR = (0, 128, 255)
BUTTON_HOVER = (0, 180, 255)
BUTTON_ACTIVE = (0, 200, 0)
LIME = (0, 255, 0)

# Clock parameters
center_x, center_y = WIDTH // 2, HEIGHT // 2
clock_radius = 150

# Fonts
font = pygame.font.SysFont('Arial', 24, bold=True)
date_font = pygame.font.SysFont('Arial', 20)
button_font = pygame.font.SysFont('Arial', 20, bold=True)
time_str_font = pygame.font.SysFont('Arial', 28, bold=True)

# Tick sound
def create_tick_sound():
    sample_rate = 44100
    duration = 0.05
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    envelope = np.exp(-50 * t)
    waveform = 0.5 * envelope * np.sign(np.sin(2 * np.pi * 1500 * t))
    waveform_int16 = np.int16(waveform * 3276)
    sound_array = np.column_stack([waveform_int16, waveform_int16])
    tick_sound = pygame.sndarray.make_sound(sound_array)
    tick_sound.set_volume(0.5)
    return tick_sound

tick = create_tick_sound()
last_second = -1

# Button
button_rect = pygame.Rect(WIDTH // 2 - 80,  80, 160, 50)
listening_active = False  # Button state
printed=False
def draw_button(mouse_pos):
    global printed
    if listening_active:
        color = BUTTON_ACTIVE
        text_str = "LISTENING..."
        if  printed==False:
            print('Start listening...')
            printed=True
    else:
        color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
        text_str = "LISTEN"
        printed=False
    pygame.draw.rect(screen, color, button_rect, border_radius=10)
    text = button_font.render(text_str, True, WHITE)
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)

# Shared variables
spoken_time_str = ""
typed_text = ""
typing_start_time = 0
typing_speed = 8
cursor_visible = True
last_cursor_toggle = 0
text_display_complete_time = None

# Speak time and trigger typing
def speak_time():
    global spoken_time_str, typed_text, typing_start_time, text_display_complete_time
    now = datetime.datetime.now()
    hour, minute = now.hour, now.minute

    # Determine AM/PM
    am_pm = "AM" if hour < 12 else "PM"
    hour_display = hour % 12
    hour_display = 12 if hour_display == 0 else hour_display

    # Determine greeting based on hour
    if 5 <= hour < 12:
        greeting = "Good morning"
    elif 12 <= hour < 17:
        greeting = "Good afternoon"
    elif 17 <= hour < 21:
        greeting = "Good evening"
    else:
        greeting = "Good night"

    # Combine greeting and time as two lines
    spoken_time_str = f"{greeting}\nIt's {hour_display:02d}:{minute:02d} {am_pm} now!"

    typed_text = ""  # Reset typing
    typing_start_time = time.time()
    text_display_complete_time = None

    # Speak TTS
    def tts_func(text):
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.say(text.replace("\n", ". "))  # Speak as single sentence
        tts_engine.runAndWait()

    threading.Thread(target=tts_func, args=(spoken_time_str,), daemon=True).start()

#  VOSK STT SETUP
MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    print(f"Missing model folder '{MODEL_PATH}'")
    sys.exit(1)

vosk_model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)
sd_stream = None  # Global reference to microphone stream

def audio_to_bytes(indata):
    try:
        return bytes(indata)
    except:
        return indata.tobytes()

def stt_listen_loop():
    global sd_stream
    try:
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype='int16',
            channels=1,
            callback=stt_callback
        ) as stream:
            sd_stream = stream
            while listening_active:
                time.sleep(0.1)
    except Exception as e:
        print("Microphone error:", e)

def stt_callback(indata, frames, time_data, status):
    if status:
        print("Audio status:", status)
    data = audio_to_bytes(indata)
    if recognizer.AcceptWaveform(data):
        result_text = json.loads(recognizer.Result()).get("text", "")
        if result_text.strip():  # Only print non-empty text
            print(f"Detected: {result_text}")
        if "time" in result_text.lower():
            speak_time()

# CLOCK DRAWING FUNCTIONS

def draw_clock_face():
    pygame.draw.circle(screen, WHITE, (center_x, center_y), clock_radius, 2)
    pygame.draw.circle(screen, DARK_GRAY, (center_x, center_y), clock_radius - 5, 2)
    for hour in range(1, 13):
        angle = math.radians(hour * 30 - 90)
        number_x = center_x + (clock_radius - 30) * math.cos(angle) - 10
        number_y = center_y + (clock_radius - 30) * math.sin(angle) - 10
        number_text = font.render(str(hour), True, WHITE)
        screen.blit(number_text, (number_x, number_y))
        tick_start_x = center_x + (clock_radius - 15) * math.cos(angle)
        tick_start_y = center_y + (clock_radius - 15) * math.sin(angle)
        tick_end_x = center_x + (clock_radius - 5) * math.cos(angle)
        tick_end_y = center_y + (clock_radius - 5) * math.sin(angle)
        pygame.draw.line(screen, WHITE, (tick_start_x, tick_start_y), (tick_end_x, tick_end_y), 3)
    for minute in range(60):
        if minute % 5 != 0:
            angle = math.radians(minute * 6 - 90)
            tick_start_x = center_x + (clock_radius - 10) * math.cos(angle)
            tick_start_y = center_y + (clock_radius - 10) * math.sin(angle)
            tick_end_x = center_x + (clock_radius - 5) * math.cos(angle)
            tick_end_y = center_y + (clock_radius - 5) * math.sin(angle)
            pygame.draw.line(screen, GRAY, (tick_start_x, tick_start_y), (tick_end_x, tick_end_y), 1)

def draw_clock_hands():
    global last_second
    now = datetime.datetime.now()
    hour, minute, second = now.hour % 12, now.minute, now.second
    if second != last_second:
        tick.play()
        last_second = second
    hour_angle = math.radians(hour * 30 + minute * 0.5 - 90)
    minute_angle = math.radians(minute * 6 + second * 0.1 - 90)
    second_angle = math.radians(second * 6 - 90)
    hour_x = center_x + clock_radius * 0.5 * math.cos(hour_angle)
    hour_y = center_y + clock_radius * 0.5 * math.sin(hour_angle)
    pygame.draw.line(screen, WHITE, (center_x, center_y), (hour_x, hour_y), 6)
    minute_x = center_x + clock_radius * 0.7 * math.cos(minute_angle)
    minute_y = center_y + clock_radius * 0.7 * math.sin(minute_angle)
    pygame.draw.line(screen, WHITE, (center_x, center_y), (minute_x, minute_y), 4)
    second_x = center_x + clock_radius * 0.8 * math.cos(second_angle)
    second_y = center_y + clock_radius * 0.8 * math.sin(second_angle)
    pygame.draw.line(screen, RED, (center_x, center_y), (second_x, second_y), 2)
    pygame.draw.circle(screen, RED, (center_x, center_y), 8)
    pygame.draw.circle(screen, WHITE, (center_x, center_y), 8, 2)
    return now

def draw_date_display(now):
    date_text = date_font.render(now.strftime("%Y-%m-%d"), True, WHITE)
    day_text = date_font.render(now.strftime("%A").upper(), True, WHITE)
    date_rect = date_text.get_rect(midtop=(center_x, center_y - clock_radius + 70))
    day_rect = day_text.get_rect(midtop=date_rect.midbottom)
    screen.blit(date_text, date_rect)
    screen.blit(day_text, day_rect)

def draw_spoken_time():
    global typed_text, last_cursor_toggle, cursor_visible, text_display_complete_time, spoken_time_str
    if spoken_time_str:
        elapsed = time.time() - typing_start_time
        # Split into lines
        lines = spoken_time_str.split("\n")
        chars_to_show = min(int(elapsed * typing_speed), sum(len(line) for line in lines))
    
        # Determine how many chars to show per line
        display_lines = []
        chars_remaining = chars_to_show
        for line in lines:
            if chars_remaining > len(line):
                display_lines.append(line)
                chars_remaining -= len(line)
            else:
                display_lines.append(line[:chars_remaining])
                break
    
        # Clear after 4 seconds of full display
        if chars_to_show == sum(len(line) for line in lines) and text_display_complete_time is None:
            text_display_complete_time = time.time()
        if text_display_complete_time and (time.time() - text_display_complete_time > 4):
            spoken_time_str = ""
            typed_text = ""
            return

        # Cursor blink timer
        if time.time() - last_cursor_toggle > 0.5:
            cursor_visible = not cursor_visible
            last_cursor_toggle = time.time()

        # Render each line
        y_offset = HEIGHT - 130
        for i, line in enumerate(display_lines):
            text_surface = time_str_font.render(line, True, LIME)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, y_offset + i*35))
            screen.blit(text_surface, text_rect)

        # Draw cursor at end of last line
        if cursor_visible and display_lines:
            last_line = display_lines[-1]
            text_surface = time_str_font.render(last_line, True, LIME)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, y_offset + (len(display_lines)-1)*35))
            cursor_x = text_rect.right + 2
            cursor_y = text_rect.top + 4
            cursor_height = text_rect.height - 2
            pygame.draw.rect(screen, LIME, (cursor_x, cursor_y-4, 3, cursor_height))


# MAIN LOOP
def main():
    global listening_active
    clock = pygame.time.Clock()
    running = True
    stt_thread = None

    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(event.pos):
                    listening_active = not listening_active
                    if listening_active:
                        # Start background STT listening
                        stt_thread = threading.Thread(target=stt_listen_loop, daemon=True)
                        stt_thread.start()
                    else:
                        # Stop listening
                        print("Stopping listening...")
                        sd_stream = None

        screen.fill(BLACK)
        draw_clock_face()
        now = draw_clock_hands()
        draw_date_display(now)
        draw_spoken_time()
        draw_button(mouse_pos)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

```

---

**Website:** https://www.pyshine.com
**Author:** PyShine
