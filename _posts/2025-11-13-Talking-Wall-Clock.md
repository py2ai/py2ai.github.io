---
description: Step-by-step beginner-friendly guide to create a live-updating PyGame wall clock with tick sound, date display, and TTS time announcement feature.
featured-img: 20251113-talking-wallclock
keywords:
- Python
- PyGame
- clock
- wall-clock
- tick sound
- real-time
- TTS
- text-to-speech
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
- text-to-speech
- beginner
- tutorial
title: Talking Clock Tutorial with Text-to-Speech
---
# PyShine Wall Clock Tutorial

This updated tutorial covers creating a wall clock using Python and Pygame with added functionality: the clock can now announce the current time via TTS (Text-to-Speech) and display it as a typing animation.

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/KH8lk9Fv3g4" 
    title="YouTube video player" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
  </iframe>
</div>

---

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Code Explanation](#code-explanation)
   - [Initializing Pygame and TTS](#initializing-pygame-and-tts)
   - [Screen, Colors, and Fonts](#screen-colors-and-fonts)
   - [Tick Sound](#tick-sound)
   - [Button Setup](#button-setup)
   - [Clock Face](#clock-face)
   - [Clock Hands](#clock-hands)
   - [Date Display](#date-display)
   - [TTS Spoken Time Display](#tts-spoken-time-display)
4. [Main Loop](#main-loop)
5. [Complete Code](#complete-code)
6. [Conclusion](#conclusion)

---

## Introduction

Have you ever wondered how digital clocks in apps or smart devices keep time, produce sounds, and even talk to you without missing a beat? In this tutorial, you'll not only build a fully functional wall clock in Python but also understand **how graphics, sound, and text-to-speech can work together in real-time**. By the end, you'll have a **hands-on, interactive clock** that looks professional, speaks the time, and animates text dynamically.

In this tutorial, you will learn how to:

- Draw a clock face with hour and minute marks
- Display hour, minute, and second hands in real-time
- Play a tick sound every second like a real analog clock
- Show the current date and day dynamically
- Implement a clickable button that announces the time using TTS
- Display the spoken time as a typing animation with a blinking cursor

This tutorial is beginner-friendly and demonstrates the **practical use of Pygame for graphics**, **NumPy for sound generation**, and **pyttsx3 for offline TTS**, giving you a solid foundation for **interactive multimedia applications** in Python. It also introduces threading and timing concepts, showing how to **keep your program responsive while handling multiple tasks simultaneously**.

---

## Setup

Install the required libraries:

```bash
pip install pygame numpy pyttsx3
```

---

## Code Explanation

### Initializing Pygame and TTS

```python
import pygame
import math
import datetime
import sys
import numpy as np
import pyttsx3
import threading
import time

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)
```

- Pygame is initialized for graphics and audio.
- The mixer is set to standard CD-quality audio.
- `pyttsx3` allows offline text-to-speech.
- `threading` ensures TTS runs without blocking the main loop.

### Screen, Colors, and Fonts

Screen dimensions, colors, and fonts are set up for the clock and text displays.

```python
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyShine Wall Clock")

BLACK, WHITE, RED, GRAY, DARK_GRAY = (0,0,0), (255,255,255), (255,0,0), (150,150,150), (50,50,50)
BUTTON_COLOR, BUTTON_HOVER, LIME = (0,128,255), (0,180,255), (0,255,0)

font = pygame.font.SysFont('Arial', 24, bold=True)
date_font = pygame.font.SysFont('Arial', 20)
button_font = pygame.font.SysFont('Arial', 20, bold=True)
time_str_font = pygame.font.SysFont('Arial', 28, bold=True)
```

- Fonts are used for hour numbers, date, button, and TTS text.

### Tick Sound

Tick sound generation is the same as before, using a short decaying waveform:

```python
def create_tick_sound():
    sample_rate = 44100
    duration = 0.05
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    envelope = np.exp(-50 * t)
    waveform = 0.5 * envelope * np.sign(np.sin(2 * np.pi * 1500 * t))
    waveform_int16 = np.int16(waveform * 32767)
    sound_array = np.column_stack([waveform_int16, waveform_int16])
    tick_sound = pygame.sndarray.make_sound(sound_array)
    tick_sound.set_volume(0.5)
    return tick_sound

tick = create_tick_sound()
last_second = -1
```

- Plays every time the second changes.

### Button Setup

A clickable button triggers the TTS functionality:

```python
button_rect = pygame.Rect(WIDTH // 2 - 80,  80, 160, 50)

def draw_button(mouse_pos):
    color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, color, button_rect, border_radius=10)
    text = button_font.render("Say Time", True, WHITE)
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)
```

### Clock Face

Draw the circular clock, hour numbers, and minute marks.

```python
def draw_clock_face():
    pygame.draw.circle(screen, WHITE, (center_x, center_y), clock_radius, 2)
    pygame.draw.circle(screen, DARK_GRAY, (center_x, center_y), clock_radius - 5, 2)

    # Hour marks
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

    # Minute marks
    for minute in range(60):
        if minute % 5 != 0:
            angle = math.radians(minute * 6 - 90)
            tick_start_x = center_x + (clock_radius - 10) * math.cos(angle)
            tick_start_y = center_y + (clock_radius - 10) * math.sin(angle)
            tick_end_x = center_x + (clock_radius - 5) * math.cos(angle)
            tick_end_y = center_y + (clock_radius - 5) * math.sin(angle)
            pygame.draw.line(screen, GRAY, (tick_start_x, tick_start_y), (tick_end_x, tick_end_y), 1)
```

- `math.radians()` converts degrees to radians.
- `pygame.draw.circle` and `pygame.draw.line` are used to create the clock face.

### Clock Hands

Draw hour, minute, and second hands with tick sound on second change.

```python
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
```

- The angles are calculated using the current time.
- `pygame.draw.line` draws the hands, and circles indicate the pivot.

### Date Display

Displays current date and day below the clock.

```python
def draw_date_display(now, clock_center, clock_radius):
    date_text = date_font.render(now.strftime("%Y-%m-%d"), True, WHITE)
    day_text = date_font.render(now.strftime("%A"), True, WHITE)
    date_rect = date_text.get_rect(midtop=(clock_center[0], clock_center[1] - clock_radius + 70))
    day_rect = day_text.get_rect(midtop=date_rect.midbottom)
    screen.blit(date_text, date_rect)
    screen.blit(day_text, day_rect)
```

- `strftime` formats the date.
- `blit` draws the text on the screen.

---

### TTS Spoken Time Display

```python
def speak_time():
    global spoken_time_str, typed_text, typing_start_time, text_display_complete_time
    now = datetime.datetime.now()
    hour, minute = now.hour, now.minute
    am_pm = "AM" if hour < 12 else "PM"
    hour_display = hour % 12 or 12
    spoken_time_str = f"The time is {hour_display:02d}:{minute:02d} {am_pm}"
    typed_text = ""
    typing_start_time = time.time()
    text_display_complete_time = None

    def tts_func(text):
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.say(text)
        tts_engine.runAndWait()

    threading.Thread(target=tts_func, args=(spoken_time_str,), daemon=True).start()
```

- Uses threading to play TTS asynchronously.
- Also triggers typing animation.

The purpose of `speak_time()` is to:

1. **Get the current time** and convert it into a human-readable string.
2. **Prepare the string for typing animation** in the Pygame window.
3. **Speak the time out loud using TTS** without freezing the main loop.

#### Step 1: Access global variables

``global spoken_time_str, typed_text, typing_start_time, text_display_complete_time``

* These variables are defined outside the function, but we need to **modify them inside** `speak_time()`.
* `spoken_time_str`: The full text that will be typed and spoken.
* `typed_text`: The portion of text currently shown in the typing animation.
* `typing_start_time`: The time when typing animation begins.
* `text_display_complete_time`: Tracks when typing finished, so we can clear text after a delay.

#### Step 2: Get current time

```
now = datetime.datetime.now()
hour, minute = now.hour, now.minute
am_pm = "AM" if hour < 12 else "PM"
hour_display = hour % 12 or 12
```

* `datetime.datetime.now()` fetches  **current system time** .
* `hour` and `minute` are extracted.
* `AM/PM` is determined with a simple conditional.
* `hour % 12 or 12` converts 24-hour format to 12-hour format.

**Example:**

| 24h hour | `hour % 12` | Displayed hour |
| -------- | ------------- | -------------- |
| 0        | 0             | 12 AM          |
| 13       | 1             | 1 PM           |
| 12       | 0             | 12 PM          |

---

#### Step 3: Prepare the spoken string

```
spoken_time_str = f"The time is {hour_display:02d}:{minute:02d} {am_pm}"
typed_text = ""
typing_start_time = time.time()
text_display_complete_time = None

```

* `spoken_time_str` is the **full sentence** we want the program to speak and display.
* `{hour_display:02d}` ensures **2-digit formatting** (e.g., `03:05`).
* `typed_text` is reset so that typing animation starts fresh.
* `typing_start_time` stores the  **exact moment typing begins** .
* `text_display_complete_time` is cleared because we haven’t finished typing yet.

#### Step 4: Define the TTS function

```python
def tts_func(text):
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.say(text)
    tts_engine.runAndWait()
```

* `pyttsx3.init()` initializes a  **TTS engine** .
* `setProperty('rate', 150)` sets the  **speed of speech** .
* `say(text)` queues the text to speak.
* `runAndWait()` actually plays the speech.
* **Important:** If we call this directly in the main loop, it would **freeze the program** until speaking is finished.


#### Step 5: Run TTS in a separate thread

* `threading.Thread(...)` starts a  **new background thread** .
* `target=tts_func` tells the thread what function to run.
* `args=(spoken_time_str,)` passes the text to speak.
* `daemon=True` ensures the thread will automatically close when the main program exits.
* `start()` actually launches the thread.

**Why threading?**

* Without threading, the main Pygame loop would **pause** while TTS speaks, freezing the clock and animations.
* With threading, TTS runs  **concurrently** , allowing animations and user interactions to continue uninterrupted.

```python
def draw_spoken_time():
    global typed_text, last_cursor_toggle, cursor_visible, text_display_complete_time, spoken_time_str

    if spoken_time_str:
        elapsed = time.time() - typing_start_time
        chars_to_show = min(int(elapsed * typing_speed), len(spoken_time_str))
        typed_text = spoken_time_str[:chars_to_show]

        if chars_to_show == len(spoken_time_str) and text_display_complete_time is None:
            text_display_complete_time = time.time()

        if text_display_complete_time and (time.time() - text_display_complete_time > 4):
            spoken_time_str = ""
            typed_text = ""
            return

        if time.time() - last_cursor_toggle > 0.5:
            cursor_visible = not cursor_visible
            last_cursor_toggle = time.time()

        text_surface = time_str_font.render(typed_text, True, LIME)
        text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT - 130))
        screen.blit(text_surface, text_rect)

        if cursor_visible:
            cursor_x = text_rect.right + 2
            cursor_y = text_rect.top + 4
            cursor_height = text_rect.height - 2
            pygame.draw.rect(screen, LIME, (cursor_x, cursor_y-4, 3, cursor_height))
```

- Shows typed text with a blinking cursor.
- Automatically clears after 4 seconds.

### Main Loop

In short, handles events, updates graphics, and triggers TTS. The main loop is the heart of any real-time Pygame application. It continuously runs while the program is active, handling input, updating graphics, and controlling timing. Let’s break it down:

```
clock = pygame.time.Clock()
running = True
while running:
```

* `pygame.time.Clock()` creates a Clock object that helps control the frame rate (how fast the loop runs).
* `running = True` is a flag to keep the loop active. Setting it to False will exit the loop and quit the program.

#### Handling Mouse and Keyboard Input

```
mouse_pos = pygame.mouse.get_pos()
for event in pygame.event.get():
```

* `pygame.mouse.get_pos()` gets the current position of the mouse.
* `pygame.event.get()` retrieves all events (mouse clicks, key presses, quitting, etc.) that occurred since the last frame.

```
if event.type == pygame.QUIT:
    running = False
```

* Triggered when the user closes the window.
* Setting `running = False` ends the loop.

```
elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
    running = False
```

* Detects if the Escape key is pressed to quit the program safely.

```
elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
    if button_rect.collidepoint(event.pos):
        threading.Thread(target=speak_time, daemon=True).start()

```

* Detects left mouse button click `(event.button == 1)`
* button_rect.collidepoint(event.pos) checks if the click happened on the TTS button
* `threading.Thread(..., daemon=True).start()` runs the speak_time() function in a separate thread so the TTS plays without freezing the main loop.

#### Updating the Screen

```
screen.fill(BLACK)
draw_clock_face()
now = draw_clock_hands()
draw_date_display(now)
draw_spoken_time()
draw_button(mouse_pos)
```

* `screen.fill(BLACK)` clears the screen at the beginning of each frame.
* `draw_clock_face()` draws the circular clock, hour numbers, and tick marks.
* `draw_clock_hands()` calculates the positions of hour, minute, and second hands and draws them. It also plays the tick sound every second.
* `draw_date_display(now)` shows the current date and weekday.
* `draw_spoken_time()` updates the typing animation for TTS text.
* `draw_button(mouse_pos)` draws the interactive button and changes color on hover.

#### Refreshing the Display

`pygame.display.flip()`

* Updates the full display to show the latest drawn frame.
* This is essential because Pygame only updates visuals when told to do so, making the loop responsible for screen updates.

#### Controlling Frame Rate

`clock.tick(30)`

* Limits the loop to 30 frames per second (FPS).
* Prevents the loop from running too fast and consuming excessive CPU.
* Ensures consistent timing for animations, ticking, and typing speed.

#### Exiting the Program

```
pygame.quit()
sys.exit()
```

* `pygame.quit()` cleans up all Pygame resources before closing.
* `sys.exit() `safely exits the Python program.

```python

def main():
    clock = pygame.time.Clock()
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(event.pos):
                    threading.Thread(target=speak_time, daemon=True).start()

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
```

---

## Complete Code

{% include codeHeader.html %}

```python
# Tutorial and Source Code available: www.pyshine.com
# Fix Windows DPI scaling issue
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
    waveform_int16 = np.int16(waveform * 32767)
    sound_array = np.column_stack([waveform_int16, waveform_int16])
    tick_sound = pygame.sndarray.make_sound(sound_array)
    tick_sound.set_volume(0.5)
    return tick_sound

tick = create_tick_sound()
last_second = -1

# Button
button_rect = pygame.Rect(WIDTH // 2 - 80,  80, 160, 50)

def draw_button(mouse_pos):
    color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, color, button_rect, border_radius=10)
    text = button_font.render("Say Time", True, WHITE)
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)

# Shared variables
spoken_time_str = ""
typed_text = ""
typing_start_time = 0
typing_speed = 8  # ≈30 WPM
cursor_visible = True
last_cursor_toggle = 0
text_display_complete_time = None

# Speak time and trigger typing
def speak_time():
    global spoken_time_str, typed_text, typing_start_time, text_display_complete_time
    now = datetime.datetime.now()
    hour, minute = now.hour, now.minute
    am_pm = "AM" if hour < 12 else "PM"
    hour_display = hour % 12
    hour_display = 12 if hour_display == 0 else hour_display
    spoken_time_str = f"The time is {hour_display:02d}:{minute:02d} {am_pm}"
    typed_text = ""  # Reset typing
    typing_start_time = time.time()
    text_display_complete_time = None

    def tts_func(text):
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.say(text)
        tts_engine.runAndWait()

    threading.Thread(target=tts_func, args=(spoken_time_str,), daemon=True).start()

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

    # Tick sound
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
        chars_to_show = min(int(elapsed * typing_speed), len(spoken_time_str))
        typed_text = spoken_time_str[:chars_to_show]

        # Once typing complete, start hold timer
        if chars_to_show == len(spoken_time_str) and text_display_complete_time is None:
            text_display_complete_time = time.time()

        # Clear after 4 seconds of full display
        if text_display_complete_time and (time.time() - text_display_complete_time > 4):
            spoken_time_str = ""
            typed_text = ""
            return

        # Cursor blink timer
        if time.time() - last_cursor_toggle > 0.5:
            cursor_visible = not cursor_visible
            last_cursor_toggle = time.time()

        # Render text (without cursor)
        text_surface = time_str_font.render(typed_text, True, LIME)
        text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT - 130))
        screen.blit(text_surface, text_rect)

        # Draw cursor only if visible
        if cursor_visible:
            # Compute cursor position at end of text
            cursor_x = text_rect.right + 2
            cursor_y = text_rect.top + 4
            cursor_height = text_rect.height - 2
            pygame.draw.rect(screen, LIME, (cursor_x, cursor_y-4, 3, cursor_height))


def main():
    clock = pygame.time.Clock()
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(event.pos):
                    threading.Thread(target=speak_time, daemon=True).start()

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

## Conclusion

The updated PyShine Wall Clock now features:

- Tick sound every second
- Real-time hour, minute, and second hands
- Date and day display
- Clickable TTS button that announces current time
- Animated typing display for spoken time with blinking cursor

This project demonstrates advanced Pygame features and integrating multimedia elements like TTS for interactive applications.

---

**Website:** https://www.pyshine.com
**Author:** PyShine
