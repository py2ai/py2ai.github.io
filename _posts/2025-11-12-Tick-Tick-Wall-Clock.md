---
description: Step-by-step beginner-friendly guide to create a live-updating PyGame wall clock with tick sound and date display.
featured-img: 20251112_wall_clock
keywords:
- Python
- PyGame
- clock
- wall-clock
- tick sound
- real-time
- beginner
- tutorial
layout: post
mathjax: false
tags:
- python
- pygame
- clock
- sound
- beginner
- tutorial
title: Wall Clock Tutorial with Tick Sound and Date Display
---
# PyShine Wall Clock Tutorial

This tutorial will guide you through creating a wall clock using Python and Pygame. It is designed for beginners and explains the code step by step. By the end, you will have a working digital clock with a tick sound and date display.

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/uOX9hG15t3g" 
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
   - [Initializing Pygame](#initializing-pygame)
   - [Screen and Colors](#screen-and-colors)
   - [Clock Face](#clock-face)
   - [Tick Sound](#tick-sound)
   - [Clock Hands](#clock-hands)
   - [Date Display](#date-display)
4. [Main Loop](#main-loop)
5. [Complete Code](#complete-code)
6. [Conclusion](#conclusion)

---

## Introduction

In this tutorial, you will learn how to:

- Draw a clock face with hour and minute marks
- Display hour, minute, and second hands
- Play a tick sound every second
- Show the current date and day
- Run a Pygame loop for a live clock

This project is beginner-friendly and helps understand Pygame basics, trigonometry, and working with real-time updates.

---

## Setup

Before running the code, ensure you have Python and Pygame installed. You can install Pygame using pip:

```bash
pip install pygame numpy
```

---

## Code Explanation

### Initializing Pygame

We first import required libraries and initialize Pygame.

```python
import pygame
import math
import datetime
import sys
import numpy as np

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)
```

- `pygame.init()` initializes all Pygame modules.
- `pygame.mixer.init()` sets up the sound system.

### Screen and Colors

We define screen dimensions, create a window, and set colors.

```python
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyShine Wall Clock")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (150, 150, 150)
DARK_GRAY = (50, 50, 50)
```

- Portrait mode is used: width 400px, height 600px.
- Colors are defined using RGB tuples.

### Clock Face

We draw the outer circle, hour numbers, and tick marks.

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

### Tick Sound

We create a short tick sound using NumPy.


**Theory Behind the Tick Sound:**

1. **Sample Rate:** `sample_rate = 44100`
   - Determines how many audio samples are played per second.
   - A standard CD-quality sample rate.

2. **Duration:** `duration = 0.05`
   - The tick lasts 50 milliseconds, making it a short click.

3. **Time Array:** `t = np.linspace(0, duration, n_samples, False)`
   - Creates evenly spaced points over the duration for waveform generation.

4. **Envelope:** `envelope = np.exp(-50 * t)`
   - Applies an exponential decay to the waveform.
   - This ensures the tick sound starts loud and fades quickly.

5. **Waveform Generation:** `waveform = 0.5 * envelope * np.sign(np.sin(2 * np.pi * 1500 * t))`
   - Uses a high frequency (1500 Hz) to create a sharp 'tick'.
   - `np.sign` converts the sine wave into a square-like waveform, giving a clicking effect.

6. **Integer Conversion:** `waveform_int16 = np.int16(waveform * 3267)`
   - Converts floating-point waveform to 16-bit integers for Pygame playback.

7. **Stereo Sound:** `sound_array = np.column_stack([waveform_int16, waveform_int16])`
   - Creates two channels (left and right) for stereo output.

8. **Make Sound Object:** `pygame.sndarray.make_sound(sound_array)`
   - Converts the NumPy array into a Pygame sound object.

9. **Volume:** `tick_sound.set_volume(0.5)`
   - Sets the playback volume to a moderate level.

By using this method, we can programmatically generate a simple but realistic tick sound without needing an external audio file. Each time the second changes, the tick plays, giving a real wall clock feel.

---

```python
def create_tick_sound():
    sample_rate = 44100
    duration = 0.05
    n_samples = int(sample_rate * duration)

    t = np.linspace(0, duration, n_samples, False)
    envelope = np.exp(-50 * t)
    waveform = 0.5 * envelope * np.sign(np.sin(2 * np.pi * 1500 * t))
    waveform_int16 = np.int16(waveform * 3267)
    sound_array = np.column_stack([waveform_int16, waveform_int16])

    tick_sound = pygame.sndarray.make_sound(sound_array)
    tick_sound.set_volume(0.5)
    return tick_sound

tick = create_tick_sound()
last_second = -1
```

- `np.linspace` creates a time array.
- `envelope` makes the sound decay quickly.
- `pygame.sndarray.make_sound` converts the array to a playable sound.

### Clock Hands

We calculate angles for hour, minute, and second hands and draw them.

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

We show the current date and weekday.

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

## Main Loop

The main loop handles events, updates the screen, and redraws the clock every frame.

```python
def main():
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        screen.fill(BLACK)
        draw_clock_face()
        now = draw_clock_hands()
        draw_date_display(now, (center_x, center_y), clock_radius)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
```

- `pygame.event.get()` captures quit or key press events.
- `screen.fill` clears the screen.
- `pygame.display.flip()` updates the display.
- `clock.tick(30)` limits to 30 frames per second.

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

# Initialize Pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# Screen dimensions for portrait mode
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyShine Wall Clock")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (150, 150, 150)
DARK_GRAY = (50, 50, 50)

# Clock parameters
center_x, center_y = WIDTH // 2, HEIGHT // 2
clock_radius = 150

# Font
font = pygame.font.SysFont('Arial', 24, bold=True)
date_font = pygame.font.SysFont('Arial', 20)

# Generate clock tick sound
def create_tick_sound():
    sample_rate = 44100
    duration = 0.05  # 50ms short tick
    n_samples = int(sample_rate * duration)
  
    # Quick decaying click
    t = np.linspace(0, duration, n_samples, False)
    envelope = np.exp(-50 * t)  # fast decay
    waveform = 0.5 * envelope * np.sign(np.sin(2 * np.pi * 1500 * t))  # high-pitched click
    waveform_int16 = np.int16(waveform * 3267)
    sound_array = np.column_stack([waveform_int16, waveform_int16])
    tick_sound = pygame.sndarray.make_sound(sound_array)
    tick_sound.set_volume(0.5)
    return tick_sound

tick = create_tick_sound()
last_second = -1  # Track last second to play tick

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
  
    # Play tick sound on every second change
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

def draw_date_display(now, clock_center, clock_radius):
    date_text = date_font.render(now.strftime("%Y-%m-%d"), True, WHITE)
    day_text = date_font.render(now.strftime("%A"), True, WHITE)
    date_rect = date_text.get_rect(midtop=(clock_center[0], clock_center[1] - clock_radius + 70))
    day_rect = day_text.get_rect(midtop=date_rect.midbottom)
    screen.blit(date_text, date_rect)
    screen.blit(day_text, day_rect)

def main():
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        screen.fill(BLACK)
        draw_clock_face()
        now = draw_clock_hands()
        draw_date_display(now, (center_x, center_y), clock_radius)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

```

---

## Conclusion

You have learned how to create a functional wall clock in Python using Pygame. You now understand:

- How to draw shapes and text
- How to calculate angles for clock hands
- How to play sounds
- How to display real-time updates

You can further enhance this project by:

- Adding an alarm feature
- Customizing the clock design
- Adding themes or color changes

This tutorial gives a solid foundation for beginner-friendly Pygame projects.

---

**Website:** https://www.pyshine.com
**Author:** PyShine
