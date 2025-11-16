---
description: Step-by-step beginner-friendly tutorial on building an interactive music maker with Pygame, including save/load functionality.
featured-img: 26072022-python-logo
keywords:
- Pygame
- Python
- music maker
- synth piano
- save load
layout: post
mathjax: false
tags:
- pygame
- music
- interactive
- beginner
- python
title: Music Maker with Save/Load in Python
---
# Interactive Music Maker in Python with Pygame

## Beginner-Friendly Tutorial – Build a Synth Piano with Save/Load

This tutorial walks you through creating an **interactive music maker** using **Python and Pygame**. You'll learn how to build a **synthetic piano**, **draw a sequencer grid**, and **save/load your music patterns**. By the end, you'll have a working **step sequencer** for simple melodies.

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/FS1Rj8BtfWg" 
    title="YouTube video player" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
  </iframe>
</div>

---

## Overview

This project creates a **simple step sequencer** where each row represents a note (C, D, E, F, G, A) and each column is a step in the melody. You can toggle notes on/off, **play the melody**, and **save/load patterns** to/from files.

Key features:

- Real-time playback of a synthetic piano
- Clickable grid for sequencing notes
- Save and load your music patterns using JSON
- Beginner-friendly Pygame implementation

## Setting Up Pygame

Start by importing libraries and initializing Pygame.

```python
import pygame, sys, time, numpy as np, json, os

pygame.init()
pygame.mixer.init(frequency=44100)
```

### Window and Font

Set the window size, caption, and font.

```python
W, H = 480, 540
S = pygame.display.set_mode((W, H))
pygame.display.set_caption("Music Maker(Save/Load)")
font = pygame.font.SysFont(None, 26)
clock = pygame.time.Clock()
```

## Creating the Synthetic Piano

We'll synthesize simple piano notes using **numpy arrays** and exponential decay envelopes.

```python
fs, duration = 44100, 1

def make_note(freq):
    t = np.linspace(0, duration, int(fs*duration), False)
    env = np.exp(-4.5 * t)  # decay envelope
    wave = (0.6*np.sin(2*np.pi*freq*t) +
            0.3*np.sin(2*np.pi*freq*2*t) +
            0.1*np.sin(2*np.pi*freq*3*t)) * env
    audio = (wave*32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((audio,audio)))

notes = {'C':261.63,'D':293.66,'E':329.63,'F':349.23,'G':392.00,'A':440.00}
sounds = {n: make_note(f) for n,f in notes.items()}
```

This creates **6 piano notes** using a **harmonic sum of sine waves**.

## Sequencer Grid & Layout

Define a **default pattern** and layout for the grid.

```python
steps = 32

def default_pattern():
    pat = {n:[0]*steps for n in notes}
    melody = [
        ('C',0),('C',1),('G',4),('G',5),
        ('A',8),('A',9),('G',12),
        ('F',16),('F',17),('E',20),('E',21),
        ('D',24),('D',25),('C',28)
    ]
    for n,p in melody: pat[n][p]=1
    return pat

pattern = default_pattern()
names = list(pattern.keys())

## Layout
margin_left, margin_top = 80, 150
grid_width = W - margin_left - 40
spacing = 3
cell_w = (grid_width - spacing*(steps-1)) / steps
cell_h = 50
```

Each **cell in the grid** represents a step in a sequence.

## Save & Load Buttons

Create **buttons** at the top for saving and loading your patterns.

```python
btn_w, btn_h = 180, 50
save_btn = pygame.Rect(40, 60, btn_w, btn_h)
load_btn = pygame.Rect(W - btn_w - 40, 60, btn_w, btn_h)
```

We'll also add a **file picker overlay** for selecting files.

```python
def file_picker_overlay(for_save=False):
    files = [f for f in os.listdir('.') if f.endswith('.txt')]
    if for_save and "new_file.txt" not in files:
        files.append("new_file.txt")
    if not files: return None

    picker_open = True
    selected = None
    while picker_open:
        S.fill((40,40,40))
        title = "Select file to SAVE:" if for_save else "Select file to LOAD:"
        S.blit(font.render(title, True, (255,255,255)), (20,20))
        file_rects = []
        for i, fname in enumerate(files):
            rect = pygame.Rect(40, 60 + i*50, 400, 40)
            color = (180,100,50) if for_save else (100,180,100)
            pygame.draw.rect(S, color, rect, border_radius=5)
            S.blit(font.render(fname, True, (0,0,0)), (rect.x+10, rect.y+8))
            file_rects.append((fname, rect))
        pygame.display.flip()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                mx,my = e.pos
                for fname, rect in file_rects:
                    if rect.collidepoint(mx,my):
                        selected = fname
                        picker_open = False
    return selected
```

## Event Handling

Handle mouse clicks for **toggle notes** and **buttons**.

```python
for e in pygame.event.get():
    if e.type == pygame.QUIT:
        pygame.quit(); sys.exit()
    elif e.type == pygame.MOUSEBUTTONDOWN:
        mx,my = e.pos
        if save_btn.collidepoint(mx,my):
            target_file = file_picker_overlay(for_save=True)
            if target_file:
                with open(target_file,"w") as f: json.dump(pattern,f)
        elif load_btn.collidepoint(mx,my):
            chosen_file = file_picker_overlay(for_save=False)
            if chosen_file:
                with open(chosen_file,"r") as f:
                    data = f.read().strip()
                    pattern = json.loads(data) if data else {n:[0]*steps for n in names}
        else:
            ## Toggle grid cell
            for r,name in enumerate(names):
                y = margin_top + r*(cell_h+10)
                for c in range(steps):
                    x = margin_left + c*(cell_w+spacing)
                    if pygame.Rect(x,y,cell_w,cell_h).collidepoint(mx,my):
                        pattern[name][c] ^= 1
```

## Playback Logic

Define **bpm, step, and timing** for playback.

```python
step, bpm, playing = 0, 190, True
beat_time, last = 60/bpm, time.time()

if playing and time.time()-last >= beat_time:
    for name in names:
        if pattern[name][step]: sounds[name].play()
    step = (step+1)%steps
    last = time.time()
```

## Complete Code

{% include codeHeader.html %}

```python
import pygame, sys, time, numpy as np, json, os

pygame.init()
pygame.mixer.init(frequency=44100)

##  Settings 
W, H = 480, 540
S = pygame.display.set_mode((W, H))
pygame.display.set_caption(" Music Maker(Save/Load)")
font = pygame.font.SysFont(None, 26)
clock = pygame.time.Clock()

##  Synthetic piano 
fs, duration = 44100, 1
def make_note(freq):
    t = np.linspace(0, duration, int(fs*duration), False)
    env = np.exp(-4.5 * t)
    wave = (0.6*np.sin(2*np.pi*freq*t) +
            0.3*np.sin(2*np.pi*freq*2*t) +
            0.1*np.sin(2*np.pi*freq*3*t)) * env
    audio = (wave*32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((audio,audio)))

notes = {'C':261.63,'D':293.66,'E':329.63,'F':349.23,'G':392.00,'A':440.00}
sounds = {n: make_note(f) for n,f in notes.items()}

##  Default pattern (Twinkle Twinkle) 
steps = 32
def default_pattern():
    pat = {n:[0]*steps for n in notes}
    melody = [
        ('C',0),('C',1),('G',4),('G',5),
        ('A',8),('A',9),('G',12),
        ('F',16),('F',17),('E',20),('E',21),
        ('D',24),('D',25),('C',28)
    ]
    for n,p in melody: pat[n][p]=1
    return pat

pattern = default_pattern()
names = list(pattern.keys())

##  Layout 
margin_left, margin_top = 80, 150
grid_width = W - margin_left - 40
spacing = 3
cell_w = (grid_width - spacing*(steps-1)) / steps
cell_h = 50

##  Buttons at Top 
btn_w, btn_h = 180, 50
save_btn = pygame.Rect(40, 60, btn_w, btn_h)
load_btn = pygame.Rect(W - btn_w - 40, 60, btn_w, btn_h)

##  Playback 
step, bpm, playing = 0, 190, True
beat_time, last = 60/bpm, time.time()

##  Pygame-native file picker 
def file_picker_overlay(for_save=False):
    files = [f for f in os.listdir('.') if f.endswith('.txt')]
    if for_save and "new_file.txt" not in files:  # allow saving to new file
        files.append("new_file.txt")
    if not files: return None

    picker_open = True
    selected = None
    while picker_open:
        S.fill((40,40,40))
        title = "Select file to SAVE:" if for_save else "Select file to LOAD:"
        S.blit(font.render(title, True, (255,255,255)), (20,20))
        file_rects = []
        for i, fname in enumerate(files):
            rect = pygame.Rect(40, 60 + i*50, 400, 40)
            color = (180,100,50) if for_save else (100,180,100)
            pygame.draw.rect(S, color, rect, border_radius=5)
            S.blit(font.render(fname, True, (0,0,0)), (rect.x+10, rect.y+8))
            file_rects.append((fname, rect))
        pygame.display.flip()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                mx,my = e.pos
                for fname, rect in file_rects:
                    if rect.collidepoint(mx,my):
                        selected = fname
                        picker_open = False
    return selected

##  Main loop 
while True:
    S.fill((25,25,25))

    ## Draw buttons
    pygame.draw.rect(S, (70,130,180), save_btn, border_radius=10)
    pygame.draw.rect(S, (100,180,100), load_btn, border_radius=10)
    S.blit(font.render("SAVE", True, (255,255,255)), (save_btn.x+65, save_btn.y+15))
    S.blit(font.render("LOAD", True, (255,255,255)), (load_btn.x+65, load_btn.y+15))

    ## Draw grid
    for r,name in enumerate(names):
        y = margin_top + r*(cell_h+10)
        S.blit(font.render(name, True, (255,255,255)), (20, y+cell_h/3))
        for c in range(steps):
            x = margin_left + c*(cell_w+spacing)
            rect = pygame.Rect(x, y, cell_w, cell_h)
            color = (0,200,200) if pattern[name][c] else (60,60,60)
            if c == step: pygame.draw.rect(S, (0,255,0), rect, 3)
            pygame.draw.rect(S, color, rect, border_radius=3)

    pygame.display.flip()

    ##  Events 
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        elif e.type == pygame.MOUSEBUTTONDOWN:
            mx,my = e.pos
            if save_btn.collidepoint(mx,my):
                target_file = file_picker_overlay(for_save=True)
                if target_file:
                    try:
                        with open(target_file,"w") as f: json.dump(pattern,f)
                        print("Saved to:", target_file)
                    except Exception as ex:
                        print("Save error:", ex)
            elif load_btn.collidepoint(mx,my):
                chosen_file = file_picker_overlay(for_save=False)
                if chosen_file:
                    try:
                        with open(chosen_file,"r") as f:
                            data = f.read().strip()
                            if data:
                                pattern = json.loads(data)
                            else:
                                pattern = {n:[0]*steps for n in names}
                        print("Loaded:", chosen_file)
                    except Exception as ex:
                        print("Load error:", ex)
                        pattern = {n:[0]*steps for n in names}
            else:
                ## Toggle grid cell
                for r,name in enumerate(names):
                    y = margin_top + r*(cell_h+10)
                    for c in range(steps):
                        x = margin_left + c*(cell_w+spacing)
                        if pygame.Rect(x,y,cell_w,cell_h).collidepoint(mx,my):
                            pattern[name][c] ^= 1

    ##  Playback 
    if playing and time.time()-last >= beat_time:
        for name in names:
            if pattern[name][step]: sounds[name].play()
        step = (step+1)%steps
        last = time.time()

    clock.tick(60)

```

An example Twinkle Twinkle Little Star music text file is here. Copy the following and paste in a new
twinkle.txt file you can load it later

### twinkle.txt

```
{"C": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "D": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0], "E": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], "F": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], "G": [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], "A": [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
```

## How to Run

1. Install dependencies:

```bash
pip install pygame numpy
```

2. Save the `.py` file.
3. Run:

```bash
python music_maker.py
```

4. Click on the grid to toggle notes.
5. Use **SAVE** and **LOAD** to persist your patterns.

## Key Learnings

- Using **Pygame mixer** for custom audio playback
- Creating **synthetic piano notes** with Numpy
- Implementing a **step sequencer grid**
- Saving/loading JSON data interactively
- Handling Pygame **mouse events and UI**

## Further Ideas

- Add more notes (B, higher octaves)
- Change **instruments** with different waveforms
- Export pattern to **MIDI**
- Add **tempo control slider**
- Enhance UI with **colors and animations**

---

This beginner-friendly tutorial introduces **music programming in Python** and **interactive sequencer logic**. With practice, you can expand this into a **full-fledged music maker** with more features.

---

**Website:** https://www.pyshine.com
**Author:** PyShine