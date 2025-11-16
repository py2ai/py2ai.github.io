---
description: Learn how to create a terminal-based Matrix rain animation in Python. Step-by-step beginner-friendly guide to build the iconic green digital rain effect.
featured-img: 20251109-matrixrain
keywords:
- Python
- Matrix
- terminal animation
- ANSI
- beginner
- tutorial
- digital rain
layout: post
mathjax: false
tags:
- python
- matrix
- animation
- terminal
- ANSI
- beginner
- tutorial
title: Matrix Rain Animation in Python – Complete Beginner's Guide
---

## Introduction

Have you ever wanted to recreate the iconic Matrix "digital rain" effect in your terminal using Python? In this tutorial, we'll build a fully working terminal-based Matrix rain animation. This guide is written for absolute beginners, so no prior experience with terminal animations is required.

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/DS2n4mHz6XA" 
    title="YouTube video player" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
  </iframe>
</div>

---

## Understanding the Requirements

Before we dive into coding, let's understand what we need:

- **Python 3 installed**
- A **terminal or command prompt** to run the script
- Familiarity with basic Python concepts (variables, loops, functions)

We'll create an animation that:

- Streams random characters in vertical columns
- Changes the intensity of green color to create the falling effect
- Runs continuously until you stop it with a keyboard shortcut

---

## Setting Up Your Environment

1. Ensure Python 3 is installed on your system by running:

```bash
python3 --version
```

2. Open your terminal or command prompt.
3. Create a new Python file called `matrix_rain.py`.

---

## Breaking Down the Code

### Importing Modules

```python
from random import randint, choice
import shutil
import time
```

- `randint` and `choice` from `random` allow us to generate random numbers and pick random characters.
- `shutil` helps get the terminal size.
- `time` allows us to control the speed of animation.

### Terminal Size

```python
columns, rows = shutil.get_terminal_size()
```

This gets the width (`columns`) and height (`rows`) of your terminal so our animation fills it perfectly.

### Characters and Streams

```python
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
streams = []
for _ in range(columns):
    streams.append({
        "pos": randint(0, rows-1),
        "length": randint(5, rows//2),
        "tail": []
    })
```

- `chars` are the characters displayed in the rain.
- Each column (`stream`) has:
  - `pos`: the current head position
  - `length`: the length of the falling tail
  - `tail`: a list of characters currently visible in the column

### Lime Green Color Function

```python
def lime_color(intensity):
    green = int(255 * intensity)
    return f"\033[38;2;0;{green};0m"
```

- This function generates a string that applies an RGB color to terminal text using ANSI escape codes.
- `intensity` is a value between 0.2 and 1 that determines how bright the green color is.
- `green = int(255 * intensity)` calculates the green channel value. 255 is the maximum intensity.

### ANSI Escape Codes Explained

- **ANSI** stands for **American National Standards Institute**. It defines standards for terminal control sequences that allow you to format text, control cursor position, and manipulate colors in a terminal.
- The escape code `\033` (or `\x1b`) is the standard prefix for ANSI sequences, telling the terminal to interpret the following characters as a command rather than text.
- Example: `\033[38;2;0;255;0m`:
  - `38;2;R;G;B` sets the foreground color in true RGB mode.
  - `0;255;0` is green at full intensity.
  - `\033[0m` resets formatting to default.

**History & Importance:**

- ANSI escape codes originated in the 1970s to provide consistent terminal behavior across different computer systems.
- They remain crucial today because they allow developers to create colorful, dynamic, and interactive terminal applications without relying on GUI frameworks.
- For future development, mastering ANSI codes is essential for CLI tools, terminal games, and custom dashboards.

### The Main Animation Loop

```python
print("\033[?25l", end="")  # Hide cursor
try:
    while True:
        screen_buffer = [[" " for _ in range(columns)] for _ in range(rows)]
        for col, stream in enumerate(streams):
            head = stream["pos"]
            length = stream["length"]
            char = choice(chars)
            stream["tail"].insert(0, char)
            if len(stream["tail"]) > length:
                stream["tail"].pop()
            for i, c in enumerate(stream["tail"]):
                y = (head - i) % rows
                intensity = max(0.2, 1 - i / length)
                screen_buffer[y][col] = f"{lime_color(intensity)}{c}\033[0m"
            stream["pos"] = (head + 1) % rows
        print("\033[H", end="")  # Move cursor to top-left
        for row in screen_buffer:
            print("".join(row))
        time.sleep(0.1)
```

- Hides the cursor and animates the Matrix rain using the concepts explained above.
- Uses ANSI sequences for color and cursor movement.

### Exiting the Animation

```python
except KeyboardInterrupt:
    print("\033[0m\033[?25h")  # reset color and show cursor
    print("\nMatrix rain stopped.")
```

- Restores default formatting and shows the cursor when you exit with Ctrl+C.

---

## Running the Script

1. Save the file as `matrix_rain.py`.
2. Open your terminal and navigate to the file location.
3. Run the script:

```bash
python3 matrix_rain.py
```

{% include codeHeader.html %}

```python
## Tutorial and Source code at www.pyshine.com
from random import randint, choice
import shutil, time
## Your Terminal window Size
columns, rows = shutil.get_terminal_size()
## Characters to display
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
## Initialize streams
streams = []
for _ in range(columns):
    streams.append({
        "pos": randint(0, rows-1), # head pos
        "length": randint(5, rows//2), # tail length
        "tail": [] # list of tail characters
    })
## Function to generate the lime Green with Intensity
def lime_color(intensity):
    green = int(255*intensity)
    return f"\033[38;2;0;{green};0m"
## Hide cursor for better effect
print("\033[?25l",end="")
try:
    while True:
        # Create empty screen buffer
        screen_buffer = [[" " for _ in range(columns)]
                         for _ in range(rows)]
        for col, stream in enumerate(streams):
            head = stream["pos"]
            length = stream["length"]
            # Add a new random character at the head
            char = choice(chars)
            stream["tail"].insert(0, char)
            # Keep tail at fixed length
            if len(stream["tail"])> length:
                stream["tail"].pop()
            # Draw tail characters in the screen buffer
            for i, c in enumerate(stream["tail"]):
                y = (head - i) % rows
                intensity = max(0.2, 1 -i /length)
                screen_buffer[y][col]=\
                f"{lime_color(intensity)}{c}\033[0m"
            # Move head down
            stream["pos"] = (head + 1) % rows
        # Move cursor to the top-left
        print("\033[H", end="")
        # Print all rows
        for row in screen_buffer:
            print("".join(row))
        # Small delay for smooth animation
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\033[0m\033[?25h") # Reset color show cursor
    print("\nMatrix rain stopped.")

```

4. Watch the Matrix rain animation in action.
5. Stop the animation with `Ctrl+C`.

---

## Conclusion

You now have a fully functional Matrix rain animation running in your terminal! This project introduces:

- Random number generation
- ANSI escape codes for color and cursor control
- Terminal manipulation with Python
- Animation using loops and buffers

Mastering ANSI codes opens the door to advanced terminal applications, interactive CLI tools, and colorful animations. Happy coding!

---

**Website:** https://www.pyshine.com
**Author:** PyShine