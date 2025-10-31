---
layout: post
title: Mini Paint (Smooth Circular Brush) in Python
mathjax: false
featured-img: 26072022-python-logo
description: Build a simple interactive painting app with smooth circular brushes using Python and Pygame.
keywords: ["Python", "Pygame", "drawing", "brush", "interactive"]
tags: ["mini-paint", "pygame", "interactive-drawing", "python-tutorial"]
---

# 🖌️ Mini Paint Tutorial (Smooth Circular Brush)

### Educational Python Project – Create an Interactive Painting App

This tutorial walks through building a **Mini Paint application** using **Python and Pygame**. The app allows users to **draw with smooth circular brushes**, choose from multiple colors, and select different brush sizes.

---

## Features

- ✅ 7 colors
- ✅ 4 brush sizes
- ✅ Smooth, circle-based strokes
- ✅ Compact toolbar (portrait 400x900)

---

## Requirements

- Python 3.x
- Pygame library

Install Pygame if you don't have it:

```bash
pip install pygame
```

---

## Step 1: Setup Pygame Window

```python
import pygame, sys
import math

pygame.init()

# Screen setup
W, H = 400, 900
S = pygame.display.set_mode((W, H))
pygame.display.set_caption("Mini Paint - Smooth Circles")
```

---

## Step 2: Define Colors and Brush Sizes

```python
colors = [
    (255, 0, 0),     # Red
    (255, 165, 0),   # Orange
    (255, 255, 0),   # Yellow
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (75, 0, 130),    # Indigo
    (255, 255, 255)  # White (eraser)
]

brush_sizes = [3, 6, 12, 24]
current_color = colors[0]
current_size = brush_sizes[1]
```

---

## Step 3: Setup Toolbar

```python
toolbar_height = 120
color_buttons = []
size_buttons = []

# Color buttons
for i, c in enumerate(colors):
    rect = pygame.Rect(10 + i * 55, 10, 40, 40)
    color_buttons.append((rect, c))

# Size buttons
for i, size in enumerate(brush_sizes):
    x = 40 + i * 90
    y = 75
    size_buttons.append((x, y, size))
```

---

## Step 4: Create Smooth Brush Function

```python
def draw_smooth_circle_line(surface, color, start, end, radius):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    distance = max(1, int(math.hypot(dx, dy)))
    for i in range(distance):
        x = int(x1 + dx * i / distance)
        y = int(y1 + dy * i / distance)
        pygame.draw.circle(surface, color, (x, y), radius)
```

---

## Step 5: Main Loop

```python
drawing = False
last_pos = None
clock = pygame.time.Clock()

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if e.type == pygame.MOUSEBUTTONDOWN:
            mx, my = e.pos
            # Select color
            for rect, c in color_buttons:
                if rect.collidepoint(mx, my):
                    current_color = c
            # Select brush size
            for x, y, s in size_buttons:
                if (mx - x)**2 + (my - y)**2 <= (s + 4)**2:
                    current_size = s
            # Start drawing
            if my > toolbar_height:
                drawing = True
                last_pos = e.pos

        if e.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None

    # Drawing
    if drawing:
        mx, my = pygame.mouse.get_pos()
        if my > toolbar_height and last_pos:
            draw_smooth_circle_line(S, current_color, last_pos, (mx, my), current_size)
        last_pos = (mx, my)

    # Toolbar background
    pygame.draw.rect(S, (40, 40, 40), (0, 0, W, toolbar_height))

    # Draw color buttons
    for rect, c in color_buttons:
        pygame.draw.rect(S, c, rect)
        if c == current_color:
            pygame.draw.rect(S, (255, 255, 255), rect, 3)

    # Draw brush size selectors
    for x, y, s in size_buttons:
        pygame.draw.circle(S, (200, 200, 200), (x, y), s, 2)
        if s == current_size:
            pygame.draw.circle(S, (255, 255, 255), (x, y), s + 4, 2)

    pygame.display.flip()
    clock.tick(120)
```

---

## Complete Python Code

```python
import pygame, sys, math
pygame.init()

# Screen setup 
W, H = 400, 900
S = pygame.display.set_mode((W, H))
pygame.display.set_caption("Mini Paint - Smooth Circles")

# Colors
colors = [
    (255, 0, 0),     # Red
    (255, 165, 0),   # Orange
    (255, 255, 0),   # Yellow
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (75, 0, 130),    # Indigo
    (255, 255, 255)  # White (eraser)
]
brush_sizes = [3, 6, 12, 24]
current_color = colors[0]
current_size = brush_sizes[1]

# Toolbar setup 
toolbar_height = 120
color_buttons = []
size_buttons = []

# Color button positions
for i, c in enumerate(colors):
    rect = pygame.Rect(10 + i * 55, 10, 40, 40)
    color_buttons.append((rect, c))

# Size buttons (circles)
for i, size in enumerate(brush_sizes):
    x = 40 + i * 90
    y = 75
    size_buttons.append((x, y, size))

drawing = False
last_pos = None
clock = pygame.time.Clock()

def draw_smooth_circle_line(surface, color, start, end, radius):
    """Draw smooth brush stroke using overlapping circles"""
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    distance = max(1, int(math.hypot(dx, dy)))
    for i in range(distance):
        x = int(x1 + dx * i / distance)
        y = int(y1 + dy * i / distance)
        pygame.draw.circle(surface, color, (x, y), radius)

# Main Loop
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit(); sys.exit()

        # Mouse down
        if e.type == pygame.MOUSEBUTTONDOWN:
            mx, my = e.pos
            # Select color
            for rect, c in color_buttons:
                if rect.collidepoint(mx, my):
                    current_color = c
            # Select brush size
            for x, y, s in size_buttons:
                if (mx - x)**2 + (my - y)**2 <= (s + 4)**2:
                    current_size = s
            # Start drawing
            if my > toolbar_height:
                drawing = True
                last_pos = e.pos

        if e.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None

    # Drawing with smooth circular brush 
    if drawing:
        mx, my = pygame.mouse.get_pos()
        if my > toolbar_height and last_pos:
            draw_smooth_circle_line(S, current_color, last_pos, (mx, my), current_size)
        last_pos = (mx, my)

    # Toolbar background 
    pygame.draw.rect(S, (40, 40, 40), (0, 0, W, toolbar_height))

    # Draw color buttons
    for rect, c in color_buttons:
        pygame.draw.rect(S, c, rect)
        if c == current_color:
            pygame.draw.rect(S, (255, 255, 255), rect, 3)

    # Draw brush size selectors
    for x, y, s in size_buttons:
        pygame.draw.circle(S, (200, 200, 200), (x, y), s, 2)
        if s == current_size:
            pygame.draw.circle(S, (255, 255, 255), (x, y), s + 4, 2)

    pygame.display.flip()
    clock.tick(120)

```

## Conclusion

You now have a working Mini Paint application with smooth circular brushes, multiple colors, and variable brush sizes! This is a fun project to practice **interactive GUI programming in Python using Pygame**.

Enjoy painting!

