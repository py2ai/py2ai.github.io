---
layout: post
title: AC to DC conversion Simulation in Python 
mathjax: true
featured-img: 26072022-python-logo
description:  Full Bridge AC/DC four Diodes based Rectifier
keywords: ["simulaiton", "PyGame"]
tags: ["ad-to-dc", "full-wave-bridge", "ac dc rectifier simulation"]
---

# Full Wave Bridge Rectifier Simulation with Pygame

### Educational Python Project – Visualize AC to DC Conversion

This tutorial walks through building a **Full Wave Bridge Rectifier Simulation** using **Python and Pygame**. The simulation visually demonstrates how **AC (Alternating Current)** is converted into **DC (Direct Current)** using **four diodes** arranged in a bridge circuit.


## Table of Contents
## Overview {#overview}
## Circuit Concept {#circuit-concept}
## Pygame Setup {#pygame-setup}
## Drawing the Circuit {#drawing-the-circuit}
## Waveform Animation {#waveform-animation}
## Current Flow Visualization {#current-flow-visualization}
## Complete Python Code {#complete-python-code}
## How to Run {#how-to-run}
## Key Learnings {#key-learnings}
## Further Ideas {#further-ideas}



## Overview

This simulation provides an interactive, animated look at how a **bridge rectifier** converts alternating current into pulsating direct current. The display shows:

- **Top:** Input AC waveform
- **Center:** Bridge circuit with diode conduction paths
- **Bottom:** Output DC waveform

The current flow alternates between two paths during positive and negative half-cycles, visually explaining the rectification process.



## Circuit Concept

A **full wave bridge rectifier** uses **four diodes (D1–D4)** arranged in a diamond shape. During each half-cycle:

- **Positive Half-Cycle:**
  - Current flows through D1 → Load → D4.
  - D1 and D4 conduct.
- **Negative Half-Cycle:**
  - Current flows through D3 → Load → D2.
  - D2 and D3 conduct.

Thus, both halves of the AC waveform are converted into unidirectional current, producing a full-wave DC output.



## Pygame Setup

We start by setting up a **650×900 portrait window** to allow space for the waveforms and the circuit diagram.

```python
import pygame, sys, math
pygame.init()

W, H = 650, 900
S = pygame.display.set_mode((W, H))
pygame.display.set_caption("Full Wave Bridge Rectifier")
F = pygame.font.SysFont("Arial", 18, bold=True)
clk = pygame.time.Clock()
```

We define a dictionary of colors for background, grid, wires, diodes, and active paths:

```python
C = {
    "BG": (0, 0, 0),
    "GRID": (50, 50, 50),
    "WIRE": (100, 100, 100),
    "DIODE": (150, 150, 150),
    "IN": (0, 255, 255),
    "OUT": (255, 220, 0),
    "ANODE": (0, 255, 0),
    "CATHODE": (255, 0, 0),
    "POSITIVE_HALF": (50, 50, 255),
    "NEGATIVE_HALF": (255, 50, 50)
}
```



## Drawing the Circuit

We build a diamond-shaped layout for the diodes:

```
       (D1)
        / \
     D2   D3
        \ /
       (D4)
```

Each diode is drawn with proper anode/cathode markers using `pygame.draw.polygon()` and `pygame.draw.circle()` for clarity.

Active diodes (the ones conducting in the current half-cycle) are highlighted in bright colors.



## Waveform Animation

Two waveforms are drawn to represent **input AC** and **output DC** signals:

- The **input AC** is plotted as a sine wave (`math.sin()`), color-coded for positive (blue) and negative (red) halves.
- The **output DC** is plotted as the **absolute value** of the sine wave, representing rectified output.

```python
for x in range(280):
    y_val = amp * math.sin(freq*(x + p))
    y_pos = box_y + y_val
    color = C["POSITIVE_HALF"] if y_val >= 0 else C["NEGATIVE_HALF"]
    draw.circle(S, color, (cx - 140 + x, int(y_pos)), 1)
```



## Current Flow Visualization

To make the current path more interactive, small **moving dots** simulate the direction of electron flow through the active diodes and load resistor. The animation updates continuously using a changing `phase` variable.

Positive and negative half-cycles each show distinct paths:

- **Blue dots:** Positive half (D1–D4 conducting)
- **Red dots:** Negative half (D2–D3 conducting)

This helps students visually understand how current always flows in the same direction through the load.



## Complete Python Code

Below is the complete code that combines everything explained above.

```python
"""
Full Wave Bridge Rectifier Simulation - Pygame
----------------------------------------------
How AC is converted into DC using 
four diodes arranged in a bridge circuit.

Controls:
  [ESC] → Quit
"""

import pygame, sys, math
pygame.init()

# --- Setup - Portrait Mode ---
W, H = 650, 900  # Portrait orientation
S = pygame.display.set_mode((W, H))
pygame.display.set_caption("Full Wave Bridge Rectifier")
F = pygame.font.SysFont("Arial", 18, bold=True)
clk = pygame.time.Clock()

# --- Colors ---
C = {
    "BG": (0, 0, 0),
    "GRID": (50, 50, 50),
    "WIRE": (100, 100, 100),  # Dimmer for non-active wires
    "DIODE": (150, 150, 150), # Dimmer for non-active diodes
    "IN": (0, 255, 255),
    "OUT": (255, 220, 0),
    "LOAD": (255, 150, 150),
    "ANODE": (0, 255, 0),    # Green for anode
    "CATHODE": (255, 0, 0),  # Red for cathode
    "POSITIVE_HALF": (50, 50, 255),  # BLUE for positive half (swapped)
    "NEGATIVE_HALF": (255, 50, 50)   # RED for negative half (swapped)
}

# --- Wave and Animation ---
amp, freq, speed, phase = 50, 0.15, .04, 0
draw = pygame.draw
txt = lambda t, p, c: S.blit(F.render(t, 1, c),
             F.render(t, 1, c).get_rect(center=p))

def grid():
    for x in range(0, W, 50):
        draw.line(S, C["GRID"], (x, 0), (x, H))
    for y in range(0, H, 50):
        draw.line(S, C["GRID"], (0, y), (W, y))

def draw_diode_with_polarity(pos, angle=0, color=None, active=False, reverse=False):
    x, y = pos
    s = 20
    
    # Create a surface for the diode
    diode_surface = pygame.Surface((s*3, s*3), pygame.SRCALPHA)
    
    # Use provided color or default
    if active and color:
        diode_color = color
        wire_color = color
        # Make active diodes brighter
        anode_color = (100, 255, 100)  # Brighter green
        cathode_color = (255, 100, 100) # Brighter red
    else:
        diode_color = C["DIODE"]
        wire_color = C["WIRE"]
        anode_color = C["ANODE"]
        cathode_color = C["CATHODE"]
    
    # Draw diode symbol - reverse the direction if needed
    if reverse:
        # Reverse diode (cathode on left, anode on right)
        pygame.draw.polygon(diode_surface, diode_color, 
                           [(s*2.2, s*0.5), (s*2.2, s*2.5), (s*1.2, s*1.5)])
        pygame.draw.polygon(diode_surface, wire_color, 
                           [(s*2.2, s*0.5), (s*2.2, s*2.5), (s*1.2, s*1.5)], 2)
        pygame.draw.line(diode_surface, wire_color, (s*0.8, s*0.5), (s*0.8, s*2.5), 3)
        
        # Anode and cathode markers (swapped for reverse diode)
        pygame.draw.circle(diode_surface, cathode_color, (int(s*0.5), int(s*1.5)), 4)  # Cathode on left
        pygame.draw.circle(diode_surface, anode_color, (int(s*2.5), int(s*1.5)), 4)    # Anode on right
    else:
        # Normal diode (anode on left, cathode on right)
        pygame.draw.polygon(diode_surface, diode_color, 
                           [(s*0.8, s*0.5), (s*0.8, s*2.5), (s*1.8, s*1.5)])
        pygame.draw.polygon(diode_surface, wire_color, 
                           [(s*0.8, s*0.5), (s*0.8, s*2.5), (s*1.8, s*1.5)], 2)
        pygame.draw.line(diode_surface, wire_color, (s*2.2, s*0.5), (s*2.2, s*2.5), 3)
        
        # Anode and cathode markers
        pygame.draw.circle(diode_surface, anode_color, (int(s*0.5), int(s*1.5)), 4)
        pygame.draw.circle(diode_surface, cathode_color, (int(s*2.5), int(s*1.5)), 4)
    
    # Rotate the surface
    rotated_surface = pygame.transform.rotate(diode_surface, angle)
    
    # Get the rect and center it at the position
    rect = rotated_surface.get_rect(center=(x, y))
    S.blit(rotated_surface, rect)

def waves(p):
    grid()
    cx, cy = W//2, H//2
    
    # --- Input Wave (AC) - TOP (moved 40px down) ---
    box_y = 160  # Was 120, now 160 (40px down)
    draw.rect(S, C["DIODE"], (cx - 150, box_y - 55, 300, 115), 2)
    
    # Draw input wave with color coding
    for x in range(280):
        y_val = amp * math.sin(freq*(x + p))
        y_pos = box_y + y_val
        color = C["POSITIVE_HALF"] if y_val >= 0 else C["NEGATIVE_HALF"]
        draw.circle(S, color, (cx - 140 + x, int(y_pos)), 1)
    
    txt("AC Input Waveform", (cx, box_y - 90), C["IN"])
    # txt("Live", (cx - 50, box_y + 80), C["IN"])
    # txt("Neutral", (cx + 50, box_y + 80), C["IN"])

    # --- Output Wave (Full-Wave DC) - BOTTOM (moved 40px up) ---
    box_y = H - 160  # Was H - 120, now H - 160 (40px up)
    draw.rect(S, C["DIODE"], (cx - 150, box_y - 70, 300, 115), 2)
    
    # Draw output wave with color coding based on original input phase
    for x in range(280):
        y_val = amp * math.sin(freq*(x + p))
        y_pos = box_y - amp * abs(math.sin(freq*(x + p)))
        color = C["POSITIVE_HALF"] if y_val >= 0 else C["NEGATIVE_HALF"]
        draw.circle(S, color, (cx - 140 + x, int(y_pos)), 1)
    
    txt("DC Output Waveform", (cx, box_y - 90), C["OUT"])
    txt("-", (cx - 50, box_y + 80), C["OUT"])
    txt("+", (cx + 50, box_y + 80), C["OUT"])

    # --- Bridge Rectifier Circuit - CENTER ---
    center_x, center_y = cx, H//2
    radius = 120
    
    # Define the four points of the diamond
    top = (center_x, center_y - radius)      # Live input
    right = (center_x + radius, center_y)    # Positive output
    bottom = (center_x, center_y + radius)   # Neutral input
    left = (center_x - radius, center_y)     # Negative output

    # Determine current half cycle for coloring
    current_sine = math.sin(freq * (140 + p))  # Sample middle of waveform
    is_positive_half = current_sine >= 0
    active_color = C["POSITIVE_HALF"] if is_positive_half else C["NEGATIVE_HALF"]

    # Draw all wires in dim color first
    draw.line(S, C["WIRE"], top, right, 3)      # Top to right
    draw.line(S, C["WIRE"], right, bottom, 3)   # Right to bottom  
    draw.line(S, C["WIRE"], bottom, left, 3)    # Bottom to left
    draw.line(S, C["WIRE"], left, top, 3)       # Left to top

    # Highlight active current paths with Z-shape
    if is_positive_half:
        # Positive half cycle: Live → D1 → Positive → Load → Negative → D4 → Neutral
        # This forms a Z-shape: top-right to bottom-left
        draw.line(S, active_color, top, right, 5)      # Live to Positive via D1
        draw.line(S, active_color, left, bottom, 5)    # Negative to Neutral via D4
    else:
        # Negative half cycle: Neutral → D3 → Positive → Load → Negative → D2 → Live
        # This forms the opposite Z-shape: bottom-right to top-left
        draw.line(S, active_color, bottom, right, 5)   # Neutral to Positive via D3
        draw.line(S, active_color, left, top, 5)       # Negative to Live via D2

    # Diodes with UPDATED directions (D2 and D3 reversed):
    # D1: Top-right - Live to Positive (normal direction)
    # D2: Top-left - Negative to Live (REVERSE direction - NOW NORMAL)
    # D3: Bottom-right - Neutral to Positive (normal direction - NOW REVERSE)  
    # D4: Bottom-left - Negative to Neutral (REVERSE direction)
    
    # Only color the two conducting diodes
    if is_positive_half:
        # Positive half cycle: D1 and D4 conduct (ONLY these two are colored)
        draw_diode_with_polarity(((top[0] + right[0])/2, (top[1] + right[1])/2), angle=135, color=active_color, active=True)  # D1 (normal)
        draw_diode_with_polarity(((left[0] + top[0])/2, (left[1] + top[1])/2), angle=225)  # D2 (inactive - FIXED: removed color)
        draw_diode_with_polarity(((right[0] + bottom[0])/2, (right[1] + bottom[1])/2), angle=45, reverse=True)  # D3 (now reverse - was normal)
        draw_diode_with_polarity(((bottom[0] + left[0])/2, (bottom[1] + left[1])/2), angle=315, color=active_color, active=True, reverse=True)  # D4 (reverse)
    else:
        # Negative half cycle: D2 and D3 conduct (ONLY these two are colored)
        draw_diode_with_polarity(((top[0] + right[0])/2, (top[1] + right[1])/2), angle=135)  # D1 (inactive)
        draw_diode_with_polarity(((left[0] + top[0])/2, (left[1] + top[1])/2), angle=225, color=active_color, active=True)  # D2 (now normal - was reverse)
        draw_diode_with_polarity(((right[0] + bottom[0])/2, (right[1] + bottom[1])/2), angle=45, color=active_color, active=True, reverse=True)  # D3 (now reverse - was normal)
        draw_diode_with_polarity(((bottom[0] + left[0])/2, (bottom[1] + left[1])/2), angle=315, reverse=True)  # D4 (inactive)

    # Label diodes
    txt("D1", (center_x + radius*0.7, center_y - radius*0.3), C["WIRE"])
    txt("D2", (center_x - radius*0.7, center_y - radius*0.3), C["WIRE"])
    txt("D3", (center_x + radius*0.7, center_y + radius*0.3), C["WIRE"])
    txt("D4", (center_x - radius*0.7, center_y + radius*0.3), C["WIRE"])

    # AC Input connections - SHORT 20px wires with labels
    # Live wire - short line upward from top point
    wire_color_live = active_color if (is_positive_half) else C["WIRE"]
    draw.line(S, wire_color_live, top, (top[0], top[1] - 20), 5 if is_positive_half else 3)
    draw.circle(S, C["IN"], (top[0], top[1] - 25), 6, 2)
    txt("L", (top[0], top[1] - 40), C["IN"])
    
    # Neutral wire - short line downward from bottom point
    wire_color_neutral = active_color if (not is_positive_half) else C["WIRE"]
    draw.line(S, wire_color_neutral, bottom, (bottom[0], bottom[1] + 20), 5 if not is_positive_half else 3)
    draw.line(S, C["IN"], (bottom[0] - 5, bottom[1] + 25), (bottom[0] + 5, bottom[1] + 25), 2)
    txt("N", (bottom[0], bottom[1] + 40), C["IN"])

    # DC Output connections - SHORT 20px wires with labels
    # Positive output - short line to right from right point
    draw.line(S, active_color, right, (right[0] + 20, right[1]), 5)
    txt("-", (right[0] + 35, right[1]), C["OUT"])
    
    # Negative output - short line to left from left point
    draw.line(S, active_color, left, (left[0] - 20, left[1]), 5)
    txt("+", (left[0] - 35, left[1]), C["OUT"])

    # Load resistor - placed INSIDE bridge width
    load_width, load_height = 60, 40
    load_x, load_y = center_x, center_y
    
    # Draw load resistor with active color
    draw.rect(S, active_color, (load_x - load_width//2, load_y - load_height//2, 
                            load_width, load_height), 3)
    txt("Load", (load_x, load_y), active_color)
    
    # Connect load to bridge - short connections with active color
    draw.line(S, active_color, (right[0] - 3, right[1]), 
              (load_x + load_width//2, load_y), 5)
    draw.line(S, active_color, (left[0] + 3, left[1]), 
              (load_x - load_width//2, load_y), 5)

    # Add current flow animation (small moving dots) - REVERSED DIRECTION
    dot_phase = (p * 10) % 20
    dot_size = 6
    if is_positive_half:
        # Positive half current flow: Live → D1 → Positive → Load → Negative → D4 → Neutral
        if dot_phase < 10:
            # First segment: Live to Positive via D1 (REVERSED: now from Positive to Live)
            progress = 1 - (dot_phase / 10)  # Reverse progress
            dot_x = top[0] + (right[0] - top[0]) * progress
            dot_y = top[1] + (right[1] - top[1]) * progress
            draw.circle(S, active_color, (int(dot_x), int(dot_y)), dot_size)
        else:
            # Second segment: Negative to Neutral via D4 (REVERSED: now from Neutral to Negative)
            progress = 1 - ((dot_phase - 10) / 10)  # Reverse progress
            dot_x = left[0] + (bottom[0] - left[0]) * progress
            dot_y = left[1] + (bottom[1] - left[1]) * progress
            draw.circle(S, active_color, (int(dot_x), int(dot_y)), dot_size)
    else:
        # Negative half current flow: Neutral → D3 → Positive → Load → Negative → D2 → Live
        if dot_phase < 10:
            # First segment: Neutral to Positive via D3 (REVERSED: now from Positive to Neutral)
            progress = 1 - (dot_phase / 10)  # Reverse progress
            dot_x = bottom[0] + (right[0] - bottom[0]) * progress
            dot_y = bottom[1] + (right[1] - bottom[1]) * progress
            draw.circle(S, active_color, (int(dot_x), int(dot_y)), dot_size)
        else:
            # Second segment: Negative to Live via D2 (REVERSED: now from Live to Negative)
            progress = 1 - ((dot_phase - 10) / 10)  # Reverse progress
            dot_x = left[0] + (top[0] - left[0]) * progress
            dot_y = left[1] + (top[1] - left[1]) * progress
            draw.circle(S, active_color, (int(dot_x), int(dot_y)), dot_size)

    # Add title and description
    txt("Full Wave Bridge Rectifier", (cx, 50), C["WIRE"])
    txt("AC to DC Conversion", (cx, 90), (200, 200, 200))
    
    # Add connection indicators (adjusted positions)
    txt("AC Input", (cx, center_y - radius - 60), C["IN"])
    txt("DC Output", (cx, center_y + radius + 60), C["OUT"])
    
    # Show current path description
    if is_positive_half:
        path_text = "Current Path (Blue): Live → D1 → (+) → Load → (-) → D4 → Neutral"
    else:
        path_text = "Current Path (Red): Neutral → D3 → (+) → Load → (-) → D2 → Live"
    txt(path_text, (cx, H - 50), active_color)

# --- Main Loop ---
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT or \
           (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            pygame.quit(); sys.exit()

    S.fill(C["BG"])
    waves(phase)
    phase += speed
    pygame.display.flip()
    clk.tick(60)
```


## How to Run

1. Install **Pygame** if you haven’t already:
   ```bash
   pip install pygame
   ```
2. Save the script as `rectifier_sim.py`.
3. Run it:
   ```bash
   python rectifier_sim.py
   ```
4. Press **[ESC]** to quit.



## Key Learnings

- How a **Full Wave Bridge Rectifier** works.
- How to **simulate electric circuits visually** using Python.
- Using **Pygame for animation** and dynamic drawing.
- Representing **waveforms and logic** through geometry and color.



## Further Ideas

Here are some fun ways to extend the project:

- Add a **filter capacitor** to smooth the output waveform.
- Display a **moving average** of the output voltage.
- Add **sound effects** when current changes direction.
- Show **voltage vs time graph** side-by-side with the circuit.



## Credits

Developed by **PyShine** — bringing electronics and Python to life through visual learning!

If you found this helpful, consider subscribing to [PyShine on YouTube](https://www.youtube.com/@pyshine_official)  for more fun Python projects.



**Happy Coding & Keep Learning! ⚡**

