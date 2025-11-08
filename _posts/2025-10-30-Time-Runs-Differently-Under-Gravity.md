---
description: Interactive simulation showing Earth Clock vs Gravity Clock and how gravity affects time.
featured-img: 26072022-python-logo
keywords:
- simulation
- PyGame
- gravity
- time dilation
layout: post
mathjax: true
tags:
- gravitational-time-dilation
- relativity
- python-simulation
- interactive
title: Gravitational Time Dilation Simulation in Python
---



# Gravitational Time Dilation Simulation with Pygame

## Educational Python Project – Visualize How Gravity Affects Time

This tutorial walks through building an **interactive Gravitational Time Dilation Simulation** using **Python and Pygame**. The simulation demonstrates how **time passes differently under varying gravitational strengths**, comparing a clock at Earth’s surface to a clock under stronger or weaker gravity.

## Table of Contents
- [Overview](#overview)
- [Theory: Gravitational Time Dilation](#theory-gravitational-time-dilation)
- [Pygame Setup](#pygame-setup)
- [Clock Display](#clock-display)
- [Gravity Adjustment](#gravity-adjustment)
- [Matplotlib Live Plot](#matplotlib-live-plot)
- [Relative Year Info](#relative-year-info)
- [Complete Python Code](#complete-python-code)
- [How to Run](#how-to-run)
- [Key Learnings](#key-learnings)
- [Further Ideas](#further-ideas)

## Overview

This project simulates **gravitational time dilation** using Python. You can interactively increase or decrease gravity and watch how the time on the gravity clock diverges from the Earth clock.

## Theory: Gravitational Time Dilation

According to general relativity, **time passes differently under different gravitational potentials**. A clock closer to a massive object ticks slower than a clock further away. The simulation uses the formula:

$$
\text{time factor} = \frac{1}{\sqrt{1 - \frac{2 G M}{r c^2}}}
$$



Where:
- \(G\) is the gravitational constant
- \(M\) is the mass of the object (Earth)
- \(r\) is the distance from the center of the mass
- \(c\) is the speed of light

## Pygame Setup

We initialize a Pygame window and define fonts for various display elements:

```python
pygame.init()
WIDTH, HEIGHT = 500, 280
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gravitational Time Dilation Clock")
```

Fonts are adjustable at the top:

```python
FONT_SIZE_INFO = 28
FONT_SIZE_CLOCK = 50
FONT_SIZE_SMALL = 40
font_info = pygame.font.SysFont(None, FONT_SIZE_INFO)
font_clock = pygame.font.SysFont(None, FONT_SIZE_CLOCK)
font_small = pygame.font.SysFont(None, FONT_SIZE_SMALL)
```

## Clock Display

The simulation shows two clocks:
- **Earth Clock**: baseline reference
- **Gravity Clock**: affected by gravity changes

## Gravity Adjustment

You can use the **UP and DOWN arrow keys** to increase or decrease gravity. The clocks update in real-time, and the relative rate is displayed.

## Matplotlib Live Plot

A live Matplotlib plot shows the **relative clock rate vs gravity**. The plot features:
- Blue line with sky-blue dots
- Gradient shadow under the curve
- Dynamic Y-axis

```python
line, = ax.plot([], [], color='blue', marker='o', markerfacecolor='skyblue', markersize=6, linestyle='-')
```

## Relative Year Info

The simulation also shows:

```
1 Year @ Earth = N Year @ Gravity Clock
```

This uses the `relative_rate` variable to dynamically display how 1 Earth year corresponds to a different duration on the gravity clock.

## Complete Python Code

```python
import pygame
import sys
import math
import time
import matplotlib.pyplot as plt

##  Constants 
G = 6.67430e-11           # Gravitational constant (m³/kg·s²)
c = 299792458             # Speed of light (m/s)
M = 5.972e24              # Mass of Earth (kg)
R_earth = 6.371e6         # Radius of Earth (m)
g_surface = G * M / R_earth**2  # Surface gravity of Earth (m/s²)
r_s = 2 * G * M / c**2    # Schwarzschild radius for Earth (m)
gf_earth = 1 / math.sqrt(1 - 2*G*M/(R_earth*c**2))  # Time dilation factor at Earth surface
SECONDS_PER_EARTH_YEAR = 365.25 * 24 * 3600  # Seconds in one Earth year

##  Adjustable font sizes 
FONT_SIZE_INFO = 28       # Info line font
FONT_SIZE_CLOCK = 50      # Clock fonts
FONT_SIZE_SMALL = 40      # Gravity / Relative Rate text

##  Pygame setup 
pygame.init()
WIDTH, HEIGHT = 500, 280
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gravitational Time Dilation Clock")
font_info = pygame.font.SysFont(None, FONT_SIZE_INFO)
font_clock = pygame.font.SysFont(None, FONT_SIZE_CLOCK)
font_small = pygame.font.SysFont(None, FONT_SIZE_SMALL)
clock = pygame.time.Clock()

##  Initial conditions 
gravity = g_surface
keys = {"up": False, "down": False}
t_earth = 0.0
t_gravity = 0.0
prev_time = time.time()

##  Matplotlib setup 
plt.ion()
fig, ax = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.tick_params(axis='x', colors='lime')
ax.tick_params(axis='y', colors='lime')
ax.xaxis.label.set_color('lime')
ax.yaxis.label.set_color('lime')
ax.title.set_color('lime')
fig.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.12)
fig.canvas.manager.set_window_title("Gravitational Time Dilation vs. Local Gravity")

line, = ax.plot([], [], color='blue', marker='o', markerfacecolor='skyblue', markersize=6, linestyle='-')
ax.set_xlabel("Gravity (m/s², log scale)")
ax.set_ylabel("Relative Rate (Earth Clock / Gravity Clock)")
ax.set_xscale('log')
ax.grid(True, color='gray', linestyle='--', alpha=0.5)

gravities = []
relative_rates = []

##  Function to calculate gravity factor safely 
def gravity_factor_from_g(g):
    r = max(math.sqrt(G*M/g), r_s * 1.0001)
    return 1 / math.sqrt(1 - 2*G*M/(r*c**2))

##  Format time function 
def format_time(t):
    t_int = int(t)
    h = t_int // 3600
    m = (t_int % 3600) // 60
    s = t_int % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

##  Main loop 
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                keys["up"] = True
            elif event.key == pygame.K_DOWN:
                keys["down"] = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                keys["up"] = False
            elif event.key == pygame.K_DOWN:
                keys["down"] = False

    ##  Compute delta time 
    current_time = time.time()
    delta_earth = current_time - prev_time
    prev_time = current_time

    ##  Update gravity 
    if keys["up"]:
        gravity *= 10**0.05
    if keys["down"]:
        gravity /= 10**0.05
    gravity = max(gravity, 1e-5)

    ##  Update clocks 
    t_earth += delta_earth / gf_earth
    gf = gravity_factor_from_g(gravity)
    t_gravity += delta_earth / gf
    relative_rate = gf_earth / gf

    ##  Update Matplotlib plot 
    if keys["up"] or keys["down"]:
        gravities.append(gravity)
        relative_rates.append(relative_rate)
        if len(gravities) > 30:
            gravities.pop(0)
            relative_rates.pop(0)
        line.set_data(gravities, relative_rates)

        ## Remove old gradient fills
        for coll in ax.collections:
            coll.remove()

        ## Gradient shadow under the curve
        alpha_max = 0.5
        n_layers = 10
        for i in range(n_layers):
            alpha = alpha_max * (i+1)/n_layers
            ax.fill_between(
                gravities,
                [r*(i/n_layers) for r in relative_rates],
                relative_rates,
                color='skyblue',
                alpha=alpha
            )

        ## Dynamic Y-axis limits
        ymin = min(relative_rates) - 0.1
        ymax = max(relative_rates) + 0.1
        ax.set_ylim(ymin, ymax)

        ax.relim()
        ax.autoscale_view(scalex=True, scaley=False)
        plt.tight_layout()
        plt.pause(0.001)

    ##  Compute 1 Year equivalence 
    gravity_years = relative_rate  # 1 Earth year = relative_rate Gravity Clock years
    info_str = f"1 Year @ Earth = {gravity_years:.6f} Year @ Gravity Clock"

    ##  Draw Pygame display 
    screen.fill((0, 0, 0))
    screen.blit(font_info.render(info_str, True, (255, 200, 0)), (20, 10))
    screen.blit(font_clock.render(f"Earth Clock: {format_time(t_earth)}", True, (255, 255, 0)), (20, 60))
    screen.blit(font_clock.render(f"Gravity Clock: {format_time(t_gravity)}", True, (0, 255, 255)), (20, 120))
    screen.blit(font_small.render(f"Gravity: {gravity:.2e} m/s²", True, (255, 100, 100)), (20, 180))
    screen.blit(font_small.render(f"Relative Rate: {relative_rate:.6f}", True, (0, 255, 0)), (20, 230))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()

```

## How to Run

1. Install required libraries:

```
pip install pygame matplotlib
```

2. Run the Python script:

```
python gravitational_time_dilation.py
```

3. Clikc on the PyGame window and then Use **UP/DOWN arrow keys** to change gravity interactively.

## Key Learnings

- Gravitational time dilation in practice
- Interactive simulations with Pygame
- Real-time plotting with Matplotlib
- Combining physics with visualization for education

## Further Ideas

- Add multiple objects with different masses
- Show a 3D representation of gravity wells
- Include relativistic effects at high velocities
- Export simulation data for further analysis
