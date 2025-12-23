---
categories:
- GUI tutorial series
description: Using Matplotlib drag the target of PSO and let the particles optimize their movements to find it.
featured-img: pso2
keywords:
- algorithm
- interactive
- development
- code
- programming
- pso
- tutorial
- python
layout: post
mathjax: true
title: How to make an interactive PSO algorithm in Python
---
<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/xEQv9YdvRiA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>

# Interactive Particle Swarm Optimization (PSO) with a Draggable Target

This tutorial explains an **interactive Particle Swarm Optimization (PSO)** project where particles dynamically move toward a **user-dragged target** on the screen. It combines optimization, visualization, and real-time user interaction — perfect for learning advanced Python concepts visually.

---

## What Is Particle Swarm Optimization (PSO)?

Particle Swarm Optimization is a **population-based optimization algorithm** inspired by how birds flock or fish school.

- Each **particle** is a possible solution
- Particles move through space
- They learn from:
  - Their **own best position**
  - The **best position found by the swarm**

Over time, particles converge toward an optimal solution.

---

## Project Overview

This project has **two Python programs running together**:

### PSO Engine (`main.py`)

- Simulates particles
- Computes fitness
- Updates velocities & positions
- Draws particles in real time

### Draggable Target (`Draggable.py`)

- Displays a draggable circle
- Writes its position to `target.csv`
- Acts as a **moving optimization goal**

Particles continuously chase the current target location.

---

## Key Concepts Used

- **Threads** → Run PSO and draggable window simultaneously
- **CSV file communication** → Share target position
- **Matplotlib animation** → Live particle movement
- **Event handling** → Mouse dragging + window close safety

---

## Code Architecture

```
main.py          → PSO simulation
Draggable.py    → Interactive target
target.csv      → Shared target position
```

---

## Safety & Stability Improvements

This project includes important safety mechanisms:

### Graceful Window Close

```python
running = True

def on_close(event):
    global running
    running = False
```

Prevents infinite loops when the window is closed.

### Signal Handling (Ctrl+C)

```python
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)
```

Ensures clean exits.

---

## The Particle Class Explained

Each particle has:

- `pos` → Current position
- `vel` → Velocity
- `best_pos` → Best personal position
- `best_error` → Best personal fitness

### Velocity Update Equation

```text
velocity = inertia
         + cognitive component (self learning)
         + social component (swarm learning)
```

This balances **exploration** and **convergence**.

---

## Fitness Function

```python
def fitness_function(x):
    x0, y0 = getXY('target.csv')
    return (x0 - x[0])**2 + (y0 - x[1])**2
```



✔ Measures **distance to the target**


✔ Lower value = better solution

---

## Interactive Target System

The draggable target:

- Uses `matplotlib.patches.Circle`
- Responds to mouse drag events
- Writes `(x, y)` to `target.csv` in real time

Particles instantly react to movement.

---

## Why This Project Is Important

- Turns abstract optimization into something **visual**
- Teaches **real-time feedback systems**
- Demonstrates **human-in-the-loop optimization**
- Bridges math, algorithms, and UI

---

## Real-World Use Cases

- Game AI (enemy tracking)
- Robotics (moving target following)
- Human-in-the-loop optimization
- Teaching optimization algorithms visually
- Calibrations

---

## Common Questions (FAQ)

### Why use a CSV file for communication?

It’s simple, cross-process, and beginner-friendly.

### Why reset global best periodically?

It prevents premature convergence and encourages exploration.

### Can this work in 3D?

Yes! Add a third dimension and use a 3D Matplotlib plot.

### Can I replace mouse dragging with keyboard input?

Absolutely — update the target position programmatically.

---

## Key Learning Outcomes

By studying this project, you learn:

- Optimization fundamentals
- Real-time visualization
- Threading basics
- Event-driven programming
- Clean shutdown techniques

---

Please note that the files below are intended for Python 3, and not for Python2. Use Matplotlib and Numpy

1) Draggable.py
2) main.py

Just make a new directory and place both these .py files. Then simply run the main.py. Lets see what's inside the Draggable.py

## Complete Code

### Draggable.py

{% include codeHeader.html %}

```python

## Welcome to PyShine
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import signal
import sys

TARGET_FILE = "target.csv"

def cleanup(*_):
    plt.close("all")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


# FIGURE SETUP (EXPLICIT WHITE)

fig, ax = plt.subplots(figsize=(4, 4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)
ax.set_title("Drag the Target")


# DRAGGABLE CLASS (NO BLITTING = NO BLACK SCREEN)
class Draggable_Target:
    lock = None

    def __init__(self, point):
        self.point = point
        self.press = None

    def connect(self):
        self.cidpress = self.point.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != ax:
            return
        contains, _ = self.point.contains(event)
        if not contains:
            return
        self.press = self.point.center, event.xdata, event.ydata
        Draggable_Target.lock = self

    def on_motion(self, event):
        if Draggable_Target.lock is not self or event.inaxes != ax:
            return

        (x0, y0), xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        self.point.center = (x0 + dx, y0 + dy)

        # write target position
        with open(TARGET_FILE, "w", newline="") as f:
            csv.writer(f).writerow(self.point.center)

        # full redraw (safe & fast enough)
        self.point.figure.canvas.draw_idle()

    def on_release(self, event):
        Draggable_Target.lock = None
        self.press = None


# CREATE TARGET
circle = patches.Circle((10, -10), 20, fc='black')
ax.add_patch(circle)

dr = Draggable_Target(circle)
dr.connect()

plt.show()

```

And here is the main.py, it will use the thread to call the above Draggable.py file. So please make sure to name the above file as Draggable, otherwise change the name accordingly in the code below under the start_drag function.

### main.py

{% include codeHeader.html %}

```python
## Welcome to PyShine

import random
import math
import numpy as np
import csv, os
import _thread
import matplotlib.pyplot as plt


# SAFETY FLAG 
running = True

def on_close(event):
    global running
    print("Figure closed — breaking loop")
    running = False


print(str(0)+','+str(0), file=open('target.csv','w'))

def start_drag():
    os.system('python Draggable.py')

_thread.start_new_thread(start_drag, ())

initial = [5,5]
bounds = [(-800,800),(-800,800)]

colors = np.array([
    (31,119,180),(174,199,232),(255,127,14),(255,187,120),
    (44,160,44),(152,223,138),(214,39,40),(255,152,150),
    (148,103,189),(197,176,213),(140,86,75),(196,156,148),
    (227,119,194),(247,182,210),(127,127,127),(199,199,199),
    (188,189,34),(219,219,141),(23,190,207),(158,218,229)
] * 4) / 255.0


class Particle: 
    def __init__(self, initial):
        self.pos=[]
        self.vel=[] 
        self.best_pos=[] 
        self.best_error=-1 
        self.error=-1   
        for i in range(0, num_dimensions): 
            self.vel.append(random.uniform(-1,1))
            self.pos.append(initial[i])
  
    def update_velocity(self, global_best_position): 
        w = 0.5
        c1 = 1 
        c2 = 2 
    
        for i in range(0, num_dimensions): 
            r1=random.random()
            r2=random.random()
            cog_vel=c1*r1*(self.best_pos[i]-self.pos[i])
            social_vel=c2*r2*(global_best_position[i]-self.pos[i])
            self.vel[i]=w*self.vel[i]+cog_vel+social_vel 
    
    def update_position(self, bounds): 
        for i in range(0, num_dimensions):
            self.pos[i]+=self.vel[i]
            if self.pos[i]>bounds[i][1]:
                self.pos[i]=bounds[i][1]
            if self.pos[i]<bounds[i][0]:
                self.pos[i]=bounds[i][0]
  
    def evaluate_fitness(self, fitness_function):
        self.error=fitness_function(self.pos) 
        print("ERROR------->", self.error)
        if self.error < self.best_error or self.best_error==-1:
            self.best_pos=self.pos 
            self.best_error=self.error


def fitness_function(x):
    x0,y0 = getXY('target.csv') 
    x0=float(x0)
    y0=float(y0)
    total=(x0-x[0])**2 +(y0-x[1])**2
    return total


def getXY(filename):
    try:
        with open(filename) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                if len(row) >= 2:
                    return row[0], row[1]
    except:
        pass
    return 0, 0



class Interactive_PSO():
    def __init__(self, fitness_function, initial, bounds, num_particles):
        global num_dimensions, running
    
        num_dimensions = len(initial) 
        global_best_error=-1         
        global_best_position=[] 
        self.gamma = 0.0001
        swarm=[]
        for i in range(0, num_particles):
            swarm.append(Particle(initial))

        plt.ion()
        fig = plt.figure()
        fig.canvas.mpl_connect('close_event', on_close)

        i=0
        while True:

            # SAFETY ESCAPE
            if not running:
                break
        
            for j in range(0, num_particles):
                swarm[j].evaluate_fitness(fitness_function)
                print('global_best_position', swarm[j].error, global_best_error)

                if swarm[j].error < global_best_error or global_best_error == -1:
                    global_best_position=list(swarm[j].pos) 
                    global_best_error=float(swarm[j].error)
                    plt.title(
                        "PyShine Interactive PSO, Particles:{}, Error:{}".format(
                            num_particles, round(global_best_error,1)
                        )
                    )
                
                if i%2==0:
                    global_best_error=-1
                    global_best_position = [
                        swarm[j].pos[0]+self.gamma*(swarm[j].error)*random.random(),
                        swarm[j].pos[1]+self.gamma*(swarm[j].error)*random.random()
                    ]
                
            pos_0 = {}
            pos_1 = {}
            for j in range(0, num_particles): 
                pos_0[j] = []
                pos_1[j] = []
        
            for j in range(0, num_particles): 
                swarm[j].update_velocity(global_best_position)
                swarm[j].update_position(bounds) 
                pos_0[j].append(swarm[j].pos[0])
                pos_1[j].append(swarm[j].pos[1])
                plt.xlim([-500, 500])
                plt.ylim([-500, 500])
            
            for j in range(0, num_particles):
                plt.plot(pos_0[j], pos_1[j], color=colors[j], marker='o')

            x,y = getXY('target.csv')	 
            plt.plot(float(x), float(y), color='k', marker='o')

            plt.pause(0.01)
            plt.clf() 
            i+=1

        print('Results')
        print('Best Position:', global_best_position)
        print('Best Error:', global_best_error)
        plt.close('all')


Interactive_PSO(fitness_function, initial, bounds, num_particles=16)

```

## Final Thoughts

This interactive PSO project transforms a classic optimization algorithm into a **living system** that responds instantly to user input. It’s an excellent foundation for more advanced AI, robotics, and simulation projects.

Once you understand this, you're no longer *just coding* — you're **building systems that react, adapt, and learn**.

---

Happy optimizing!
