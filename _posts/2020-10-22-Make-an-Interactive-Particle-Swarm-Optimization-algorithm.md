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


Hello friends, today we will use Matplotlib in Python to make an interactive PSO environment, where you can change the target as well as 
the number of particles including their various parameters.

# IMPORTANT 

Please note that the files below are intended for Python 3, and not for Python2. Use Matplotlib version 2.2.4

1) Draggable.py
2) main.py

Just make a new directory and place both these .py files. Then simply run the main.py. Lets see what's inside the Draggable.py

## Draggable.py
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

## main.py
{% include codeHeader.html %}
```python
## Welcome to PyShine

import random
import math
import numpy as np
import csv, os
import _thread
import matplotlib.pyplot as plt

# ============================================================
# SAFETY FLAG (ADDED)
# ============================================================
running = True

def on_close(event):
    global running
    print("Figure closed — breaking loop")
    running = False

# ============================================================
# ORIGINAL CODE (UNCHANGED)
# ============================================================
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
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            return row[0], row[1]


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

        # ====================================================
        # SAFETY: CREATE FIGURE + CLOSE HANDLER (ADDED)
        # ====================================================
        plt.ion()
        fig = plt.figure()
        fig.canvas.mpl_connect('close_event', on_close)

        i=0
        while True:

            # 🔴 SAFETY ESCAPE (ADDED)
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




