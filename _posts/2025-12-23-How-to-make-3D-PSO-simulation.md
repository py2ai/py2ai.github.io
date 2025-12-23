---
layout: post
title: "Interactive 3D PSO with a Draggable Target in Python"
description: "A beginner-friendly tutorial explaining how to build an interactive 3D Particle Swarm Optimization (PSO) demo using Matplotlib, keyboard & mouse events, and inter-process communication via CSV."
featured-img: 20251223-interactive-pso/20251223-interactive-pso
keywords:
- Python
- Matplotlib
- PSO
- Particle Swarm Optimization
- Interactive visualization
- Beginner tutorial
---

# Interactive 3D PSO with a Draggable Target in Python

This tutorial explains **two Python scripts working together**:

- `Draggable.py` → an interactive 2D window where you drag a target and adjust its **Z value**
- `main.py` → a **3D Particle Swarm Optimization (PSO)** simulation that continuously chases that target

This is a powerful example of **interactive optimization**, **real-time visualization**, and **process communication** in Python.

---

## Big Picture: How Everything Works

```
Mouse / Keyboard
      ↓
Draggable.py  ──► target.csv  ──► main.py (PSO)
      ↑                                   ↓
   Visual UI                    3D Particles Follow Target
```

- You drag a point in 2D (X, Y)
- Press **U / D** to move Z up or down
- The target position is written to `target.csv`
- The PSO loop continuously reads that file
- Particles move toward the live target

This pattern is extremely useful for **human-in-the-loop optimization**.

---

## File 1: Draggable.py (Interactive Target Controller)

### Purpose
- Lets the user **drag a target with the mouse**
- Shows live **X, Y, Z values** on screen
- Writes the target position to a CSV file

### Key Concepts Used
- Matplotlib mouse events
- Keyboard events
- Text overlays
- Safe exit handling

### Important Sections Explained

#### 1. Clean Exit Handling
```python
def cleanup(*_):
    plt.close("all")
    sys.exit(0)
```
This ensures the window closes cleanly when you press **Ctrl+C** or close the app.

---

#### 2. Interactive Target
```python
target = [10.0, -10.0, 0.0]
```
This represents **X, Y, Z**. Only X & Y are draggable; Z is controlled by keys.

---

#### 3. Mouse Drag Logic
```python
def on_motion(event):
    if not pressed or event.inaxes != ax:
        return
    target[0], target[1] = event.xdata, event.ydata
    update()
```
This allows smooth dragging inside the plot area.

---

#### 4. Keyboard Control for Z Axis
```python
def on_key(event):
    if event.key == 'u':
        target[2] += 10
    elif event.key == 'd':
        target[2] -= 10
    update()
```
This adds **3D control using a 2D UI**.

---

#### 5. Live Text Overlay
```python
txt.set_text(
    f"x: {target[0]:.1f}\n"
    f"y: {target[1]:.1f}\n"
    f"z: {target[2]:.1f}"
)
```
Users always see exact coordinates — extremely helpful for debugging and demos.

---

## File 2: main.py (Interactive PSO Engine)

### Purpose
- Runs a **3D Particle Swarm Optimization** loop
- Continuously reads the target from `target.csv`
- Visualizes particles chasing a moving target

---

## What is PSO (Quick Theory)

Particle Swarm Optimization is inspired by **bird flocks and fish schools**.

Each particle:
- Has a position
- Has a velocity
- Moves toward a target using momentum + attraction

Formula intuition:
```
new_velocity = inertia + attraction_to_target
new_position = old_position + new_velocity
```

---

## Particle Class Explained

```python
class Particle:
    def evaluate(self):
        self.error = fitness(self.pos)
```
Each particle computes **distance to the target**.

```python
self.vel[i] = w*self.vel[i] + c*random.random()*(target[i]-self.pos[i])
```
- `w` → inertia (smooth motion)
- `c` → attraction strength

---

## Why CSV Is Used (And Why It’s Smart)

Using `target.csv` is:

✔ Simple
✔ Debuggable
✔ Language-agnostic
✔ Easy to visualize

This technique is common in:
- Robotics
- Simulation systems
- Multi-process AI experiments

---

## Why This Project Is Important

This is **not just a demo** — it teaches:

- Event-driven programming
- Real-time visualization
- Optimization algorithms
- Inter-process communication
- Human-in-the-loop control

Many beginners never see these ideas combined.

---

## Practical Use Cases

1. **AI parameter tuning with human guidance**
2. **Robotics target following simulations**
3. **Game AI movement systems**
4. **Research demos & visual explanations**
5. **Teaching optimization interactively**

---

## Complete Code

### Draggable.py

{% include codeHeader.html %}

```python
## Welcome to PyShine
import matplotlib.pyplot as plt
import csv
import signal, sys

TARGET_FILE = "target.csv"

#  CLEAN EXIT 
def cleanup(*_):
    plt.close("all")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

#  FIGURE 
fig, ax = plt.subplots(figsize=(4, 4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)
ax.set_title("Drag Target | U / D = Z")

#  TARGET 
target = [10.0, -10.0, 0.0]

pt, = ax.plot(
    [target[0]], [target[1]],
    'ko', markersize=10
)

# TEXT OVERLAY (TOP-LEFT, AXES COORDS)
txt = ax.text(
    0.02, 0.95,
    "",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
)

#  FILE WRITE 
def write_target():
    with open(TARGET_FILE, "w", newline="") as f:
        csv.writer(f).writerow(target)

#  UPDATE 
def update():
    pt.set_data([target[0]], [target[1]])   # sequence-safe
    txt.set_text(
        f"x: {target[0]:.1f}\n"
        f"y: {target[1]:.1f}\n"
        f"z: {target[2]:.1f}"
    )
    write_target()
    fig.canvas.draw_idle()

# initial write + text
update()

#  EVENTS 
pressed = False

def on_press(event):
    global pressed
    if event.inaxes == ax:
        pressed = True

def on_release(event):
    global pressed
    pressed = False

def on_motion(event):
    if not pressed or event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return
    target[0], target[1] = event.xdata, event.ydata
    update()

def on_key(event):
    if event.key == 'u':
        target[2] += 10
    elif event.key == 'd':
        target[2] -= 10
    update()

#  CONNECT 
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()

```

### main.py

{% include codeHeader.html %}

```python
## Welcome to PyShine
import random, csv, os, _thread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

#  SAFETY 
running = True

def on_close(event):
    global running
    print("Figure closed — breaking loop")
    running = False

with open("target.csv","w") as f:
    f.write("0,0,0")

def start_drag():
    os.system("python Draggable.py")

_thread.start_new_thread(start_drag, ())

#  UTILS 
def getXYZ(file):
    try:
        with open(file) as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    return float(row[0]), float(row[1]), float(row[2])
    except:
        pass
    return 0.0, 0.0, 0.0

def fitness(pos):
    tx,ty,tz = getXYZ("target.csv")
    return (tx-pos[0])**2 + (ty-pos[1])**2 + (tz-pos[2])**2

#  PARTICLE 
class Particle:
    def __init__(self, initial, color, pid):
        self.pos = initial[:]
        self.vel = [random.uniform(-5,5) for _ in initial]
        self.color = color
        self.id = pid
        self.error = float('inf')

    def evaluate(self):
        self.error = fitness(self.pos)
        print(f"Particle {self.id:02d} ERROR -----> {self.error:.2f}")
        return self.error

    def update(self, target, bounds):
        w = 0.7
        c = 1.8
        for i in range(len(self.pos)):
            self.vel[i] = w*self.vel[i] + c*random.random()*(target[i]-self.pos[i])
            self.pos[i] += self.vel[i]
            self.pos[i] = max(bounds[i][0],min(bounds[i][1],self.pos[i]))

#  PSO 
class Interactive_PSO:
    def __init__(self, initial, bounds, num_particles, colormap="hsv"):
        global running

        cmap = plt.get_cmap(colormap)
        colors = [cmap(i/num_particles) for i in range(num_particles)]

        swarm = [
            Particle(initial, colors[i], i)
            for i in range(num_particles)
        ]

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        fig.canvas.mpl_connect("close_event", on_close)

        step = 0

        while running:
            ax.clear()
            ax.set_xlim(-500,500)
            ax.set_ylim(-500,500)
            ax.set_zlim(-500,500)

            target = list(getXYZ("target.csv"))

            global_best_error = float('inf')

            print(f"\n=== ITERATION {step} ===")
            print(f"TARGET ---> {target}")

            for p in swarm:
                err = p.evaluate()
                if err < global_best_error:
                    global_best_error = err

            for p in swarm:
                p.update(target, bounds)
                ax.scatter(*p.pos, c=[p.color], s=20)

            ax.scatter(*target, c='k', s=150)

            ax.set_title(
                "PyShine Interactive PSO (3D) | Particles:{} | Error:{:.2f}"
                .format(num_particles, global_best_error)
            )

            plt.pause(0.01)
            step += 1

        print("\nResults")
        print("Final Target:", getXYZ("target.csv"))
        print("Final Best Error:", global_best_error)
        plt.close('all')

#  RUN 
Interactive_PSO(
    initial=[5,5,5],
    bounds=[(-800,800)]*3,
    num_particles=16,     # any number
    colormap="hsv"
)

```

## Common Questions (FAQ)

### Q: Why not use sockets or shared memory?
CSV is simpler, safer, and perfect for teaching concepts first.

---

### Q: Can this be extended to real AI models?
Yes. Replace `fitness()` with a real evaluation function.

---

### Q: Can I control velocity or PSO parameters live?
Absolutely — add keyboard bindings just like Z control.

---

### Q: Why is this powerful?
Because **you are steering an optimizer in real time**.

---

## Final Thoughts

This project beautifully combines:

- Visualization
- Optimization
- Interaction
- Clean Python design

It’s an excellent **advanced-beginner to intermediate** Python project and a great foundation for AI demos.

---

Happy coding!

Built with ❤️ by PyShine

