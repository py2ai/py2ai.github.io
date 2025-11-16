---
description: Python Turtle Graphics Tutorial to animate falling cherry blossoms
featured-img: 20251109-blossoms3
keywords:
- Python
- Turtle
- Cherry Blossom
- Animation
- Graphics
layout: post
mathjax: false
tags:
- python
- turtle
- animation
- graphics
- nature
title: Make a Tree with falling Flowers
---

## Overview

This tutorial expands on our previous **Cherry Blossom Tree** by adding **animated falling petals** that gently sway in the wind and settle on the ground.
We’ll use `turtle` graphics and a lightweight physics simulation with gravity, wind, and smooth animation updates.

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/QAFBXJtgIOc" 
    title="YouTube video player" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
  </iframe>
</div>

--- 

## Setup and Imports

```python
import turtle
from random import random, uniform, seed
import math

## === Settings ===
W, H = 420, 420
DEPTH = 8
LENGTH = 80
BA = 25
GRAVITY = 1.5
WIND = 2.0
FPS = 0.02
seed(192)
```

We initialize the screen size, branch depth, and animation constants for wind and gravity.

---

## Generating Branch Parameters

Before drawing, we create random parameters for branch angles and lengths.
This ensures a natural look while keeping the structure reproducible.

```python
branch_params = []
def record_my_params(length, depth, level=0):
    if depth == 0:
        return
    a1 = uniform(BA - 10, BA + 10)
    a2 = uniform(BA - 10, BA + 10)
    l1 = length * uniform(0.6, 0.8)
    l2 = length * uniform(0.6, 0.8)
    branch_params.append((a1, a2, l1, l2, level))
    record_my_params(l1, depth - 1, level + 1)
    record_my_params(l2, depth - 1, level + 1)

record_my_params(LENGTH, DEPTH)

def param_gen():
    for p in branch_params:
        yield p
```

The `record_my_params()` function recursively stores all angles and lengths before we start drawing.

---

## Tree and Flower Drawing Functions

We use helper functions to **blend colors**, **draw flowers**, and **build branches** recursively.

```python
def blend(c1, c2, t):
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
    )

def draw_flower(pen, x, y, color, size, rotation=0):
    pen.penup()
    pen.goto(x, y)
    pen.setheading(rotation)
    pen.pendown()
    pen.fillcolor(color)
    pen.pencolor(color)
    pen.begin_fill()
    for _ in range(5):
        pen.circle(size, 72)
        pen.left(144)
    pen.end_fill()
    pen.penup()
    pen.goto(x, y)
    pen.dot(size * 0.6, (0.9, 0.4, 0.6))
```

Each blossom is drawn using a **five-petal shape** with subtle pink tones.

---

## Animating Falling Blossoms

After building the tree, we animate petals detaching, swaying in the wind, and landing softly.

```python
## Main animation loop
while True:
    frame_count += 1
    tree_pen.clear()
    fall_pen.clear()
    draw_tree()

    breeze_phase += breeze_frequency
    current_breeze = math.sin(breeze_phase) * breeze_strength

    ## Detach blossoms randomly
    for f in flowers:
        if f[4] and frame_count > 10 and random() < 0.04:
            f[4] = False
            velocities[id(f)] = [
                uniform(-WIND * 0.7, WIND * 0.7) + current_breeze * 0.4,
                -GRAVITY * uniform(0.8, 1.2)
            ]
            rotations[id(f)] = uniform(0, 360)
            fallen_count += 1
            print(f"🌸 dropped! Total fallen: {fallen_count}/{len(flowers)}")

    ## Move and draw each blossom
    ...
```

The simulation uses **gravity and sinusoidal wind motion** for realistic drifting.

---

## Complete Code

{% include codeHeader.html %}

```python
import turtle
from random import random, uniform, seed
import math

## === Settings ===
W, H = 420, 420
DEPTH = 8
LENGTH = 80
BA = 25
GRAVITY = 1.5
WIND = 2.0
FPS = 0.02
seed(192)

## === Pre-generate random branch params ===
branch_params = []
def record_my_params(length, depth, level=0):
    if depth == 0:
        return
    a1 = uniform(BA - 10, BA + 10)
    a2 = uniform(BA - 10, BA + 10)
    l1 = length * uniform(0.6, 0.8)
    l2 = length * uniform(0.6, 0.8)
    branch_params.append((a1, a2, l1, l2, level))
    record_my_params(l1, depth - 1, level + 1)
    record_my_params(l2, depth - 1, level + 1)
record_my_params(LENGTH, DEPTH)

def param_gen():
    for p in branch_params:
        yield p


def turtle_tree():
    screen = turtle.Screen()
    screen.setup(W, H)
    screen.title("Turtle Tree with Falling Flowers")
    screen.bgcolor("black")
    screen.tracer(0, 0)

    tree_pen = turtle.Turtle()
    tree_pen.hideturtle()
    tree_pen.speed(0)

    fall_pen = turtle.Turtle()
    fall_pen.hideturtle()
    fall_pen.speed(0)
    fall_pen.penup()

    flowers = []  # [x, y, color, size, attached]
    fallen_count = 0  # <--- NEW counter

    def blend(c1, c2, t):
        return (
            c1[0] + (c2[0] - c1[0]) * t,
            c1[1] + (c2[1] - c1[1]) * t,
            c1[2] + (c2[2] - c1[2]) * t,
        )

    def draw_flower(pen, x, y, color, size, rotation=0):
        pen.penup()
        pen.goto(x, y)
        pen.setheading(rotation)
        pen.pendown()
        pen.fillcolor(color)
        pen.pencolor(color)
        pen.begin_fill()
        for _ in range(5):
            pen.circle(size, 72)
            pen.left(144)
        pen.end_fill()
        pen.penup()
        pen.goto(x, y)
        pen.dot(size * 0.6, (0.9, 0.4, 0.6))

    def draw_tree():
        tree_pen.penup()
        tree_pen.goto(0, -H//2 + 30)
        tree_pen.setheading(90)
        tree_pen.pendown()
        params = param_gen()

        def draw_branch(length, depth):
            if depth == 0:
                return
            t = (DEPTH - depth) / DEPTH
            brown = (139/255, 69/255, 19/255)
            green = (34/255, 139/255, 34/255)
            r, g, b = blend(brown, green, t)
            tree_pen.pencolor(r, g, b)
            tree_pen.pensize(max(1, depth / 2))
            tree_pen.forward(length)
            try:
                a1, a2, l1, l2, level = next(params)
            except StopIteration:
                return
            tree_pen.left(a1)
            draw_branch(l1, depth - 1)
            tree_pen.right(a1 + a2)
            draw_branch(l2, depth - 1)
            tree_pen.left(a2)
            tree_pen.backward(length)

        draw_branch(LENGTH, DEPTH)

    ## === Build tree & record flower positions ===
    tree_pen.penup()
    tree_pen.goto(0, -H//2 + 30)
    tree_pen.setheading(90)
    params = param_gen()

    def add_flowers(length, depth):
        if depth == 0:
            size = uniform(2, 5)
            color = (1.0, uniform(0.6, 0.8), uniform(0.7, 0.9))
            flowers.append([tree_pen.xcor(), tree_pen.ycor(), color, size, True])
            return
        tree_pen.forward(length)
        if random() < 0.12 and depth < DEPTH - 2:
            size = uniform(2, 5)
            color = (1.0, uniform(0.6, 0.8), uniform(0.7, 0.9))
            flowers.append([tree_pen.xcor(), tree_pen.ycor(), color, size, True])
        try:
            a1, a2, l1, l2, level = next(params)
        except StopIteration:
            return
        tree_pen.left(a1)
        add_flowers(l1, depth - 1)
        tree_pen.right(a1 + a2)
        add_flowers(l2, depth - 1)
        tree_pen.left(a2)
        tree_pen.backward(length)

    add_flowers(LENGTH, DEPTH)
    print(f"Tree complete with {len(flowers)} blossoms")

    velocities = {}
    rotations = {}
    frame_count = 0
    breeze_phase = 0
    breeze_strength = 0.8
    breeze_frequency = 0.08

    while True:
        frame_count += 1
        tree_pen.clear()
        fall_pen.clear()
        draw_tree()

        breeze_phase += breeze_frequency
        current_breeze = math.sin(breeze_phase) * breeze_strength

        ## Detach blossoms
        for f in flowers:
            if f[4] and frame_count > 10 and random() < 0.04:
                f[4] = False
                velocities[id(f)] = [
                    uniform(-WIND * 0.7, WIND * 0.7) + current_breeze * 0.4,
                    -GRAVITY * uniform(0.8, 1.2)
                ]
                rotations[id(f)] = uniform(0, 360)
                fallen_count += 1  # increment counter
                print(f"🌸 dropped! Total fallen: {fallen_count}/{len(flowers)}")

        all_fallen = True
        for f in flowers:
            x, y, color, size, attached = f
            if attached:
                sway_x = current_breeze * 0.1 * math.sin(frame_count * 0.2 + x * 0.01)
                draw_flower(tree_pen, x + sway_x, y, color, size, current_breeze * 2)
                all_fallen = False
            else:
                flower_id = id(f)
                vx, vy = velocities.get(flower_id, (0, 0))
                x += vx + current_breeze * 0.05
                y += vy
                vy -= GRAVITY * 0.08
                vx += current_breeze * 0.015 + uniform(-0.008, 0.008)
                vx *= 0.998
                vy *= 0.998
                if flower_id in rotations:
                    rotations[flower_id] += vx * 3
                ground_level = -H//2 + size * 1.5
                if y <= ground_level:
                    y = ground_level
                    if abs(vy) > 0.3:
                        vy = -vy * 0.2
                    else:
                        vy = 0
                        vx *= 0.8
                velocities[flower_id] = [vx, vy]
                f[0], f[1] = x, y
                rotation = rotations.get(flower_id, 0)
                draw_flower(fall_pen, x, y, color, size, rotation)
                if vy != 0 or vx != 0:
                    all_fallen = False

        screen.update()
        ## time.sleep(FPS)
        if all_fallen and frame_count > 60:
            break

    print(f"All {fallen_count} flowers have fallen!")
    turtle.done()


if __name__ == "__main__":
    turtle_tree()

```

---

## How It Works

- **Tree Drawing:** Recursive function generates symmetrical branches.
- **Blossom Attachment:** Each flower starts attached to a branch.
- **Falling Motion:** Randomly detaches and moves downward using basic gravity physics.
- **Wind:** Adds horizontal sway using a sine wave pattern.
- **Collision Detection:** Flowers stop gently upon reaching the ground.

---

## Run the Script

Save as `falling_cherry_tree.py` and run:

```bash
python3 falling_cherry_tree.py
```

You’ll see a black background with blossoms drifting gracefully from the branches and settling at the base.

---

## Conclusion

This project brings life to your fractal tree by adding **motion and realism**.
Experiment with `WIND`, `GRAVITY`, and `FPS` to create slow drifting or stormy blossom effects 🌸💨
Next, try combining this animation with **sound or seasonal background music** for a serene interactive art piece!

---

**Website:** https://www.pyshine.com
**Author:** PyShine
