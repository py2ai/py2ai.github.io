---
description: Python Turtle Graphics Tutorial to quickly draw a Tree
featured-img: 20251109-blossom2
keywords:
- Python
- Turtle
- Fractal Tree
- Recursion
- Graphics
layout: post
mathjax: false
tags:
- python
- turtle
- recursion
- graphics
- fractal
title: Make a Tree with Blossoms in Python
---

## Overview

This tutorial demonstrates how to create a **beautiful cherry blossom tree** using Python’s `turtle` graphics module.
The program uses recursion to generate realistic branches, randomization for natural variation, and color blending to simulate blooming flowers.

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/cyQ-63VKJAA" 
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

## Settings
W, H = 420, 420
DEPTH = 8
LENGTH = 80
BA = 25
seed(192)
```

We set up the screen dimensions, recursion depth, branch length, and base branch angle.

---

## Generating Random Branch Parameters

Before drawing, we pre-generate random angles and lengths for each branch.
This ensures consistent structure each time the script runs (because we fixed the random seed).

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

---

## Drawing Functions

Two helper functions are used:

- `blend()` for color blending
- `draw_flower()` for drawing small cherry blossoms

```python
def blend(c1, c2, t):
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
    )

def draw_flower():
    size = uniform(2, 5)
    color = (1.0, uniform(0.6, 0.8), uniform(0.7, 0.9))
    pen.fillcolor(color)
    pen.pencolor(color)
    pen.begin_fill()
    for _ in range(5):
        pen.circle(size, 72)
        pen.left(144)
    pen.end_fill()
    pen.dot(size * 0.6, (0.9, 0.4, 0.6))
```

---

## Main Tree Function

The recursive function `draw_branch()` handles branching, coloring, and flower placement.

```python
def draw_branch(length, depth):
    if depth == 0:
        draw_flower()
        return
    t = (DEPTH - depth) / DEPTH
    brown = (139/255, 69/255, 19/255)
    green = (34/255, 139/255, 34/255)
    r, g, b = blend(brown, green, t)
    pen.pencolor(r, g, b)
    pen.pensize(max(1, depth / 2))
    pen.forward(length)

    if random() < 0.12 and depth < DEPTH - 2:
        draw_flower()

    try:
        a1, a2, l1, l2, level = next(params)
    except StopIteration:
        return

    pen.left(a1)
    draw_branch(l1, depth - 1)
    pen.right(a1 + a2)
    draw_branch(l2, depth - 1)
    pen.left(a2)
    pen.backward(length)
    print(f'Level: {level} Depth: {depth}')
```

---

## Complete Code

{% include codeHeader.html %}

```python
import turtle
from random import random, uniform, seed

W, H = 420, 420
DEPTH = 8
LENGTH = 80
BA = 25
seed(192)

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
    screen.title("Turtle Tree with Flowers")
    screen.bgcolor("black")
    pen = turtle.Turtle()
    pen.hideturtle()
    pen.speed(0)
    pen.left(90)
    pen.penup()
    pen.goto(0, -H//2 + 30)
    pen.pendown()

    def blend(c1, c2, t):
        return (
            c1[0] + (c2[0] - c1[0]) * t,
            c1[1] + (c2[1] - c1[1]) * t,
            c1[2] + (c2[2] - c1[2]) * t,
        )

    def draw_flower():
        size = uniform(2, 5)
        color = (1.0, uniform(0.6, 0.8), uniform(0.7, 0.9))
        pen.fillcolor(color)
        pen.pencolor(color)
        pen.begin_fill()
        for _ in range(5):
            pen.circle(size, 72)
            pen.left(144)
        pen.end_fill()
        pen.dot(size * 0.6, (0.9, 0.4, 0.6))

    def draw_branch(length, depth):
        if depth == 0:
            draw_flower()
            return
        t = (DEPTH - depth) / DEPTH
        brown = (139/255, 69/255, 19/255)
        green = (34/255, 139/255, 34/255)
        r, g, b = blend(brown, green, t)
        pen.pencolor(r, g, b)
        pen.pensize(max(1, depth / 2))
        pen.forward(length)
        if random() < 0.12 and depth < DEPTH - 2:
            draw_flower()
        try:
            a1, a2, l1, l2, level = next(params)
        except StopIteration:
            return
        pen.left(a1)
        draw_branch(l1, depth - 1)
        pen.right(a1 + a2)
        draw_branch(l2, depth - 1)
        pen.left(a2)
        pen.backward(length)
        print(f'Level: {level} Depth: {depth}')

    params = param_gen()
    draw_branch(LENGTH, DEPTH)
    print("Tree complete.")
    turtle.done()

if __name__ == "__main__":
    turtle_tree()
```

---

## How It Works

- **Recursion:** Each branch splits into two sub-branches with random angles and lengths.
- **Color blending:** Gradually transitions from brown to green as branches move outward.
- **Flowers:** Small pink blossoms are drawn at branch tips and occasionally mid-branch.
- **Param reuse:** Pre-generated parameters keep the tree consistent on reruns.

---

## Run the Script

Save as `turtle_tree.py` and run:

```bash
python3 turtle_tree.py
```

A black screen with a growing cherry blossom tree should appear.

---

## Conclusion

This project combines **recursion**, **randomness**, and **color blending** to simulate organic growth.
You can modify parameters like `DEPTH`, `LENGTH`, and flower colors to create your own natural variations!

---

**Website:** https://www.pyshine.com
**Author:** PyShine
