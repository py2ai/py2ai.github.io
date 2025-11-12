---
description: Learn how to build a beautiful recursive fractal tree using Python’s turtle graphics module with randomness for natural effects.
featured-img: 26072022-python-logo
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
title: Fractal Tree Generator in Python with Turtle
---
# Fractal Tree Generator in Python

## A Beginner’s Guide to Recursion and Turtle Graphics

In this tutorial, you’ll learn how to create a **beautiful fractal tree** using **Python’s turtle graphics module**. The project combines **recursion**, **randomness**, and **geometry** to draw realistic, organic-looking trees.

---

## Table of Contents

- [Overview](#overview)
- [Project Setup](#project-setup)
- [Recursive Tree Logic](#recursive-tree-logic)
- [Randomization for Natural Shapes](#randomization-for-natural-shapes)
- [Drawing with Turtle](#drawing-with-turtle)
- [Complete Code](#complete-code)
- [How to Run](#how-to-run)
- [Key Learnings](#key-learnings)
- [Further Experiments](#further-experiments)

---

## Overview

Fractals are **self-similar geometric patterns** that repeat at smaller scales. Trees are a perfect example — each branch splits into smaller branches that resemble the whole tree.

In this project, we’ll use **recursive functions** to draw a tree, where each branch spawns two smaller branches at random angles and lengths.

Here’s what we’ll build:

- Recursive branching up to a defined depth
- Randomized angles and lengths for organic look
- A vibrant turtle graphics visualization

---

## Project Setup

First, let’s import the required modules and set up some global parameters:

```python
import turtle, random

## Screen dimensions
W, H = 400, 400

## Tree configuration
DEPTH = 8      # Levels of recursion (branch generations)
LENGTH = 80    # Average branch length
BA = 25        # Average branch angle in degrees

random.seed(192)  # Fix the random seed for reproducibility
```

---

## Recursive Tree Logic

The magic of this fractal tree lies in recursion — a function that calls itself to create smaller branches.

We’ll use a helper function called `record_my_params()` to **generate and store random parameters** for branch angles and lengths at every recursive level.

```python
branch_params = []

def record_my_params(length, depth, level=0):
    """Recursively generate random branch parameters"""
    if depth == 0: return

    ## Random left and right angles
    a1 = random.uniform(BA-10, BA+10)
    a2 = random.uniform(BA-10, BA+10)

    ## Random left and right branch lengths
    l1 = length * random.uniform(0.6, 0.8)
    l2 = length * random.uniform(0.6, 0.8)

    ## Store the parameters for reuse
    branch_params.append((a1, a2, l1, l2, level))

    ## Recurse for next levels
    record_my_params(l1, depth-1, level+1)
    record_my_params(l2, depth-1, level+1)

## Initialize all branch parameters once
record_my_params(LENGTH, DEPTH)
```

This step ensures that even if we re-run the drawing, we use **the same random tree structure**, resulting in reproducibility.

---

## Randomization for Natural Shapes

In nature, no two branches are exactly alike — randomness is key to realism.

Here’s what we randomized:

- **Branch Angles (`a1`, `a2`)** → vary around ±10° from the base angle `BA`.
- **Branch Lengths (`l1`, `l2`)** → randomly between 60% and 80% of the parent branch.

This small variation makes the generated tree look **organic and lifelike** rather than geometric.

---

## Drawing with Turtle

Now, we’ll use the **Turtle Graphics** library to draw the tree recursively.

```python
def param_gen():
    for p in branch_params:
        yield p

def turtle_tree():
    screen = turtle.Screen()
    screen.setup(W, H)
    screen.title("Turtle Tree")
    screen.bgcolor("black")

    pen = turtle.Turtle()
    pen.hideturtle()
    pen.speed(0)
    pen.color("lime")

    ## Start position and orientation
    pen.left(90)
    pen.penup()
    pen.goto(0, -H//2 + 30)
    pen.pendown()

    params = param_gen()

    def draw_tree(length, depth):
        if depth == 0: return

        pen.pensize(max(1, depth/2))
        pen.forward(length)

        try:
            a1, a2, l1, l2, level = next(params)
        except StopIteration:
            return

        ## Left branch
        pen.left(a1)
        draw_tree(l1, depth - 1)

        ## Right branch
        pen.right(a1 + a2)
        draw_tree(l2, depth - 1)

        ## Return to original orientation and backtrack
        pen.left(a2)
        pen.backward(length)

        print(f'Level: {level} Depth: {depth}')

    draw_tree(LENGTH, DEPTH)
    print("Turtle finished drawing...")
    turtle.done()
```

When you run this, you’ll see your turtle gracefully draw the branches, recursively building the entire tree structure.

---

## Complete Code

Here’s the complete Python script for your fractal tree:

{% include codeHeader.html %}

```python
import turtle, random
W,H = 400, 400
DEPTH = 8 # Recursive level or generations of branch
LENGTH = 80 # Average branch length
BA = 25 # Branch Angle on average
random.seed(192) 
## Generate Random angle and length sequence
branch_params = []
def record_my_params(length, depth, level=0):
    """Recursively generate random branch parameters"""
    if depth == 0: return 
    ## Random branch angles (variation around BA)
    a1 = random.uniform(BA-10,BA+10) # left angle
    a2 = random.uniform(BA-10,BA+10) # right angle
    ## Random branch lenghts (60% to 80% of parent)
    l1 = length * random.uniform(0.6,0.8) # left length
    l2 = length * random.uniform(0.6,0.8) # right length
    ## Store parameters (for reuse later on)
    branch_params.append((a1,a2,l1,l2,level))
    ## Recurse deeper
    record_my_params(l1, depth-1, level+1)
    record_my_params(l2, depth-1, level+1)
## Call once to populate the branch_params list
record_my_params(LENGTH, DEPTH)
## Iterator to reuse identical random parameters
def param_gen():
    for p in branch_params: yield p 

def turtle_tree():
    screen = turtle.Screen(); screen.setup(W,H)
    screen.title("Turtle Tree")
    screen.bgcolor("black")
    pen = turtle.Turtle(); pen.hideturtle()
    pen.speed(0);pen.color("lime"); pen.left(90)
    pen.penup();pen.goto(0,-H//2+30);pen.pendown()
    params = param_gen()
    def draw_tree(length, depth):
        if depth == 0: return 
        pen.pensize(max(1,depth/2))
        pen.forward(length)
        try: a1,a2,l1,l2,level=next(params)
        except StopIteration: return
        pen.left(a1);draw_tree(l1,depth-1)
        pen.right(a1+a2);draw_tree(l2,depth-1)
        pen.left(a2); pen.backward(length)
        print(f'Level: {level} Depth: {depth}')
    draw_tree(LENGTH, DEPTH)
    print(f"Turtle finished drawing...")
    turtle.done()
if __name__ == "__main__":
    turtle_tree()
```

---

## How to Run

1. Install Python (3.8+ recommended).
2. Save the code as `turtle_tree.py`.
3. Run it in your terminal or IDE:
   ```bash
   python turtle_tree.py
   ```
4. Watch the tree being drawn branch by branch on a black canvas.

---

## Key Learnings

- **Recursion**: Each branch draws smaller branches by calling itself.
- **Randomization**: Adds organic variety to otherwise mathematical patterns.
- **Parameter Storage**: Precomputing random parameters ensures consistent results.
- **Turtle Graphics**: Provides a simple way to visualize geometric structures.

---

## Further Experiments

- Change colors for different depth levels (green → brown gradient)
- Add wind animation by slightly rotating angles in real-time
- Generate holiday trees with random decorations
- Try increasing `DEPTH` for more detailed fractals (but slower drawing)

---

With just a few lines of recursive code, you’ve created an **organic fractal tree** that grows beautifully on your screen. This project is a fun introduction to **recursive graphics and procedural generation** — concepts that power both **natural simulations** and **game development**!
