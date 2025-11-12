---
description: '> A detailed, step-by-step tutorial explaining a Pygame program that draws a Fibonacci-based tree. This tutorial shows how the original recursive growth work...'
featured-img: 26072022-python-logo
keywords:
- recursion
- PyGame
layout: post
mathjax: true
tags:
- fibonacci
- draw tree
- random tree
title: Recursive function to grow TREE in Python
---
# Fibonacci Tree Growth (Multithreaded)

> A detailed, step-by-step tutorial explaining a Pygame program that draws a Fibonacci-based tree. This tutorial shows how the original recursive growth works and how to modify it so branches grow in parallel using multithreading (safe and practical approach).

## Table of contents

## Table of Contents

- [Introduction](#1-introduction)
- [Prerequisites](#2-prerequisites)
- [Project Structure](#3-project-structure)
- [Full Original Code](#4-simple-code)
- [Deep Explanation — Line by Line and Concept by Concept](#5-deep-explanation--line-by-line-and-concept-by-concept)
- [Why and When to Parallelize](#6-why-and-when-to-parallelize)
- [Design Choices for Multithreading](#7-design-choices-for-multithreading)
- [Multithreaded Implementation — Full Code](#8-multithreaded-implementation--full-code)
- [How to Run and Test](#9-how-to-run-and-test)
- [Performance Considerations &amp; Debugging Tips](#10-performance-considerations--debugging-tips)
- [FAQ and Closing Notes](#11-faq-and-closing-notes)

## 1. Introduction

This tutorial teaches you how a small Pygame program simulates a fractal, Fibonacci-based tree. Branch lengths and spreads follow Fibonacci-derived ratios so the resulting shape looks natural and organic.

You'll see:

- How the recursive `grow()` function builds branches.
- Why some branches look thicker or greener.
- How to convert the growth stage into parallel tasks so multiple branches can be computed at the same time, using Python threads.

The tutorial ends with a complete multithreaded version ready to copy and run.

## 2. Prerequisites

- Python 3.8+ (works well with 3.10 / 3.11)
- `pygame` installed (`pip install pygame`)
- Basic knowledge of Python functions and threading concepts
- Optional: `concurrent.futures` familiarity

## 3. Project structure

Single file: `fibonacci_tree_parallel.py` . Run it with:

```bash
python fibonacci_tree_parallel.py
```

## 4. Simple code

{% include codeHeader.html %}

```python
import pygame, sys, math, random

## INITIAL SETUP 
pygame.init()

WIN_WIDTH, WIN_HEIGHT = 1200, 600
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Fibonacci Tree Growth")

## COLOR DEFINITIONS 
SKY_COLOR   = (135, 206, 235)
BROWN_COLOR = (139, 69, 19)
GREEN_COLOR = (34, 139, 34)

## TREE PARAMETERS 
GROUND_HEIGHT   = 50
TREE_DEPTH      = 10
BRANCH_WIDTH    = 20
SPREAD_BASE     = 25
SPREAD_FACTOR   = 70         # slightly larger spread for wider crown
LENGTH_DECAY    = 0.75
LENGTH_VARIANCE = 0.1
BRANCH_STEP     = 3
FPS             = 60

## FONT 
font = pygame.font.SysFont(None, 28, bold=True)

## GLOBALS 
branches = []
draw_step = 0
started = False

## FIBONACCI FUNCTION 
def fib(n):
    """Return the nth Fibonacci number."""
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a

## TREE GENERATION 
def grow(x, y, length, angle, depth, width):
    """
    Recursive Fibonacci-based branching function.
    Ensures branches stay mostly above the horizon and spread outward naturally.
    """
    if depth <= 0 or length < 4:
        return

    rad = math.radians(angle)
    end_x = x + math.sin(rad) * length
    end_y = y - math.cos(rad) * length

    ## prevent branches from going below the ground
    if end_y > WIN_HEIGHT - GROUND_HEIGHT:
        end_y = WIN_HEIGHT - GROUND_HEIGHT

    ## add this branch
    branches.append(((x, y), (end_x, end_y), width, depth))

    ## Fibonacci ratio scaling
    f_ratio = fib(depth + 2) / fib(TREE_DEPTH + 2)
    spread = SPREAD_BASE + SPREAD_FACTOR * f_ratio

    ## number of child branches
    sub_branches = 2 + fib(depth) % 3

    ## nonlinear upward bias — keeps branches from pointing downward
    for _ in range(sub_branches):
        new_length = length * (LENGTH_DECAY + LENGTH_VARIANCE * f_ratio)
      
        ## bias the angle upwards: restrict below horizontal (no downward growth)
        bias = random.uniform(-spread, spread)
        new_angle = angle + bias

        ## Clamp angles: ensure they stay above -90 (horizontal) and below 90
        new_angle = max(-70, min(70, new_angle))

        new_width = max(3, int(width * 0.75))
        grow(end_x, end_y, new_length, new_angle, depth - 1, new_width)

## INITIAL TREE CREATION 
INITIAL_LENGTH = (WIN_HEIGHT - 120) // (TREE_DEPTH * 0.8)
grow(WIN_WIDTH // 2, WIN_HEIGHT - GROUND_HEIGHT, INITIAL_LENGTH * 2, 0, TREE_DEPTH, BRANCH_WIDTH)

## CLOCK 
clock = pygame.time.Clock()

## MAIN LOOP 
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            started = True

    screen.fill(SKY_COLOR)
    pygame.draw.rect(screen, GREEN_COLOR, (0, WIN_HEIGHT - GROUND_HEIGHT, WIN_WIDTH, GROUND_HEIGHT))

    if not started:
        msg = font.render("CLICK TO START", True, GREEN_COLOR)
        screen.blit(msg, (WIN_WIDTH // 2 - msg.get_width() // 2, WIN_HEIGHT // 2))
    else:
        draw_step = min(draw_step + BRANCH_STEP, len(branches))
        for (start, end, width, depth) in branches[:draw_step]:
            if depth > 4:
                color = BROWN_COLOR
            else:
                color = tuple(
                    int(BROWN_COLOR[i] * ((depth - 1) / 4) + GREEN_COLOR[i] * (1 - (depth - 1) / 4))
                    for i in range(3)
                )
            pygame.draw.line(screen, color, start, end, width)

        text = font.render(f"BRANCH COUNT: {draw_step}", True, GREEN_COLOR)
        screen.blit(text, (WIN_WIDTH - text.get_width() - 20, 10))

    pygame.display.flip()
    clock.tick(FPS)
```

## 5. Deep explanation — line by line and concept by concept

I'll walk through the important pieces and why they exist.

### a) Initialization and constants

- `pygame.init()` starts Pygame.
- `WIN_WIDTH`, `WIN_HEIGHT` — screen size.
- Color constants defined as RGB tuples.
- Tree parameters such as `TREE_DEPTH`, `BRANCH_WIDTH`, `SPREAD_BASE`, etc., control how deep the recursion goes, how thick branches start, and how widely they spread.

**Tip:** Keep `TREE_DEPTH` moderate (7–12). Too large creates many branches and can slow rendering and thread overhead.

### b) Fibonacci function

```python
def fib(n):
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a
```

- Returns the nth Fibonacci number (1-indexed here).
- Used to compute ratios that change with depth — making lower/higher branches differ organically.

### c) Recursive growth: `grow()`

`grow(x, y, length, angle, depth, width)` does the heavy lifting.

Key steps inside:

1. Base case: stop when depth <= 0 or length is tiny.
2. Compute end coordinates using trigonometry:
   - `end_x = x + sin(angle) * length`
   - `end_y = y - cos(angle) * length`
3. Prevent branches dipping below the ground by clamping `end_y`.
4. Append the branch to the `branches` list: `branches.append(((x, y), (end_x, end_y), width, depth))`.
5. Compute Fibonacci-based scale `f_ratio` and `spread` to determine how much children diverge.
6. Decide `sub_branches` — number of children, varied with Fibonacci.
7. Loop through children, derive `new_length`, `new_angle` (biased upwards), clamp angles, reduce width, then recursively call `grow()` with `depth - 1`.

This recursion builds a flattened list `branches` which is later drawn progressively in the main loop.

### d) Drawing loop and `draw_step`

- `branches` is precomputed once by `grow(...)` starting at the trunk base.
- Once user clicks, `draw_step` increases and the program draws `branches[:draw_step]` to animate growth.
- Color interpolation between brown and green depends on depth to simulate branch → leaf transition.

## 6. Why and when to parallelize

The algorithm is CPU-heavy when `TREE_DEPTH` is high — recursively creating many branches. Parallelizing the **generation** (not the drawing) can bring speed benefits on multicore machines.

**Important**: Pygame drawing calls must happen in the main thread on many platforms. So you parallelize only the computationally heavy part: generating branch geometry (`branches` list). Drawing stays single-threaded.

When to parallelize:

- If `TREE_DEPTH >= 11` or you notice long pauses at startup while branches are computed.
- If you want quicker precomputation before animating the growth.

## 7. Design choices for multithreading

Options:

- `threading.Thread` and manual queue/lock management.
- `concurrent.futures.ThreadPoolExecutor` — simpler, higher-level.

Constraints & safety:

- The shared `branches` list must be protected by a `threading.Lock` during append operations.
- Avoid creating thousands of concurrent threads. Use a bounded pool (`max_workers = min(32, os.cpu_count() or 4)`) to limit resource usage.
- Keep recursion depth per thread modest; better approach: have each submitted task compute an entire subtree (e.g., `grow()` from a given node down to a shallow depth) and append the resulting local list to the global list under a lock.

Strategy implemented below:

- The main `grow_parallel()` will push top-level child `grow` tasks into an executor.
- Each worker runs a variant `grow_worker()` that returns a local list of branches for that subtree.
- The main thread collects futures and merges results into the global `branches` once each future completes (safe merging under lock).

This design reduces lock contention (threads only lock briefly to append a chunk) and keeps Pygame drawing safe.

## 8. Multithreaded implementation — full code

Copy this file as `fibonacci_tree_parallel.py` and run it. This includes the parallel `generate_tree()` which uses `ThreadPoolExecutor` and a safe `branches` merge.

{% include codeHeader.html %}

```python
import pygame, sys, math, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os

## INITIAL SETUP
pygame.init()
WIN_WIDTH, WIN_HEIGHT = 1200, 600
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Fibonacci Tree Growth (Multithreaded)")

## COLORS
SKY_COLOR   = (135, 206, 235)
BROWN_COLOR = (139, 69, 19)
GREEN_COLOR = (34, 139, 34)

## TREE PARAMETERS
GROUND_HEIGHT   = 50
TREE_DEPTH      = 10
BRANCH_WIDTH    = 20
SPREAD_BASE     = 25
SPREAD_FACTOR   = 70
LENGTH_DECAY    = 0.75
LENGTH_VARIANCE = 0.1
BRANCH_STEP     = 3
FPS             = 60

font = pygame.font.SysFont(None, 28, bold=True)

## GLOBALS
branches = []             # global branch list used by the renderer
branches_lock = threading.Lock()  # protect writes to `branches`
started = False

## FIB
def fib(n):
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a

## Worker-grow produces a local list and returns it (no shared writes).
def grow_worker(x, y, length, angle, depth, width, tree_depth):
    local = []
    def _g(x, y, length, angle, depth, width):
        if depth <= 0 or length < 4:
            return
        rad = math.radians(angle)
        end_x = x + math.sin(rad) * length
        end_y = y - math.cos(rad) * length
        if end_y > WIN_HEIGHT - GROUND_HEIGHT:
            end_y = WIN_HEIGHT - GROUND_HEIGHT
        local.append(((x, y), (end_x, end_y), width, depth))

        f_ratio = fib(depth + 2) / fib(tree_depth + 2)
        spread = SPREAD_BASE + SPREAD_FACTOR * f_ratio
        sub_branches = 2 + fib(depth) % 3

        for _ in range(sub_branches):
            new_length = length * (LENGTH_DECAY + LENGTH_VARIANCE * f_ratio)
            bias = random.uniform(-spread, spread)
            new_angle = angle + bias
            new_angle = max(-70, min(70, new_angle))
            new_width = max(3, int(width * 0.75))
            _g(end_x, end_y, new_length, new_angle, depth - 1, new_width)

    _g(x, y, length, angle, depth, width)
    return local

## Multithreaded generation that submits the trunk's immediate children as tasks.
def generate_tree_parallel(root_x, root_y, initial_length, root_depth, root_width):
    ## We'll compute the trunk synchronously, then spawn worker tasks for each top-level child subtree.
    global branches
    branches = []

    ## Add trunk (depth=root_depth)
    rad = math.radians(0)
    trunk_end_x = root_x + math.sin(rad) * initial_length
    trunk_end_y = root_y - math.cos(rad) * initial_length
    if trunk_end_y > WIN_HEIGHT - GROUND_HEIGHT:
        trunk_end_y = WIN_HEIGHT - GROUND_HEIGHT
    branches.append(((root_x, root_y), (trunk_end_x, trunk_end_y), root_width, root_depth))

    ## Prepare tasks for child subtrees emerging from trunk_end
    f_ratio = fib(root_depth + 2) / fib(TREE_DEPTH + 2)
    spread = SPREAD_BASE + SPREAD_FACTOR * f_ratio
    sub_branches = 2 + fib(root_depth) % 3

    ## Decide worker pool size
    max_workers = min(32, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _ in range(sub_branches):
            new_length = initial_length * (LENGTH_DECAY + LENGTH_VARIANCE * f_ratio)
            bias = random.uniform(-spread, spread)
            new_angle = max(-70, min(70, bias))
            new_width = max(3, int(root_width * 0.75))
            ## Submit each subtree
            futures.append(executor.submit(grow_worker, trunk_end_x, trunk_end_y, new_length, new_angle, root_depth - 1, new_width, TREE_DEPTH))

        ## As futures complete, merge their local lists into the global branches list with minimal locking.
        for fut in as_completed(futures):
            local_branches = fut.result()
            with branches_lock:
                branches.extend(local_branches)

## INITIAL TREE CREATION (parallel)
INITIAL_LENGTH = (WIN_HEIGHT - 120) // (TREE_DEPTH * 0.8)
## generate using parallel generator
generate_tree_parallel(WIN_WIDTH // 2, WIN_HEIGHT - GROUND_HEIGHT, INITIAL_LENGTH * 2, TREE_DEPTH, BRANCH_WIDTH)

## CLOCK
clock = pygame.time.Clock()

## MAIN LOOP
clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            started = True

    screen.fill(SKY_COLOR)
    pygame.draw.rect(screen, GREEN_COLOR, (0, WIN_HEIGHT - GROUND_HEIGHT, WIN_WIDTH, GROUND_HEIGHT))

    if not started:
        msg = font.render("CLICK TO START", True, GREEN_COLOR)
        screen.blit(msg, (WIN_WIDTH // 2 - msg.get_width() // 2, WIN_HEIGHT // 2))
    else:
        ## animate growth
        draw_step = min(len(branches), (pygame.time.get_ticks() // 10))
        for (start, end, width, depth) in branches[:draw_step]:
            if depth > 4:
                color = BROWN_COLOR
            else:
                color = tuple(
                    int(BROWN_COLOR[i] * ((depth - 1) / 4) + GREEN_COLOR[i] * (1 - (depth - 1) / 4))
                    for i in range(3)
                )
            pygame.draw.line(screen, color, start, end, width)

        text = font.render(f"BRANCH COUNT: {min(draw_step, len(branches))}", True, GREEN_COLOR)
        screen.blit(text, (WIN_WIDTH - text.get_width() - 20, 10))

    pygame.display.flip()
    clock.tick(FPS)
```

**Notes about the code above**:

- `grow_worker()` performs the recursion only *locally* and returns a `local` list of branches rather than appending to a global list while computing. This avoids heavy lock contention.
- `generate_tree_parallel()` handles trunk creation synchronously, then schedules multiple subtree computations via `ThreadPoolExecutor`. When a future completes, its result is merged into `branches` under `branches_lock`.
- The renderer still draws from `branches` on the main thread.

## 9. How to run and test

1. Save the file as `fibonacci_tree_parallel.py`.
2. Install pygame if necessary: `pip install pygame`.
3. Run: `python fibonacci_tree_parallel.py`.
4. Click the window to start the growth animation.

**Testing tips**:

- Try small/large `TREE_DEPTH` values to observe performance changes.
- Toggle `max_workers` or the strategy to see CPU/time tradeoffs.

## 10. Performance considerations & debugging tips

- Python threads are limited by the Global Interpreter Lock (GIL) for pure-Python CPU-bound tasks. However, splitting heavy recursion into multiple threads often helps if some operations release the GIL (not the case here) OR if overhead (waiting for random numbers, I/O) exists. Still, for pure CPU-bound tasks, `multiprocessing` may give better scaling. The thread approach helps eliminate GUI freeze because branch generation tasks are off the main thread.
- If you want true parallel CPU utilization for heavy depth, consider `multiprocessing` instead. It requires serializing results between processes (e.g., `multiprocessing.Pool`), but it avoids the GIL.
- Keep `branches` merging minimal and batched. Appending many single branches under a lock causes contention. Returning a local list and merging it once avoids this.
- On Windows, `pygame` must run in the main thread. Don't attempt to call Pygame display or drawing from worker threads.

## 11. FAQ and closing notes

**Q: Why not call `grow()` directly in multiple threads (no `local` lists)?**
A: That creates lock contention (every branch append needs a lock) and increases the chance of subtle bugs. Using local results and merging is safer and faster.

**Q: Can I parallelize drawing?**
A: No — drawing must stay in the main thread for portability and correctness.

**Q: When should I use `multiprocessing` instead?**
A: Use it when `TREE_DEPTH` is large (e.g., 12+) and CPU usage is the bottleneck. It has higher overhead but avoids the GIL.

### Final thoughts

This tutorial shows how to understand the original recursive fractal generator and how to safely run expensive geometry generation in parallel without touching the GUI thread. Use the provided code as a starting point — tweak parameters, depths, and the pool strategy to match your CPU and artistic goal.

Happy coding and happy tree-growing!
