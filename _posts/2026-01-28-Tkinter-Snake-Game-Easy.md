---
layout: post
title: "Building a Snake Game in Python with Tkinter"
description: "A beginner-friendly tutorial explaining how to build a classic Snake game using Python and Tkinter."
featured-img: snakegame-2025-01-29/snakegame-2025-01-29
author: PyShine
tags: [python, tkinter, game-development, beginners]
---

# Building a Snake Game in Python with Tkinter

This tutorial walks you through a **complete Snake game** written in Python using **Tkinter**.  
It is designed for **beginners** who want to learn:

- GUI programming with Tkinter  
- Game loops and state management  
- Keyboard handling  
- Object‑oriented Python  

---

## Requirements

- Python 3.8+
- Tkinter (comes preinstalled with Python on most systems)

Run the game using:

```bash
python main.py
```

---

## Game Features

- Arrow‑key movement
- Pause / Resume (`P` key or button)
- Score tracking
- Start / Reset buttons
- Collision detection
- Wrap‑around walls
- Clean object‑oriented structure

---

## Core Concepts Explained

### The Direction Enum

```python
from enum import Enum

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
```

**Why this matters:**  
Enums prevent invalid directions and make movement logic readable and safe.

---

### Game Grid Logic

Each cell is a square in a 20×20 grid:

```python
self.cell_size = 20
self.grid_width = 20
self.grid_height = 20
```

The snake moves **one cell at a time**, not pixel‑by‑pixel — a common technique in grid‑based games.

---

### The Game Loop (`after()`)

```python
self.root.after(self.game_speed, self.move_snake)
```

Tkinter doesn’t have a traditional `while` loop for games.  
Instead, it schedules updates using `after()` — this keeps the UI responsive.

---

### Snake Movement Logic

```python
new_head = (
    (head_x + dx) % self.grid_width,
    (head_y + dy) % self.grid_height
)
```

**Why `%` modulo?**  
This allows the snake to **wrap around the screen**, instead of crashing into walls.

---

### Collision Detection

```python
if new_head in self.snake:
    self.game_over()
```

The game ends if the snake collides with itself — a classic Snake rule.

---

### Drawing with Canvas

```python
self.canvas.create_rectangle(...)
self.canvas.create_oval(...)
```

Tkinter’s `Canvas` lets you draw:
- Rectangles → Snake body
- Ovals → Food
- Text → Game Over screen

---

## Full Game Code

{% include codeHeader.html %}
```python

import tkinter as tk
import random
from enum import Enum

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class SnakeGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Snake Game")
        self.root.resizable(False, False)
        
        # Game settings
        self.cell_size = 20
        self.grid_width = 20
        self.grid_height = 20
        self.game_speed = 100  # milliseconds
        
        # Calculate window size
        self.canvas_width = self.grid_width * self.cell_size
        self.canvas_height = self.grid_height * self.cell_size
        
        # Create canvas
        self.canvas = tk.Canvas(
            root, 
            width=self.canvas_width, 
            height=self.canvas_height,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10)
        
        # Create score label
        self.score_label = tk.Label(root, text="Score: 0", font=("Arial", 14))
        self.score_label.pack()
        
        # Create buttons frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)
        
        # Control buttons
        self.start_button = tk.Button(button_frame, text="Start", command=self.start_game)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = tk.Button(button_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset_game)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Initialize game state
        self.reset_game()
        
        # Bind keyboard events
        self.root.focus_set()
        self.root.bind("<KeyPress>", self.handle_keypress)
        
        # Draw initial state
        self.draw()

    def reset_game(self):
        # Initial snake position (centered)
        start_x = self.grid_width // 2
        start_y = self.grid_height // 2
        self.snake = [(start_x, start_y)]
        
        # Initial direction
        self.direction = Direction.RIGHT
        self.next_direction = self.direction
        
        # Place first food
        self.place_food()
        
        # Reset score
        self.score = 0
        self.update_score()
        
        # Reset game state
        self.is_paused = False
        self.is_game_over = False
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        
        # Clear canvas and draw initial state
        self.canvas.delete(tk.ALL)
        self.draw()

    def place_food(self):
        while True:
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def move_snake(self):
        if self.is_paused or self.is_game_over:
            return
            
        # Update direction
        self.direction = self.next_direction
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_head = ((head_x + dx) % self.grid_width, (head_y + dy) % self.grid_height)
        
        # Check for collision with self
        if new_head in self.snake:
            self.game_over()
            return
            
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check if food is eaten
        if new_head == self.food:
            self.score += 10
            self.update_score()
            self.place_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()
            
        # Draw updated game state
        self.draw()
        
        # Schedule next move
        self.root.after(self.game_speed, self.move_snake)

    def handle_keypress(self, event):
        key = event.keysym
        if key == "Up" and self.direction != Direction.DOWN:
            self.next_direction = Direction.UP
        elif key == "Down" and self.direction != Direction.UP:
            self.next_direction = Direction.DOWN
        elif key == "Left" and self.direction != Direction.RIGHT:
            self.next_direction = Direction.LEFT
        elif key == "Right" and self.direction != Direction.LEFT:
            self.next_direction = Direction.RIGHT
        elif key.lower() == 'p':
            self.toggle_pause()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.config(text="Resume")
        else:
            self.pause_button.config(text="Pause")
            self.move_snake()

    def start_game(self):
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.move_snake()

    def game_over(self):
        self.is_game_over = True
        self.canvas.create_text(
            self.canvas_width // 2,
            self.canvas_height // 2,
            text=f"GAME OVER\nScore: {self.score}",
            fill="red",
            font=("Arial", 24),
            justify=tk.CENTER
        )
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)

    def update_score(self):
        self.score_label.config(text=f"Score: {self.score}")

    def draw(self):
        # Clear canvas
        self.canvas.delete(tk.ALL)
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = "lime green" if i == 0 else "green"  # Head is different color
            self.canvas.create_rectangle(
                x * self.cell_size,
                y * self.cell_size,
                (x + 1) * self.cell_size,
                (y + 1) * self.cell_size,
                fill=color,
                outline="dark green"
            )
        
        # Draw food
        fx, fy = self.food
        self.canvas.create_oval(
            fx * self.cell_size,
            fy * self.cell_size,
            (fx + 1) * self.cell_size,
            (fy + 1) * self.cell_size,
            fill="red",
            outline="dark red"
        )

if __name__ == "__main__":
    root = tk.Tk()
    game = SnakeGame(root)
    root.mainloop()
```


---

## Possible Extensions

Try improving the game by adding:

- Difficulty levels (speed increase)
- Sound effects
- High‑score saving (file or database)
- Walls and obstacles
- Color themes
- Multiplayer snake 👀

---

##  Common Questions (FAQ)

### Q: Why use Tkinter instead of Pygame?
Tkinter is simpler, built‑in, and perfect for learning GUI + logic fundamentals.

### Q: Why not use a `while` loop?
Tkinter uses an **event‑driven loop**. `after()` prevents freezing the UI.

### Q: How does pause work?
The game simply stops scheduling the next `move_snake()` call.

### Q: Can I turn off wrap‑around walls?
Yes! Replace modulo logic with boundary collision detection.

---

## Final Thoughts

This project teaches **real‑world Python skills**:
- Structuring programs
- Managing state
- Event‑driven programming
- GUI rendering

If you can build this — you’re officially **past beginner level** 🎉

Happy coding!
