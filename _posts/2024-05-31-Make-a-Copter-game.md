---
description: Learn how to let's build a copter game with this comprehensive Python tutorial.
featured-img: 26072022-python-logo
keywords:
- Python
- Pygame
- game development
- copter game
- game tutorial
- programming
- Python game
layout: post
mathjax: true
tags:
- Python
- Pygame
- game development
- copter game
- tutorial
title: Let's build a copter game
---



# Introduction
Games have always been a fascinating way to blend creativity with programming skills. In this tutorial, we will build a simple yet engaging helicopter game using Python's Pygame library. This game will help you understand the basics of game development, including handling user inputs, rendering graphics, and managing game states.

## What is Pygame?
Pygame is a set of Python modules designed for writing video games. It provides functionalities for creating graphics, handling input devices, and managing game events. Pygame is ideal for beginners who want to get started with game development due to its simplicity and ease of use.

## Game Concept
In our helicopter game, the player controls a helicopter that navigates through a series of obstacles. The objective is to avoid hitting the obstacles or the screen boundaries while trying to achieve the highest score possible. The helicopter can be moved up and down using the arrow keys.

## Getting Started
To start with this tutorial, you need to have Python and Pygame installed on your system. You can install Pygame using pip:

```
pip install pygame
```

## Step-by-Step Guide to Building the Game
## Setting Up Pygame
First, we need to initialize Pygame and set up the display window where our game will run.

{% include codeHeader.html %}
```python
import pygame
import random

## Initialize Pygame
pygame.init()

## Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Copter Game")

```

## Defining Constants and Colors
Next, we define the constants and colors used in the game. These include screen dimensions, game physics constants, obstacle properties, and colors for different game elements.
{% include codeHeader.html %}
```python
## Constants
FPS = 60
GRAVITY = 0.5
FLAP_STRENGTH = -10
OBSTACLE_WIDTH = 70
OBSTACLE_GAP = 200
OBSTACLE_SPEED = 5
UPWARD_SPEED = -5
DOWNWARD_SPEED = 5

## Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
SKY_BLUE = (135, 206, 235)
GREY = (169, 169, 169)
```
## Initializing Game Variables
We then initialize the game variables such as the helicopter's position, velocity, obstacles, score, and game state.

{% include codeHeader.html %}
```python
## Game variables
helicopter_rect = pygame.Rect(100, HEIGHT // 2, 42, 28)  # Reduced size to 70%
velocity = 0
obstacles = []
score = 0
game_over = False

## Fonts
font = pygame.font.Font(None, 36)

```

## Creating Obstacles
We define a function to create new obstacles. Obstacles consist of two rectangles (top and bottom) with a gap in between for the helicopter to pass through.

{% include codeHeader.html %}
```python
def create_obstacle():
    y = random.randint(100, HEIGHT - 100 - OBSTACLE_GAP)
    top_rect = pygame.Rect(WIDTH, 0, OBSTACLE_WIDTH, y)
    bottom_rect = pygame.Rect(WIDTH, y + OBSTACLE_GAP, OBSTACLE_WIDTH, HEIGHT - (y + OBSTACLE_GAP))
    return top_rect, bottom_rect

```

## Drawing the Helicopter
The draw_helicopter function renders the helicopter on the screen. The helicopter consists of several shapes, including an ellipse for the body, rectangles for the tail and landing skids, and more.
{% include codeHeader.html %}
```python
def draw_helicopter(rect):
    ## Body
    pygame.draw.ellipse(screen, RED, rect)
    
    ## Cockpit window
    cockpit_rect = pygame.Rect(rect.x + 14, rect.y + 7, 12, 7)  # Adjusted for size change
    pygame.draw.ellipse(screen, BLUE, cockpit_rect)
    
    ## Top rotor (oval-shaped with rotor block)
    top_rotor_rect = pygame.Rect(rect.x - 33, rect.y - 8, rect.width + 66, 6)  # Adjusted for size change
    pygame.draw.ellipse(screen, SKY_BLUE, top_rotor_rect)
    rotor_block_rect = pygame.Rect(rect.x + rect.width // 2 - 2, rect.y, 4, 4)  # Adjusted for size change
    pygame.draw.rect(screen, BLACK, rotor_block_rect)  # Changed propeller color
    
    ## Tail
    pygame.draw.rect(screen, GREY, (rect.x - 28, rect.y + rect.height // 4, 28, 7))  # Adjusted for size change
    
    ## Tail rotor
    pygame.draw.rect(screen, SKY_BLUE, (rect.x - 35, rect.y + rect.height // 4 - 4, 7, 14))  # Adjusted for size change
    
    ## Landing skids
    pygame.draw.rect(screen, BLACK, (rect.x + 7, rect.y + rect.height - 3, rect.width - 14, 3))  # Adjusted for size change
    pygame.draw.rect(screen, BLACK, (rect.x + 3, rect.y + rect.height, 3, 6))  # Adjusted for size change
    pygame.draw.rect(screen, BLACK, (rect.x + rect.width - 7, rect.y + rect.height, 3, 6))  # Adjusted for size change
```

## Main Game Loop
The main game loop handles the game logic, including updating the helicopter's position, generating obstacles, detecting collisions, and rendering everything on the screen.

{% include codeHeader.html %}
```python
## Main game loop
clock = pygame.time.Clock()

while True:
    clock.tick(FPS)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        velocity = UPWARD_SPEED
    elif keys[pygame.K_DOWN]:
        velocity = DOWNWARD_SPEED
    else:
        velocity += GRAVITY

    if not game_over:
        ## Apply gravity and control
        helicopter_rect.y += velocity
        
        ## Check for collisions with screen boundaries
        if helicopter_rect.top < 0 or helicopter_rect.bottom > HEIGHT:
            game_over = True
        
        ## Move and create obstacles
        for obstacle in obstacles:
            obstacle[0].x -= OBSTACLE_SPEED
            obstacle[1].x -= OBSTACLE_SPEED
        if len(obstacles) == 0 or obstacles[-1][0].x < WIDTH - 300:
            obstacles.append(create_obstacle())
        if obstacles[0][0].x < -OBSTACLE_WIDTH:
            obstacles.pop(0)
            score += 1
        
        ## Check for collisions with obstacles
        for top_rect, bottom_rect in obstacles:
            if helicopter_rect.colliderect(top_rect) or helicopter_rect.colliderect(bottom_rect):
                game_over = True
    
    ## Draw everything
    screen.fill(WHITE)
    draw_helicopter(helicopter_rect)
    
    for top_rect, bottom_rect in obstacles:
        pygame.draw.rect(screen, GREEN, top_rect)
        pygame.draw.rect(screen, GREEN, bottom_rect)
    
    ## Display score
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))
    
    if game_over:
        game_over_text = font.render("Game Over! Press R to Restart! Fly with Arrow Keys (Up/Down)", True, BLACK)
        screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2))
    
    pygame.display.flip()

```
## Restarting the Game
We add logic to restart the game when the 'R' key is pressed:

{% include codeHeader.html %}
```python

    ## Restart game on pressing 'R'
    if game_over and keys[pygame.K_r]:
        helicopter_rect.y = HEIGHT // 2
        velocity = 0
        obstacles.clear()
        score = 0
        game_over = False
```

## Conclusion
Congratulations! You have created a simple helicopter game using Pygame. This tutorial covered the basics of setting up a game window, handling user input, rendering graphics, and managing game states. With these foundations, you can further enhance the game by adding new features such as sound effects, advanced graphics, and more complex obstacle patterns.

