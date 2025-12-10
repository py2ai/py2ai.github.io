---
title: How to Make a Zombie Shooter Game in Pygame (Beginner Tutorial)
description: Learn how to build a simple Zombie Shooter game step-by-step using Pygame. Perfect for beginners who want to start making games in Python!
featured-img: 26072022-python-logo
layout: post
mathjax: true
tags:
  - zombie shooter game
  - pygame tutorial
  - pygame basic tutorial
keywords:
  - zombie shooter game
  - pygame tutorial
  - pygame basic tutorial
---


Welcome to this tutorial where we’ll be building a simple "Zombie Shooter" game using Pygame! This game involves a player controlling a shooter at the bottom of the screen, shooting bullets upwards to eliminate incoming zombies. The goal is to survive as long as possible while accumulating points by shooting zombies. 

By the end of this tutorial, you’ll have a fully working arcade-style shooter game.
Let’s get started!
---

## What We'll Build

In this game:

- Use **Left/Right arrow keys** to move  
- Press **Spacebar** to shoot  
- Avoid letting zombies reach the bottom  
- Earn points by shooting zombies  
- Beat your **top score**, saved in a local file  

Here is a preview:

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/yCKxrxnGAj0" 
    title="YouTube video player"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

---

## Game Breakdown (Step-by-Step)

### 1. **Game Initialization**
We start by:

- Initializing Pygame  
- Creating the main window  
- Defining colors  
- Loading assets (player + zombie images)  

This sets the stage for everything else.

---

### 2. **Player Setup**
- The player sprite sits at the bottom of the screen  
- It moves left and right  
- Movement is handled using the arrow keys  

We store the player's position using `player_x`, `player_y`.

---

### 3. **Bullet Mechanics**
Bullets:

- Spawn at the player's current location  
- Move upward each frame  
- Disappear when leaving the screen  

We store bullets as a list of `[x, y]` positions.

---

### 4. **Zombie Mechanics**
Zombies:

- Spawn at random x-positions at the top  
- Move downward at a constant speed  
- Trigger game over if they reach the bottom  

We store zombies just like bullets—lists of `[x, y]`.

---

### 5. **Collision Detection**
We compare bullet and zombie positions:

- If a bullet hits a zombie:
  - Remove both  
  - Increase score  
  - Update top score if necessary  

---

### 6. **Game Over & Restart**
If a zombie passes the bottom:

- Game shows “Game Over!”  
- Displays the final score  
- Player can restart by pressing **R**  

The `reset_game()` function clears bullets, zombies, score, and positions.

---



## Summary

Player Setup: This is where we load the player's image and place it at the bottom center of the screen. The player moves left and right using the arrow keys, and we track the player’s position using player_x and player_y.

Bullet Mechanics: When the player presses space, bullets are created at the player's current position. Each bullet moves upwards at a constant speed. We remove bullets when they go off-screen.

Zombie Mechanics: Zombies spawn at random positions at the top of the screen and slowly move downwards. If a zombie reaches the bottom, the game ends, and the player loses.

Collision and Scoring: Whenever a bullet hits a zombie, the zombie and bullet are removed from the game, and the score increases by 1. We also keep track of the top score in a file and display it on the screen.

Game Over and Restart: If the game is over, the player can press "R" to restart the game, resetting everything to its initial state.

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/yCKxrxnGAj0" 
    title="YouTube video player" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
  </iframe>
</div>


## Source code

{% include codeHeader.html %}
```python
import pygame
import random

## Initialize pygame
pygame.init()

## Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Zombie Shooter")

## Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

## Load assets
player_img = pygame.image.load("static/icon.png")
zombie_img = pygame.image.load("static/zombie.png")

## Resize images
player_size = 50
player_img = pygame.transform.scale(player_img, (player_size, player_size))
zombie_img = pygame.transform.scale(zombie_img, (player_size, player_size))

## Player setup
player_x = WIDTH // 2
player_y = HEIGHT - 70
player_speed = 5

## Bullet setup
bullets = []
bullet_speed = 7

## Zombie setup
zombies = []
zombie_speed = 2
spawn_rate = 25  # Lower is faster

## Font setup
font = pygame.font.Font(None, 36)

## Load top score from file
def load_top_score():
    try:
        with open("top_score.txt", "r") as file:
            return int(file.read().strip())
    except (FileNotFoundError, ValueError):
        return 0

def save_top_score(score):
    with open("top_score.txt", "w") as file:
        file.write(str(score))

top_score = load_top_score()

## Game loop
running = True
game_over = False
clock = pygame.time.Clock()
score = 0

def reset_game():
    global player_x, player_y, bullets, zombies, score, game_over
    player_x = WIDTH // 2
    player_y = HEIGHT - 70
    bullets = []
    zombies = []
    score = 0
    game_over = False

while running:
    screen.fill(BLACK)
    
    if not game_over:
        ## Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bullets.append([player_x + player_size // 2, player_y])
        
        ## Player movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and player_x > 0:
            player_x -= player_speed
        if keys[pygame.K_RIGHT] and player_x < WIDTH - player_size:
            player_x += player_speed
        
        ## Bullet movement
        for bullet in bullets[:]:
            bullet[1] -= bullet_speed
            if bullet[1] < 0:
                bullets.remove(bullet)
        
        ## Spawn zombies
        if random.randint(1, spawn_rate) == 1:
            zombies.append([random.randint(0, WIDTH - player_size), 0])
        
        ## Zombie movement
        for zombie in zombies[:]:
            zombie[1] += zombie_speed
            if zombie[1] > HEIGHT:
                game_over = True  # Game over if a zombie reaches bottom
        
        ## Collision detection
        for bullet in bullets[:]:
            for zombie in zombies[:]:
                if zombie[0] < bullet[0] < zombie[0] + player_size and \
                   zombie[1] < bullet[1] < zombie[1] + player_size:
                    zombies.remove(zombie)
                    bullets.remove(bullet)
                    score += 1
        
        ## Draw elements
        screen.blit(player_img, (player_x, player_y))
        for bullet in bullets:
            pygame.draw.rect(screen, RED, (bullet[0], bullet[1], 5, 10))
        for zombie in zombies:
            screen.blit(zombie_img, (zombie[0], zombie[1]))
        
        ## Update and display score
        if score > top_score:
            top_score = score
            save_top_score(top_score)
        
        top_score_text = font.render(f"Top Score: {top_score}", True, RED)
        screen.blit(top_score_text, (WIDTH - 200, 10))
        
        score_text = font.render(f"Score: {score}", True, GREEN)
        screen.blit(score_text, (WIDTH - 200, 40))
    
    else:
        ## Game over screen
        game_over_text = font.render("Game Over! Your score: " + str(score), True, WHITE)
        screen.blit(game_over_text, (WIDTH // 2 - 150, HEIGHT // 2 - 50))
        
        restart_text = font.render("Press R to Restart", True, WHITE)
        screen.blit(restart_text, (WIDTH // 2 - 100, HEIGHT // 2 + 10))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_game()
    
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
```


---

**Website:** https://www.pyshine.com
**Author:** PyShine

