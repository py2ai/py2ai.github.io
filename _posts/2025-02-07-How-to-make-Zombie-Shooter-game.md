---
layout: post
title: How to make a zombie shooter game 
mathjax: true
featured-img: 26072022-python-logo
description:  Display remote Matplotlib window in localhost
keywords: ["zombie shooter game", "pygame tutorial", "pygame basic tutorial"]
tags: ["zombie shooter game", "pygame tutorial", "pygame basic tutorial"]

---

Welcome to this tutorial where we’ll be building a simple "Zombie Shooter" game using Pygame! This game involves a player controlling a shooter at the bottom of the screen, shooting bullets upwards to eliminate incoming zombies. The goal is to survive as long as possible while accumulating points by shooting zombies. We'll break down the code into several sections to make it easy to understand:

Game Initialization: We start by initializing Pygame and setting up the game screen. The screen size is defined, along with the title and color scheme.

Player Setup: We load and resize the player's image and set the initial position and movement speed. The player can move left and right using the arrow keys.

Bullet Mechanism: Bullets are fired when the player presses the spacebar. We track the position and movement of the bullets, removing them when they go off-screen.

Zombie Setup: Zombies are spawned at random positions at the top of the screen and move downwards. If a zombie reaches the bottom, the game ends.

Collision Detection: The game checks for collisions between bullets and zombies. When a bullet hits a zombie, both are removed from the screen, and the player’s score increases.

Score and Game Over: The score is tracked and displayed on the screen. If the player’s score surpasses the top score, it is saved to a file. If the game ends, the player can restart by pressing "R".


# Key Sections of the Code:

Player Setup: This is where we load the player's image and place it at the bottom center of the screen. The player moves left and right using the arrow keys, and we track the player’s position using player_x and player_y.

Bullet Mechanics: When the player presses space, bullets are created at the player's current position. Each bullet moves upwards at a constant speed. We remove bullets when they go off-screen.

Zombie Mechanics: Zombies spawn at random positions at the top of the screen and slowly move downwards. If a zombie reaches the bottom, the game ends, and the player loses.

Collision and Scoring: Whenever a bullet hits a zombie, the zombie and bullet are removed from the game, and the score increases by 1. We also keep track of the top score in a file and display it on the screen.

Game Over and Restart: If the game is over, the player can press "R" to restart the game, resetting everything to its initial state.


{% include codeHeader.html %}
```python
import pygame
import random

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Zombie Shooter")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Load assets
player_img = pygame.image.load("static/icon.png")
zombie_img = pygame.image.load("static/zombie.png")

# Resize images
player_size = 50
player_img = pygame.transform.scale(player_img, (player_size, player_size))
zombie_img = pygame.transform.scale(zombie_img, (player_size, player_size))

# Player setup
player_x = WIDTH // 2
player_y = HEIGHT - 70
player_speed = 5

# Bullet setup
bullets = []
bullet_speed = 7

# Zombie setup
zombies = []
zombie_speed = 2
spawn_rate = 25  # Lower is faster

# Font setup
font = pygame.font.Font(None, 36)

# Load top score from file
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

# Game loop
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
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bullets.append([player_x + player_size // 2, player_y])
        
        # Player movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and player_x > 0:
            player_x -= player_speed
        if keys[pygame.K_RIGHT] and player_x < WIDTH - player_size:
            player_x += player_speed
        
        # Bullet movement
        for bullet in bullets[:]:
            bullet[1] -= bullet_speed
            if bullet[1] < 0:
                bullets.remove(bullet)
        
        # Spawn zombies
        if random.randint(1, spawn_rate) == 1:
            zombies.append([random.randint(0, WIDTH - player_size), 0])
        
        # Zombie movement
        for zombie in zombies[:]:
            zombie[1] += zombie_speed
            if zombie[1] > HEIGHT:
                game_over = True  # Game over if a zombie reaches bottom
        
        # Collision detection
        for bullet in bullets[:]:
            for zombie in zombies[:]:
                if zombie[0] < bullet[0] < zombie[0] + player_size and \
                   zombie[1] < bullet[1] < zombie[1] + player_size:
                    zombies.remove(zombie)
                    bullets.remove(bullet)
                    score += 1
        
        # Draw elements
        screen.blit(player_img, (player_x, player_y))
        for bullet in bullets:
            pygame.draw.rect(screen, RED, (bullet[0], bullet[1], 5, 10))
        for zombie in zombies:
            screen.blit(zombie_img, (zombie[0], zombie[1]))
        
        # Update and display score
        if score > top_score:
            top_score = score
            save_top_score(top_score)
        
        top_score_text = font.render(f"Top Score: {top_score}", True, RED)
        screen.blit(top_score_text, (WIDTH - 200, 10))
        
        score_text = font.render(f"Score: {score}", True, GREEN)
        screen.blit(score_text, (WIDTH - 200, 40))
    
    else:
        # Game over screen
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


