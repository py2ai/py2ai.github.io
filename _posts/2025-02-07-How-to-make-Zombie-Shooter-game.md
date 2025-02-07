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
