---
layout: post
title: Let's build a simple "Rock, Paper, Scissors" game
mathjax: true
featured-img: 26072022-python-logo
description:  make a game in python
tags:
  - Python
  - Programming
  - Game Development
  - Beginner Projects
  - Coding Tutorial
keywords:
  - Rock Paper Scissors game
  - Python game tutorial
  - Random choice Python
  - Conditional statements Python
  - User input Python
  - Python beginner projects
---

Let's create a simple "Rock, Paper, Scissors" game where the user can play against the computer. This is a classic game that demonstrates user input, random choices by the computer, and conditional statements to determine the winner.

Here's the Python script for the "Rock, Paper, Scissors" game:

### game.py

{% include codeHeader.html %}
```python
import random

def rock_paper_scissors():
    print("Welcome to 'Rock, Paper, Scissors'!")
    print("Instructions: Enter 'rock', 'paper', or 'scissors' to play.")
    
    choices = ['rock', 'paper', 'scissors']
    
    while True:
        user_choice = input("Enter your choice (or 'quit' to exit): ").lower()
        
        if user_choice == 'quit':
            print("Thanks for playing! Goodbye.")
            break
        
        if user_choice not in choices:
            print("Invalid choice. Please choose 'rock', 'paper', or 'scissors'.")
            continue
        
        computer_choice = random.choice(choices)
        print(f"Computer chose: {computer_choice}")
        
        if user_choice == computer_choice:
            print("It's a tie!")
        elif (user_choice == 'rock' and computer_choice == 'scissors') or \
             (user_choice == 'scissors' and computer_choice == 'paper') or \
             (user_choice == 'paper' and computer_choice == 'rock'):
            print("You win!")
        else:
            print("You lose!")
        
        print()  # Print a blank line for better readability

if __name__ == "__main__":
    rock_paper_scissors()
```

# Explanation

### Import the random module:
This module is used to generate random choices for the computer.

### Define the rock_paper_scissors function:
This function contains the logic of the game.

### Welcome Message:
Print a welcome message and instructions for the user.

### Define Possible Choices:
Create a list choices containing the valid choices: 'rock', 'paper', and 'scissors'.

### Main Game Loop:
Use a while True loop to continuously prompt the user for their choice.

### Inside the loop:
Get the user's choice and convert it to lowercase for consistency.
If the user enters 'quit', exit the game.
If the user enters an invalid choice, prompt them again.
Generate the computer's choice using random.choice(choices).
Print the computer's choice.
Determine the winner using conditional statements.
Print the result (win, lose, or tie).

### Exit Condition:
The loop exits when the user types 'quit'.

### Check if the Script is Running Directly:

Use if `__name__ == "__main__"`: to ensure the game runs when the script is executed directly.

# Running the Game
To play the game, simply run the script in a Python environment.
The user will be prompted to enter their choice of 'rock', 'paper', or 'scissors'. The computer will then randomly select one of the three options, and the game will determine and display the result.
The game will continue until the user types 'quit' to exit.
This script demonstrates basic user interaction, randomization, and control flow in Python, making it an excellent tutorial example for beginners.


# Example:

```
python game.py
Welcome to 'Rock, Paper, Scissors'!
Instructions: Enter 'rock', 'paper', or 'scissors' to play.
Enter your choice (or 'quit' to exit): rock
Computer chose: rock
It's a tie!

Enter your choice (or 'quit' to exit): rock
Computer chose: rock
It's a tie!

Enter your choice (or 'quit' to exit): rock
Computer chose: rock
It's a tie!

Enter your choice (or 'quit' to exit): paper
Computer chose: scissors
You lose!

Enter your choice (or 'quit' to exit): scissor
Invalid choice. Please choose 'rock', 'paper', or 'scissors'.
Enter your choice (or 'quit' to exit): scissors
Computer chose: paper
You win!

Enter your choice (or 'quit' to exit): quit
Thanks for playing! Goodbye.

```
