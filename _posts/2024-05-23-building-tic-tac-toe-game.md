---
layout: post
title:  Building a Tic-Tac-Toe Game in Python!
mathjax: true
featured-img: 26072022-python-logo
description:  Learn python with fun!
---

# Building a Tic-Tac-Toe Game in Python

In this tutorial, we'll create a simple Tic-Tac-Toe game in Python where a user can play against the computer. The computer will make random moves.

## Overview

Tic-Tac-Toe is a classic game played on a 3x3 grid. The objective is to place three of your marks in a horizontal, vertical, or diagonal row to win the game.

## Prerequisites

- Basic knowledge of Python programming
- Python installed on your machine

## Step-by-Step Guide

### Step 1: Create the Game Board

First, we'll create a function to print the game board. The board is a 3x3 grid represented as a list of lists.

{% include codeHeader.html %}
```python
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)
```
### Step 2: Check for a Win
Next, we need to check if a player has won. We'll create a function that checks all rows, columns, and diagonals.

{% include codeHeader.html %}
```python
def check_win(board, player):
    # Check rows
    for row in board:
        if all([spot == player for spot in row]):
            return True

    # Check columns
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True

    # Check diagonals
    if all([board[i][i] == player for i in range(3)]):
        return True
    if all([board[i][2 - i] == player for i in range(3)]):
        return True

    return False
```

### Step 3: Check for a Draw

We also need a function to check if the game is a draw (i.e., the board is full and there is no winner).

{% include codeHeader.html %}
```python
def check_draw(board):
    return all([spot in ['X', 'O'] for row in board for spot in row])
```

### Step 4: Get Empty Positions

To help the computer make a move, we'll create a function that returns a list of empty positions on the board.
{% include codeHeader.html %}
```python
def get_empty_positions(board):
    empty_positions = []
    for row in range(3):
        for col in range(3):
            if board[row][col] == " ":
                empty_positions.append((row, col))
    return empty_positions
```

### Step 5: Computer's Move

We'll create a function for the computer to make a random move from the available positions.
{% include codeHeader.html %}
```python
def computer_move(board):
    empty_positions = get_empty_positions(board)
    return random.choice(empty_positions)

```

### Step 6: Main Game Logic
Finally, we'll combine everything into the main function that handles the game flow, alternating turns between the user and the computer.
{% include codeHeader.html %}
```python
import random

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

def check_win(board, player):
    # Check rows
    for row in board:
        if all([spot == player for spot in row]):
            return True

    # Check columns
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True

    # Check diagonals
    if all([board[i][i] == player for i in range(3)]):
        return True
    if all([board[i][2 - i] == player for i in range(3)]):
        return True

    return False

def check_draw(board):
    return all([spot in ['X', 'O'] for row in board for spot in row])

def get_empty_positions(board):
    empty_positions = []
    for row in range(3):
        for col in range(3):
            if board[row][col] == " ":
                empty_positions.append((row, col))
    return empty_positions

def computer_move(board):
    empty_positions = get_empty_positions(board)
    return random.choice(empty_positions)

def tic_tac_toe():
    board = [[" " for _ in range(3)] for _ in range(3)]
    current_player = "X"  # User is 'X', computer is 'O'

    while True:
        print_board(board)
        if current_player == "X":
            row = int(input("Enter the row (0, 1, or 2): "))
            col = int(input("Enter the column (0, 1, or 2): "))
        else:
            row, col = computer_move(board)
            print(f"Computer chose: {row}, {col}")

        if board[row][col] != " ":
            print("Spot already taken. Try again.")
            continue

        board[row][col] = current_player

        if check_win(board, current_player):
            print_board(board)
            if current_player == "X":
                print("Player wins!")
            else:
                print("Computer wins!")
            break

        if check_draw(board):
            print_board(board)
            print("It's a draw!")
            break

        current_player = "O" if current_player == "X" else "X"

if __name__ == "__main__":
    tic_tac_toe()

```

Running the Game
Save the script to a file, for example, `tic_tac_toe.py`
Run the script using Python 3:

{% include codeHeader.html %}
```python
python tic_tac_toe.py
```



The user plays as "X" and the computer plays as "O". The game will alternate turns between the user and the computer, and will announce the result once there is a winner or a draw.

Example of playing game

```
python tic_tac_toe.py
  |   |  
-----
  |   |  
-----
  |   |  
-----
Enter the row (0, 1, or 2): 0
Enter the column (0, 1, or 2): 1
  | X |  
-----
  |   |  
-----
  |   |  
-----
Computer chose: 0, 0
O | X |  
-----
  |   |  
-----
  |   |  
-----
Enter the row (0, 1, or 2): 0
Enter the column (0, 1, or 2): 2
O | X | X
-----
  |   |  
-----
  |   |  
-----
Computer chose: 2, 2
O | X | X
-----
  |   |  
-----
  |   | O
-----
Enter the row (0, 1, or 2): 2
Enter the column (0, 1, or 2): 0
O | X | X
-----
  |   |  
-----
X |   | O
-----
Computer chose: 1, 2
O | X | X
-----
  |   | O
-----
X |   | O
-----
Enter the row (0, 1, or 2): 1
Enter the column (0, 1, or 2): 1
O | X | X
-----
  | X | O
-----
X |   | O
-----
Player wins!

(py38) python tic_tac_toe.py
  |   |  
-----
  |   |  
-----
  |   |  
-----
Enter the row (0, 1, or 2): 0
Enter the column (0, 1, or 2): 0
X |   |  
-----
  |   |  
-----
  |   |  
-----
Computer chose: 2, 2
X |   |  
-----
  |   |  
-----
  |   | O
-----
Enter the row (0, 1, or 2): 0
Enter the column (0, 1, or 2): 1
X | X |  
-----
  |   |  
-----
  |   | O
-----
Computer chose: 1, 2
X | X |  
-----
  |   | O
-----
  |   | O
-----
Enter the row (0, 1, or 2): 1
Enter the column (0, 1, or 2): 1
X | X |  
-----
  | X | O
-----
  |   | O
-----
Computer chose: 1, 0
X | X |  
-----
O | X | O
-----
  |   | O
-----
Enter the row (0, 1, or 2): 0
Enter the column (0, 1, or 2): 0
Spot already taken. Try again.
X | X |  
-----
O | X | O
-----
  |   | O
-----
Enter the row (0, 1, or 2): 2
Enter the column (0, 1, or 2): 0
X | X |  
-----
O | X | O
-----
X |   | O
-----
Computer chose: 0, 2
X | X | O
-----
O | X | O
-----
X |   | O
-----
Computer wins!

```

# Conclusion

Congratulations! You've successfully created a Tic-Tac-Toe game in Python. This game can be further enhanced by improving the computer's strategy or adding a graphical user interface. Happy coding!



