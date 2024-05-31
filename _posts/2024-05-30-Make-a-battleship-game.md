---
layout: post
title: Let's build a simple "Battleship" game
mathjax: true
featured-img: 26072022-python-logo
summary:  make a battleship game in python
---

In this tutorial, we'll create a simple Battleship game that you can play via the command line. Battleship is a classic two-player game where players take turns guessing the locations of the opponent's ships on a grid.

# Step 1: Setting Up the Environment
First, ensure you have Python installed on your machine. You can download Python from python.org.

# Step 2: Defining the Game Board
We'll start by defining the game board and the basic structure of the game. Each player will have a 5x5 grid where they can place their ships and make guesses.

{% include codeHeader.html %}
```python
import random

def create_board(size):
    return [['~'] * size for _ in range(size)]

def print_board(board):
    for row in board:
        print(' '.join(row))
    print()

def place_ships(board, num_ships):
    ships = 0
    while ships < num_ships:
        x, y = random.randint(0, len(board) - 1), random.randint(0, len(board) - 1)
        if board[x][y] == '~':
            board[x][y] = 'S'
            ships += 1

def get_user_guess():
    while True:
        guess = input("Enter your guess (row and column, e.g., 2 3): ").split()
        if len(guess) == 2 and guess[0].isdigit() and guess[1].isdigit():
            return int(guess[0]), int(guess[1])
        else:
            print("Invalid input. Please enter two numbers separated by a space.")

```


# Step 3: Setting Up the Game
Next, we'll set up the game, including the boards for both players, and allow players to place their ships.

{% include codeHeader.html %}
```python
def setup_game(board_size=5, num_ships=3):
    player1_board = create_board(board_size)
    player2_board = create_board(board_size)
    
    print("Player 1, place your ships:")
    place_ships(player1_board, num_ships)
    print_board(player1_board)
    
    print("Player 2, place your ships:")
    place_ships(player2_board, num_ships)
    print_board(player2_board)
    
    return player1_board, player2_board
```

# Step 4: Main Game Loop
We'll create the main game loop where players take turns guessing the locations of the opponent's ships. We'll also keep track of the number of ships each player has left.

{% include codeHeader.html %}
```python
def play_game():
    board_size = 5
    num_ships = 3
    player1_board, player2_board = setup_game(board_size, num_ships)
    player1_guesses = create_board(board_size)
    player2_guesses = create_board(board_size)
    
    player1_ships = num_ships
    player2_ships = num_ships
    turn = 0
    
    while player1_ships > 0 and player2_ships > 0:
        if turn % 2 == 0:
            print("Player 1's turn")
            print_board(player1_guesses)
            guess = get_user_guess()
            if player2_board[guess[0]][guess[1]] == 'S':
                print("Hit!")
                player1_guesses[guess[0]][guess[1]] = 'X'
                player2_board[guess[0]][guess[1]] = 'X'
                player2_ships -= 1
            else:
                print("Miss.")
                player1_guesses[guess[0]][guess[1]] = 'O'
        else:
            print("Player 2's turn")
            print_board(player2_guesses)
            guess = get_user_guess()
            if player1_board[guess[0]][guess[1]] == 'S':
                print("Hit!")
                player2_guesses[guess[0]][guess[1]] = 'X'
                player1_board[guess[0]][guess[1]] = 'X'
                player1_ships -= 1
            else:
                print("Miss.")
                player2_guesses[guess[0]][guess[1]] = 'O'
        
        turn += 1
    
    if player1_ships == 0:
        print("Player 2 wins!")
    else:
        print("Player 1 wins!")

if __name__ == "__main__":
    play_game()
```

# Explanation:
## create_board Function: 
This function creates an empty game board of the specified size.
## print_board Function: 
This function prints the game board in a readable format.
## place_ships Function: 
This function randomly places a specified number of ships on the board.
## get_user_guess Function: 
This function prompts the player to enter their guess and validates the input.
## setup_game Function: 
This function sets up the game by creating boards for both players and placing their ships.
## play_game Function: 
This is the main game loop where players take turns guessing the locations of the opponent's ships. It keeps track of the number of ships each player has left and announces the winner when all of one player's ships are sunk.
## Running the Game
Save the code to a file named battleship.py and run it from your command line:

# Complete code

### battleship.py
{% include codeHeader.html %}
```python

import random

def create_board(size):
    return [['~'] * size for _ in range(size)]

def print_board(board):
    for row in board:
        print(' '.join(row))
    print()

def place_ships(board, num_ships):
    ships = 0
    while ships < num_ships:
        x, y = random.randint(0, len(board) - 1), random.randint(0, len(board) - 1)
        if board[x][y] == '~':
            board[x][y] = 'S'
            ships += 1

def get_user_guess():
    while True:
        guess = input("Enter your guess (row and column, e.g., 2 3): ").split()
        if len(guess) == 2 and guess[0].isdigit() and guess[1].isdigit():
            return int(guess[0]), int(guess[1])
        else:
            print("Invalid input. Please enter two numbers separated by a space.")


def setup_game(board_size=5, num_ships=3):
    player1_board = create_board(board_size)
    player2_board = create_board(board_size)
    
    print("Player 1, place your ships:")
    place_ships(player1_board, num_ships)
    print_board(player1_board)
    
    print("Player 2, place your ships:")
    place_ships(player2_board, num_ships)
    print_board(player2_board)
    
    return player1_board, player2_board

def play_game():
    board_size = 5
    num_ships = 3
    player1_board, player2_board = setup_game(board_size, num_ships)
    player1_guesses = create_board(board_size)
    player2_guesses = create_board(board_size)
    
    player1_ships = num_ships
    player2_ships = num_ships
    turn = 0
    
    while player1_ships > 0 and player2_ships > 0:
        if turn % 2 == 0:
            print("Player 1's turn")
            print_board(player1_guesses)
            guess = get_user_guess()
            if player2_board[guess[0]][guess[1]] == 'S':
                print("Hit!")
                player1_guesses[guess[0]][guess[1]] = 'X'
                player2_board[guess[0]][guess[1]] = 'X'
                player2_ships -= 1
            else:
                print("Miss.")
                player1_guesses[guess[0]][guess[1]] = 'O'
        else:
            print("Player 2's turn")
            print_board(player2_guesses)
            guess = get_user_guess()
            if player1_board[guess[0]][guess[1]] == 'S':
                print("Hit!")
                player2_guesses[guess[0]][guess[1]] = 'X'
                player1_board[guess[0]][guess[1]] = 'X'
                player1_ships -= 1
            else:
                print("Miss.")
                player2_guesses[guess[0]][guess[1]] = 'O'
        
        turn += 1
    
    if player1_ships == 0:
        print("Player 2 wins!")
    else:
        print("Player 1 wins!")

if __name__ == "__main__":
    play_game()

```

{% include codeHeader.html %}
```python
python battleship.py
```
Conclusion
You've now created a simple command-line Battleship game in Python. Players can place ships, take turns guessing, and the game will announce the winner when all ships are sunk. This basic implementation can be expanded with features like different board sizes, more ships, and improved user interfaces. Enjoy your game!

```
python battleship.py

Player 1, place your ships:
~ ~ ~ ~ ~
~ ~ ~ S S
~ S ~ ~ ~
~ ~ ~ ~ ~
~ ~ ~ ~ ~

Player 2, place your ships:
~ ~ ~ ~ ~
~ ~ ~ ~ ~
~ ~ S ~ ~
~ S ~ ~ ~
~ S ~ ~ ~

Player 1's turn
~ ~ ~ ~ ~
~ ~ ~ ~ ~
~ ~ ~ ~ ~
~ ~ ~ ~ ~
~ ~ ~ ~ ~

Enter your guess (row and column, e.g., 2 3): 2 2
Hit!
Player 2's turn
~ ~ ~ ~ ~
~ ~ ~ ~ ~
~ ~ ~ ~ ~
~ ~ ~ ~ ~
~ ~ ~ ~ ~

Enter your guess (row and column, e.g., 2 3): 1 1
Miss.
Player 1's turn
~ ~ ~ ~ ~
~ ~ ~ ~ ~
~ ~ X ~ ~
~ ~ ~ ~ ~
~ ~ ~ ~ ~

Enter your guess (row and column, e.g., 2 3): 3 1
Hit!
Player 2's turn
~ ~ ~ ~ ~
~ O ~ ~ ~
~ ~ ~ ~ ~
~ ~ ~ ~ ~
~ ~ ~ ~ ~

Enter your guess (row and column, e.g., 2 3): 2 3
Miss.
Player 1's turn
~ ~ ~ ~ ~
~ ~ ~ ~ ~
~ ~ X ~ ~
~ X ~ ~ ~
~ ~ ~ ~ ~

Enter your guess (row and column, e.g., 2 3): 4 1
Hit!
Player 1 wins!

```
