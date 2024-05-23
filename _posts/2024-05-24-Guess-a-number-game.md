---
layout: post
title: How to make a simple guess a number game
mathjax: true
featured-img: 26072022-python-logo
summary:  Guess a number
---

Let's create a simple game where the user and the computer can play "Guess the Number". The computer will randomly select a number within a given range, and the user will have to guess it. The computer will provide feedback whether the guess is too high, too low, or correct.

Here’s a Python script for this game:

guess.py

{% include codeHeader.html %}
```python
import random

def guess_the_number():
    print("Welcome to 'Guess the Number'!")
    print("I am thinking of a number between 1 and 100.")
    
    number_to_guess = random.randint(1, 100)
    attempts = 0
    
    while True:
        try:
            user_guess = int(input("Enter your guess: "))
            attempts += 1
            
            if user_guess < number_to_guess:
                print("Too low! Try again.")
            elif user_guess > number_to_guess:
                print("Too high! Try again.")
            else:
                print(f"Congratulations! You guessed the number in {attempts} attempts.")
                break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
if __name__ == "__main__":
    guess_the_number()
```

Explanation
Import the random module:

This module is used to generate random numbers.
Define the guess_the_number function:

This function contains the logic of the game.
Welcome Message:

Print a welcome message to introduce the game.
Generate a Random Number:

Use random.randint(1, 100) to generate a random number between 1 and 100.
Initialize Attempt Counter:

Initialize a counter to track the number of attempts.
Main Game Loop:

Use a while True loop to continuously prompt the user for guesses until they guess correctly.
Inside the loop:
Get the user's guess and convert it to an integer.
Increment the attempt counter.
Provide feedback if the guess is too low, too high, or correct.
Handle invalid input (non-numeric values) with a try-except block.
Exit Condition:

The loop exits when the user correctly guesses the number.
Check if the Script is Running Directly:

Use if `__name__ == "__main__"`: to ensure the game runs when the script is executed directly.
Running the Game
To play the game, simply run the script in a Python environment.
The user will be prompted to guess a number, and the computer will provide feedback until the user guesses correctly.

Here it looks like:

```
python guess.py 
Welcome to 'Guess the Number'!
I am thinking of a number between 1 and 100.
Enter your guess: 9
Too low! Try again.
Enter your guess: 20
Too low! Try again.
Enter your guess: 80
Too high! Try again.
Enter your guess: 70
Too high! Try again.
Enter your guess: 60
Too high! Try again.
Enter your guess: 50
Too high! Try again.
Enter your guess: 40
Too high! Try again.
Enter your guess: 30
Too high! Try again.
Enter your guess: 20
Too low! Try again.
Enter your guess: 23
Congratulations! You guessed the number in 10 attempts.
```
