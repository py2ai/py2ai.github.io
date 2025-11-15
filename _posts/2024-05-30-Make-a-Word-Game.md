---
description: In this tutorial, we'll create a simple word game inspired by Scrabble that you can play via the command line. The game will allow two players to take turns ...
featured-img: 26072022-python-logo
keywords:
- Python
- Scrabble
- word game
- game development
- command line game
- PyDictionary
layout: post
mathjax: true
tags:
- Python
- Scrabble
- word game
- game development
- tutorial
title: Let's build a simple "word game inspired by Scrabble"
---


In this tutorial, we'll create a simple word game inspired by Scrabble that you can play via the command line. The game will allow two players to take turns forming words from a set of randomly chosen letters. The goal is to score the highest points by creating valid words.

# Step 1: Setting Up the Environment
First, ensure you have Python installed on your machine. You can download Python from python.org.

## Step 2: Importing Necessary Libraries
We will use the random library to generate random letters and string to access a list of alphabets.

{% include codeHeader.html %}
```python
import random
import string
```

## Step 3: Creating the Game Logic
Generate Random Letters: We'll start by generating a list of random letters for players to use.

{% include codeHeader.html %}
```python
def generate_letters(n=7):
    return random.choices(string.ascii_uppercase, k=n)
```

## Scoring System: 

Define a scoring system similar to Scrabble.

{% include codeHeader.html %}
```python
scrabble_scores = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
    'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
    'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
    'Y': 4, 'Z': 10
}

def calculate_score(word):
    return sum(scrabble_scores.get(letter, 0) for letter in word.upper())
```

## Checking Word Validity: 

For simplicity, we'll assume all user-entered words are valid English words. In a more robust implementation, you could use a dictionary API or a word list.

{% include codeHeader.html %}
```python
def is_valid_word(word, available_letters):
    available_letters = available_letters.copy()
    for letter in word:
        if letter.upper() in available_letters:
            available_letters.remove(letter.upper())
        else:
            return False
    return True
```

## Step 4: Main Game Loop

The game will run in a loop, allowing two players to take turns.

{% include codeHeader.html %}
```python
def play_game():
    player_scores = [0, 0]
    turn = 0
    
    while True:
        letters = generate_letters()
        print(f"Player {turn + 1}'s turn")
        print("Available letters:", ' '.join(letters))
        
        word = input("Enter a word (or 'q' to quit): ").upper()
        
        if word == 'Q':
            break
        
        if is_valid_word(word, letters):
            score = calculate_score(word)
            player_scores[turn] += score
            print(f"Valid word! Score: {score}")
        else:
            print("Invalid word or letters not in available set.")
        
        turn = 1 - turn  # Switch turn
        
        print(f"Scores: Player 1 - {player_scores[0]}, Player 2 - {player_scores[1]}")
        print()

    print("Game Over")
    print(f"Final Scores: Player 1 - {player_scores[0]}, Player 2 - {player_scores[1]}")

if __name__ == "__main__":
    play_game()
```

## Step 5: Running the Game
To play the game, save the above code to a file named word_game.py and run it from your command line:

{% include codeHeader.html %}
```python
python word_game.py
```


You've now created a simple command-line word game inspired by Scrabble. Players can generate letters, form words, and keep track of their scores. This basic implementation can be expanded with features like word validation using a dictionary API, more complex scoring systems, and enhanced user interfaces. Enjoy your game!
Lets see how it works:

```
word_game.py
Player 1's turn
Available letters: V K X N M G O
Enter a word (or 'q' to quit): GO
Valid word! Score: 3
Scores: Player 1 - 3, Player 2 - 0

Player 2's turn
Available letters: R U Z Q Q U C
Enter a word (or 'q' to quit): R
Valid word! Score: 1
Scores: Player 1 - 3, Player 2 - 1

Player 1's turn
Available letters: U M B V Q C S
Enter a word (or 'q' to quit): BUS
Valid word! Score: 5
Scores: Player 1 - 8, Player 2 - 1

Player 2's turn
Available letters: S C A F F Q J
Enter a word (or 'q' to quit): SCAFF
Valid word! Score: 13
Scores: Player 1 - 8, Player 2 - 14

Player 1's turn
Available letters: N Y H P D M A
Enter a word (or 'q' to quit): MAN
Valid word! Score: 5
Scores: Player 1 - 13, Player 2 - 14

Player 2's turn
Available letters: Y B V U N V G
Enter a word (or 'q' to quit): BUG
Valid word! Score: 6
Scores: Player 1 - 13, Player 2 - 20

Player 1's turn
Available letters: Q A Q D C O U
Enter a word (or 'q' to quit): DO
Valid word! Score: 3
Scores: Player 1 - 16, Player 2 - 20

Player 2's turn
Available letters: V P Q I B K M
Enter a word (or 'q' to quit): VIM
Valid word! Score: 8
Scores: Player 1 - 16, Player 2 - 28

Player 1's turn
Available letters: V X F K R R D
Enter a word (or 'q' to quit): q
Game Over
Final Scores: Player 1 - 16, Player 2 - 28

```

Emm, although Player 2 wins with 28 score but something is still missing. How about we add some kind of Dictionary so that the
word is checked for its correctness in English and validity as well. To enhance our command-line word game by verifying the correctness of user-entered words, we can use an English dictionary library. One such library is PyDictionary, which can check if a word exists in the dictionary.

First, you'll need to install the PyDictionary library. You can install it using pip:
{% include codeHeader.html %}
```python
pip install PyDictionary
```

Now, let's modify our game to include a word validation step using PyDictionary. Here's the complete code with the new functionality:

### word_game.py

{% include codeHeader.html %}
```python

import random
import string
from PyDictionary import PyDictionary

## Initialize the dictionary
dictionary = PyDictionary()

def generate_letters(n=7):
    return random.choices(string.ascii_uppercase, k=n)

scrabble_scores = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
    'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
    'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
    'Y': 4, 'Z': 10
}

def calculate_score(word):
    return sum(scrabble_scores.get(letter, 0) for letter in word.upper())

def is_valid_word(word, available_letters):
    available_letters = available_letters.copy()
    for letter in word:
        if letter.upper() in available_letters:
            available_letters.remove(letter.upper())
        else:
            return False
    return True

def word_exists(word):
    meaning = dictionary.meaning(word)
    return meaning is not None

def play_game():
    player_scores = [0, 0]
    turn = 0
    
    while True:
        letters = generate_letters()
        print(f"Player {turn + 1}'s turn")
        print("Available letters:", ' '.join(letters))
        
        while True:
            word = input("Enter a word (or 'q' to quit): ").upper()
            
            if word == 'Q':
                print("Game Over")
                print(f"Final Scores: Player 1 - {player_scores[0]}, Player 2 - {player_scores[1]}")
                return
            
            if is_valid_word(word, letters) and word_exists(word):
                score = calculate_score(word)
                player_scores[turn] += score
                print(f"Valid word! Score: {score}")
                break
            else:
                print("Invalid word or letters not in available set. Please try again.")
        
        turn = 1 - turn  # Switch turn
        
        print(f"Scores: Player 1 - {player_scores[0]}, Player 2 - {player_scores[1]}")
        print()

if __name__ == "__main__":
    play_game()
```

## Explanation:
PyDictionary Integration: We added the PyDictionary library to check if the entered word exists in the dictionary.
word_exists Function: This function uses PyDictionary to verify if a word has a valid meaning.
Validation Loop: Inside the main game loop, we now have a nested loop to repeatedly ask the player for a valid word if the entered word is invalid.
Running the Game
Ensure you have PyDictionary installed, then run the game script as before:

{% include codeHeader.html %}
```python
python word_game.py
```

Now, the game will prompt players to re-enter words if they are not valid English words, enhancing the overall gameplay experience. Enjoy your improved word game!

```
python word_game.py
Player 1's turn
Available letters: M Z V J W N P
Enter a word (or 'q' to quit): MJP
Error: The Following Error occured: list index out of range
Invalid word or letters not in available set. Please try again.
Enter a word (or 'q' to quit): VJ
Error: The Following Error occured: list index out of range
Invalid word or letters not in available set. Please try again.
Enter a word (or 'q' to quit): N
Valid word! Score: 1
Scores: Player 1 - 1, Player 2 - 0

Player 2's turn
Available letters: E I Y I F J B
Enter a word (or 'q' to quit): BI
Valid word! Score: 4
Scores: Player 1 - 1, Player 2 - 4

Player 1's turn
Available letters: M Y H V V G Y
Enter a word (or 'q' to quit): H
Valid word! Score: 4
Scores: Player 1 - 5, Player 2 - 4

Player 2's turn
Available letters: C O B W X B C
Enter a word (or 'q' to quit): COW
Valid word! Score: 8
Scores: Player 1 - 5, Player 2 - 12

Player 1's turn
Available letters: X X X Q T P R
Enter a word (or 'q' to quit): X
Valid word! Score: 8
Scores: Player 1 - 13, Player 2 - 12

Player 2's turn
Available letters: J N O Z Z T S
Enter a word (or 'q' to quit): NOT
Valid word! Score: 3
Scores: Player 1 - 13, Player 2 - 15

Player 1's turn
Available letters: S U T G A G H
Enter a word (or 'q' to quit): HUT
Valid word! Score: 6
Scores: Player 1 - 19, Player 2 - 15

Player 2's turn
Available letters: C T U S P I A
Enter a word (or 'q' to quit): CUPS
Error: The Following Error occured: list index out of range
Invalid word or letters not in available set. Please try again.
Enter a word (or 'q' to quit): CUP
Valid word! Score: 7
Scores: Player 1 - 19, Player 2 - 22

Player 1's turn
Available letters: Y Y D K Q R B
Enter a word (or 'q' to quit): K
Valid word! Score: 5
Scores: Player 1 - 24, Player 2 - 22

Player 2's turn
Available letters: V Q S I Z B H
Enter a word (or 'q' to quit): H
Valid word! Score: 4
Scores: Player 1 - 24, Player 2 - 26

Player 1's turn
Available letters: K F W S T W C
Enter a word (or 'q' to quit): F
Valid word! Score: 4
Scores: Player 1 - 28, Player 2 - 26

Player 2's turn
Available letters: U Y Y R U J X
Enter a word (or 'q' to quit): R
Valid word! Score: 1
Scores: Player 1 - 28, Player 2 - 27

Player 1's turn
Available letters: U S G D A G R
Enter a word (or 'q' to quit): DUG
Valid word! Score: 5
Scores: Player 1 - 33, Player 2 - 27

Player 2's turn
Available letters: U H Y N P O C
Enter a word (or 'q' to quit): NO
Valid word! Score: 2
Scores: Player 1 - 33, Player 2 - 29

Player 1's turn
Available letters: M R T F W V D
Enter a word (or 'q' to quit): q
Game Over
Final Scores: Player 1 - 33, Player 2 - 29
```
---
**Website:** https://www.pyshine.com
**Author:** PyShine








