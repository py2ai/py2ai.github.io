---
layout: post
title: Building a Simple Chess Game in Python
mathjax: true
featured-img: 26072022-python-logo
summary:  make a chess game in python
---

In this tutorial, we'll create a basic chess game using Python. The game will have a simple text-based interface where players can input their moves. We'll explain the code step by step, focusing on how the ChessBoard class works and how the game logic is implemented.

Understanding the Code
Let's start by examining the provided code:

{% include codeHeader.html %}
```python
class ChessBoard:
    def __init__(self):
        self.board = self.create_board()
        self.turn = 'white'  # 'white' starts first

    def create_board(self):
        # Unicode symbols for the chess pieces
        board = [
            ['♜', '♞', '♝', '♛', '♚', '♝', '♞', '♜'],
            ['♟', '♟', '♟', '♟', '♟', '♟', '♟', '♟'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['♙', '♙', '♙', '♙', '♙', '♙', '♙', '♙'],
            ['♖', '♘', '♗', '♕', '♔', '♗', '♘', '♖']
        ]
        return board

    def print_board(self):
        # Print top label for Player 1 side
        print('      Player 1')
        print('  a b c d e f g h')

        # Print rows with side labels
        for i, row in enumerate(self.board):
            # Print row number and left side label
            print(f'{8 - i} ', end='')
            # Print row content
            for piece in row:
                print(piece, end=' ')
            # Print right side label
            print(f'{8 - i}')

        # Print bottom label for Player 2 side
        print('  a b c d e f g h')
        print('      Player 2\n')

    def move_piece(self, start_pos, end_pos):
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        piece = self.board[start_row][start_col]
        self.board[start_row][start_col] = '.'
        self.board[end_row][end_col] = piece

        # Toggle turn
        self.turn = 'black' if self.turn == 'white' else 'white'

def parse_position(pos):
    col, row = pos
    return 8 - int(row), ord(col) - ord('a')

def main():
    chess_board = ChessBoard()
    chess_board.print_board()

    while True:
        player = 'Player 1' if chess_board.turn == 'white' else 'Player 2'
        start_pos = input(f"{player}, enter the start position (e.g., 'e2'): ")
        end_pos = input(f"{player}, enter the end position (e.g., 'e4'): ")

        start_pos = parse_position(start_pos)
        end_pos = parse_position(end_pos)

        chess_board.move_piece(start_pos, end_pos)
        chess_board.print_board()

if __name__ == "__main__":
    main()
```

# Explanation

## ChessBoard Class

This class represents the chessboard and contains methods to initialize the board, print it, and move pieces.

`__init__(self)`: This method initializes a new chessboard. It creates the board using the create_board method and sets the initial turn to 'white'.

`create_board(self)`: This method creates the initial configuration of the chessboard using Unicode symbols to represent the pieces.

`print_board(self)`: This method prints the current state of the chessboard to the console. It labels the rows and columns and prints each piece accordingly.

`move_piece(self, start_pos, end_pos)`: This method moves a piece from start_pos to end_pos on the board. It updates the board state and toggles the turn to the next player.

## parse_position Function
This function converts the input positions (e.g., 'e2') into row and column indices understandable by the ChessBoard class.

## main Function
This function orchestrates the game. It creates an instance of ChessBoard, prints the initial board, and then enters a loop where players input their moves. After each move, it prints the updated board.

## Running the Game
To run the game, simply execute the script. Players will be prompted to input their moves in the format 'e2' to 'e4', for example. The game will continue indefinitely until terminated manually.

# Conclusion
In this tutorial, we built a simple text-based chess game in Python. We explored the code for the ChessBoard class and explained how it manages the board state and player turns. You can further extend this game by implementing more complex rules, adding special moves, or creating a graphical interface. Happy coding!

Example run:

```
python chess.py
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟ 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . . . . . 3
2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 1, enter the start position (e.g., 'e2'): d7
Player 1, enter the end position (e.g., 'e4'): d6
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ ♟ ♟ . ♟ ♟ ♟ ♟ 7
6 . . . ♟ . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . . . . . 3
2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 2, enter the start position (e.g., 'e2'): e2
Player 2, enter the end position (e.g., 'e4'): e3
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ ♟ ♟ . ♟ ♟ ♟ ♟ 7
6 . . . ♟ . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . . . ♙ . . . 3
2 ♙ ♙ ♙ ♙ . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 1, enter the start position (e.g., 'e2'): d6
Player 1, enter the end position (e.g., 'e4'): d5
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ ♟ ♟ . ♟ ♟ ♟ ♟ 7
6 . . . . . . . . 6
5 . . . ♟ . . . . 5
4 . . . . . . . . 4
3 . . . . ♙ . . . 3
2 ♙ ♙ ♙ ♙ . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 2, enter the start position (e.g., 'e2'): e3
Player 2, enter the end position (e.g., 'e4'): e4
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ ♟ ♟ . ♟ ♟ ♟ ♟ 7
6 . . . . . . . . 6
5 . . . ♟ . . . . 5
4 . . . . ♙ . . . 4
3 . . . . . . . . 3
2 ♙ ♙ ♙ ♙ . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 1, enter the start position (e.g., 'e2'): d5
Player 1, enter the end position (e.g., 'e4'): e4
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ ♟ ♟ . ♟ ♟ ♟ ♟ 7
6 . . . . . . . . 6
5 . . . . . . . . 5
4 . . . . ♟ . . . 4
3 . . . . . . . . 3
2 ♙ ♙ ♙ ♙ . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 2, enter the start position (e.g., 'e2'): f1
Player 2, enter the end position (e.g., 'e4'): b5
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ ♟ ♟ . ♟ ♟ ♟ ♟ 7
6 . . . . . . . . 6
5 . ♗ . . . . . . 5
4 . . . . ♟ . . . 4
3 . . . . . . . . 3
2 ♙ ♙ ♙ ♙ . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 1, enter the start position (e.g., 'e2'): c7
Player 1, enter the end position (e.g., 'e4'): c6
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ ♟ . . ♟ ♟ ♟ ♟ 7
6 . . ♟ . . . . . 6
5 . ♗ . . . . . . 5
4 . . . . ♟ . . . 4
3 . . . . . . . . 3
2 ♙ ♙ ♙ ♙ . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 2, enter the start position (e.g., 'e2'): b5
Player 2, enter the end position (e.g., 'e4'): c6
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ ♟ . . ♟ ♟ ♟ ♟ 7
6 . . ♗ . . . . . 6
5 . . . . . . . . 5
4 . . . . ♟ . . . 4
3 . . . . . . . . 3
2 ♙ ♙ ♙ ♙ . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 1, enter the start position (e.g., 'e2'): b7
Player 1, enter the end position (e.g., 'e4'): c6
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ . . . ♟ ♟ ♟ ♟ 7
6 . . ♟ . . . . . 6
5 . . . . . . . . 5
4 . . . . ♟ . . . 4
3 . . . . . . . . 3
2 ♙ ♙ ♙ ♙ . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 2, enter the start position (e.g., 'e2'): b1
Player 2, enter the end position (e.g., 'e4'): b1
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ . . . ♟ ♟ ♟ ♟ 7
6 . . ♟ . . . . . 6
5 . . . . . . . . 5
4 . . . . ♟ . . . 4
3 . . . . . . . . 3
2 ♙ ♙ ♙ ♙ . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 1, enter the start position (e.g., 'e2'): a8
Player 1, enter the end position (e.g., 'e4'): a8
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ . . . ♟ ♟ ♟ ♟ 7
6 . . ♟ . . . . . 6
5 . . . . . . . . 5
4 . . . . ♟ . . . 4
3 . . . . . . . . 3
2 ♙ ♙ ♙ ♙ . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 2, enter the start position (e.g., 'e2'): c2
Player 2, enter the end position (e.g., 'e4'): c3
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ 8
7 ♟ . . . ♟ ♟ ♟ ♟ 7
6 . . ♟ . . . . . 6
5 . . . . . . . . 5
4 . . . . ♟ . . . 4
3 . . ♙ . . . . . 3
2 ♙ ♙ . ♙ . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 1, enter the start position (e.g., 'e2'): d8
Player 1, enter the end position (e.g., 'e4'): d5
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ . ♚ ♝ ♞ ♜ 8
7 ♟ . . . ♟ ♟ ♟ ♟ 7
6 . . ♟ . . . . . 6
5 . . . ♛ . . . . 5
4 . . . . ♟ . . . 4
3 . . ♙ . . . . . 3
2 ♙ ♙ . ♙ . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 2, enter the start position (e.g., 'e2'): d2
Player 2, enter the end position (e.g., 'e4'): d3
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ . ♚ ♝ ♞ ♜ 8
7 ♟ . . . ♟ ♟ ♟ ♟ 7
6 . . ♟ . . . . . 6
5 . . . ♛ . . . . 5
4 . . . . ♟ . . . 4
3 . . ♙ ♙ . . . . 3
2 ♙ ♙ . . . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 1, enter the start position (e.g., 'e2'): e4
Player 1, enter the end position (e.g., 'e4'): d3
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ . ♚ ♝ ♞ ♜ 8
7 ♟ . . . ♟ ♟ ♟ ♟ 7
6 . . ♟ . . . . . 6
5 . . . ♛ . . . . 5
4 . . . . . . . . 4
3 . . ♙ ♟ . . . . 3
2 ♙ ♙ . . . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ ♕ ♔ . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 2, enter the start position (e.g., 'e2'): d1
Player 2, enter the end position (e.g., 'e4'): d3
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ . ♚ ♝ ♞ ♜ 8
7 ♟ . . . ♟ ♟ ♟ ♟ 7
6 . . ♟ . . . . . 6
5 . . . ♛ . . . . 5
4 . . . . . . . . 4
3 . . ♙ ♕ . . . . 3
2 ♙ ♙ . . . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ . ♔ . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 1, enter the start position (e.g., 'e2'): d5
Player 1, enter the end position (e.g., 'e4'): d3
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ . ♚ ♝ ♞ ♜ 8
7 ♟ . . . ♟ ♟ ♟ ♟ 7
6 . . ♟ . . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . ♙ ♛ . . . . 3
2 ♙ ♙ . . . ♙ ♙ ♙ 2
1 ♖ ♘ ♗ . ♔ . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 2, enter the start position (e.g., 'e2'): e1
Player 2, enter the end position (e.g., 'e4'): e2
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ . ♚ ♝ ♞ ♜ 8
7 ♟ . . . ♟ ♟ ♟ ♟ 7
6 . . ♟ . . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . ♙ ♛ . . . . 3
2 ♙ ♙ . . ♔ ♙ ♙ ♙ 2
1 ♖ ♘ ♗ . . . ♘ ♖ 1
  a b c d e f g h
      Player 2

Player 1, enter the start position (e.g., 'e2'): d3 
Player 1, enter the end position (e.g., 'e4'): e2
      Player 1
  a b c d e f g h
8 ♜ ♞ ♝ . ♚ ♝ ♞ ♜ 8
7 ♟ . . . ♟ ♟ ♟ ♟ 7
6 . . ♟ . . . . . 6
5 . . . . . . . . 5
4 . . . . . . . . 4
3 . . ♙ . . . . . 3
2 ♙ ♙ . . ♛ ♙ ♙ ♙ 2
1 ♖ ♘ ♗ . . . ♘ ♖ 1
  a b c d e f g h
      Player 2
```
Oops, Player1 wins, but you got the idea! Of course you can add more functionality to the above code. Stay Tune!










