---
layout: page
title: GO Game
permalink: /games/go/
---

<div class="go-game-container">
  <h1>⚫⚪ GO Game</h1>
  
  <div class="game-description">
    <p><strong>Objective:</strong> Control more territory than your opponent by surrounding empty areas!</p>
    <p><strong>How to Play:</strong> 
       <br>🖱️ <strong>Mouse:</strong> Click on intersections to place stones
       <br>⌨️ <strong>Keyboard:</strong> Arrow keys to move cursor, Enter/Space to place stone
       <br>⚠️ <strong>Important:</strong> Once placed, stones CANNOT be moved! Use Undo if you make a mistake.
       <br>🤖 <strong>AI Mode:</strong> Play against computer (select "vs AI" in Game Mode)
    </p>
    <p><strong>Rules:</strong> 
       <br>• Black plays first (you are Black in AI mode)
       <br>• Stones with no liberties are captured
       <br>• Game ends when both players pass
       <br>• Final score = Captured stones + Territory
    </p>
    <p><strong>Controls:</strong> Pass (skip turn) | Undo (remove last move) | New Game (restart)</p>
    <p><strong>AI Difficulty:</strong> 
       <br>• Easy: Random moves
       <br>• Medium: Strategic with some randomness
       <br>• Hard: Best strategic moves
    </p>
  </div>
  
  <div class="game-wrapper">
    <div class="game-info">
      <div class="score-panel">
        <div class="score-item black-player">
          <div class="player-indicator black-stone"></div>
          <span class="score-label">Black Captures:</span>
          <span class="score-value" id="black-score">0</span>
        </div>
        <div class="score-item white-player">
          <div class="player-indicator white-stone"></div>
          <span class="score-label">White Captures:</span>
          <span class="score-value" id="white-score">0</span>
        </div>
        <div class="score-item">
          <span class="score-label">Turn:</span>
          <span class="score-value" id="turn-indicator">Black</span>
          <span class="ai-thinking" id="ai-thinking" style="display: none;">AI Thinking...</span>
        </div>
        <div class="score-item">
          <span class="score-label">Captures:</span>
          <span class="score-value" id="capture-indicator">B: 0 | W: 0</span>
        </div>
      </div>
      
      <div class="game-controls">
        <button id="pass-btn" class="game-btn">Pass</button>
        <button id="undo-btn" class="game-btn">Undo</button>
        <button id="restart-btn" class="game-btn">New Game</button>
        <button id="home-btn" class="game-btn">Back to Games</button>
      </div>
      
      <div class="board-size-selector">
        <label for="board-size">Board Size:</label>
        <select id="board-size">
          <option value="9">9×9</option>
          <option value="13">13×13</option>
          <option value="19" selected>19×19</option>
        </select>
      </div>
      
      <div class="game-mode-selector">
        <label for="game-mode">Game Mode:</label>
        <select id="game-mode">
          <option value="2player">2 Players</option>
          <option value="ai" selected>vs AI</option>
        </select>
      </div>
      
      <div class="ai-difficulty-selector" id="ai-difficulty-selector">
        <label for="ai-difficulty">AI Difficulty:</label>
        <select id="ai-difficulty">
          <option value="easy">Easy</option>
          <option value="medium" selected>Medium</option>
          <option value="hard">Hard</option>
        </select>
      </div>
    </div>
    
    <div id="game-over-screen" class="game-over-screen" style="display: none;">
      <h2 id="game-over-title">Game Over!</h2>
      <p id="game-over-message">Final Score: <strong id="final-score">0</strong></p>
      <button id="play-again-btn" class="restart-btn">Play Again</button>
    </div>
    
    <div class="board-container">
      <canvas id="go-board" width="600" height="600"></canvas>
    </div>
  </div>
</div>

<style>
.go-game-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
}

.go-game-container h1 {
  text-align: center;
  font-size: 2.5em;
  margin-bottom: 30px;
  color: #333;
}

.game-description {
  max-width: 600px;
  margin: 0 auto 30px;
  background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
  padding: 20px 25px;
  border-radius: 15px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
  color: white;
}

.game-description p {
  margin: 10px 0;
  font-size: 1.05em;
  line-height: 1.6;
}

.game-description strong {
  color: #f39c12;
}

.game-wrapper {
  max-width: 900px;
  margin: 0 auto 40px;
  position: relative;
}

.game-info {
  margin-bottom: 20px;
}

.score-panel {
  background: #1a1a2e;
  padding: 15px 20px;
  border-radius: 10px;
  border: 3px solid #2c3e50;
  color: white;
  margin-bottom: 15px;
}

.score-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 8px 0;
  font-size: 1.1em;
}

.player-indicator {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  margin-right: 10px;
}

.black-stone {
  background: radial-gradient(circle at 30% 30%, #555, #000);
  box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}

.white-stone {
  background: radial-gradient(circle at 30% 30%, #fff, #ccc);
  box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.score-label {
  font-weight: bold;
  color: #a0a0a0;
}

.score-value {
  font-weight: bold;
  color: #f39c12;
}

.game-controls {
  display: flex;
  justify-content: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 15px;
}

.game-btn {
  padding: 10px 20px;
  font-size: 1em;
  border: none;
  border-radius: 25px;
  cursor: pointer;
  font-weight: bold;
  transition: all 0.3s ease;
  background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
  color: white;
}

.game-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 5px 15px rgba(44, 62, 80, 0.4);
}

.board-size-selector {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
}

.board-size-selector label {
  font-weight: bold;
  color: #333;
}

.board-size-selector select {
  padding: 8px 15px;
  font-size: 1em;
  border: 2px solid #2c3e50;
  border-radius: 5px;
  background: white;
  cursor: pointer;
}

.game-mode-selector, .ai-difficulty-selector {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
}

.game-mode-selector label, .ai-difficulty-selector label {
  font-weight: bold;
  color: #333;
}

.game-mode-selector select, .ai-difficulty-selector select {
  padding: 8px 15px;
  font-size: 1em;
  border: 2px solid #2c3e50;
  border-radius: 5px;
  background: white;
  cursor: pointer;
}

.board-container {
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #d4a574 0%, #c49a6c 100%);
  padding: 30px;
  border-radius: 15px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

#go-board {
  display: block;
  background: #dcb35c;
  border-radius: 5px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
  cursor: pointer;
}

.game-over-screen {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: rgba(255, 255, 255, 0.95);
  padding: 40px;
  border-radius: 20px;
  text-align: center;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
  z-index: 100;
  min-width: 300px;
  border: 3px solid #2c3e50;
}

.game-over-screen h2 {
  font-size: 2em;
  margin-bottom: 20px;
  color: #333;
}

.restart-btn {
  padding: 12px 30px;
  font-size: 1.1em;
  border: none;
  border-radius: 25px;
  cursor: pointer;
  font-weight: bold;
  transition: all 0.3s ease;
  background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
  color: white;
}

.restart-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 5px 15px rgba(44, 62, 80, 0.4);
}

#final-score {
  font-size: 1.5em;
  color: #f39c12;
}

.ai-thinking {
  margin-left: 10px;
  font-size: 0.9em;
  color: #f39c12;
  font-weight: bold;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

@media (max-width: 768px) {
  .go-game-container h1 {
    font-size: 2em;
  }
  
  #go-board {
    width: 100%;
    height: auto;
  }
  
  .board-container {
    padding: 15px;
  }
  
  .game-controls {
    flex-direction: column;
    align-items: center;
  }
  
  .game-btn {
    width: 100%;
    max-width: 200px;
  }
}
</style>

<script>
class GoGame {
  constructor() {
    this.canvas = document.getElementById('go-board');
    this.ctx = this.canvas.getContext('2d');
    
    this.boardSize = 19;
    this.cellSize = this.canvas.width / (this.boardSize + 1);
    
    this.board = [];
    this.currentPlayer = 'black';
    this.moveHistory = [];
    this.blackCaptures = 0;
    this.whiteCaptures = 0;
    this.passCount = 0;
    this.gameOver = false;
    this.koPoint = null;
    
    this.cursorX = Math.floor(this.boardSize / 2);
    this.cursorY = Math.floor(this.boardSize / 2);
    this.showCursor = false;
    
    this.gameMode = 'ai';
    this.aiDifficulty = 'medium';
    this.aiThinking = false;
    
    this.init();
  }
  
  init() {
    this.resetBoard();
    this.bindEvents();
    
    const aiDifficultySelector = document.getElementById('ai-difficulty-selector');
    if (aiDifficultySelector) {
      aiDifficultySelector.style.display = this.gameMode === 'ai' ? 'flex' : 'none';
    }
    
    this.draw();
  }
  
  resetBoard() {
    this.board = [];
    for (let i = 0; i < this.boardSize; i++) {
      this.board[i] = [];
      for (let j = 0; j < this.boardSize; j++) {
        this.board[i][j] = null;
      }
    }
    
    this.currentPlayer = 'black';
    this.moveHistory = [];
    this.blackCaptures = 0;
    this.whiteCaptures = 0;
    this.passCount = 0;
    this.gameOver = false;
    this.koPoint = null;
    
    this.cursorX = Math.floor(this.boardSize / 2);
    this.cursorY = Math.floor(this.boardSize / 2);
    this.showCursor = false;
    
    this.updateDisplay();
  }
  
  bindEvents() {
    const passBtn = document.getElementById('pass-btn');
    const undoBtn = document.getElementById('undo-btn');
    const restartBtn = document.getElementById('restart-btn');
    const homeBtn = document.getElementById('home-btn');
    const boardSizeSelect = document.getElementById('board-size');
    const playAgainBtn = document.getElementById('play-again-btn');
    
    if (passBtn) {
      passBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.pass();
      });
    }
    
    if (undoBtn) {
      undoBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.undo();
      });
    }
    
    if (restartBtn) {
      restartBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.restart();
      });
    }
    
    if (homeBtn) {
      homeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        window.location.href = '/games';
      });
    }
    
    if (boardSizeSelect) {
      boardSizeSelect.addEventListener('change', (e) => {
        this.boardSize = parseInt(e.target.value);
        this.cellSize = this.canvas.width / (this.boardSize + 1);
        this.resetBoard();
        this.draw();
      });
    }
    
    const gameModeSelect = document.getElementById('game-mode');
    if (gameModeSelect) {
      gameModeSelect.addEventListener('change', (e) => {
        this.gameMode = e.target.value;
        const aiDifficultySelector = document.getElementById('ai-difficulty-selector');
        if (aiDifficultySelector) {
          aiDifficultySelector.style.display = this.gameMode === 'ai' ? 'flex' : 'none';
        }
        this.resetBoard();
        this.draw();
      });
    }
    
    const aiDifficultySelect = document.getElementById('ai-difficulty');
    if (aiDifficultySelect) {
      aiDifficultySelect.addEventListener('change', (e) => {
        this.aiDifficulty = e.target.value;
      });
    }
    
    if (playAgainBtn) {
      playAgainBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.restart();
      });
    }
    
    this.canvas.addEventListener('click', (e) => {
      this.handleClick(e);
    });
    
    this.canvas.addEventListener('mouseenter', () => {
      this.showCursor = false;
      this.draw();
    });
    
    this.canvas.addEventListener('mouseleave', () => {
      this.showCursor = false;
      this.draw();
    });
    
    this.canvas.addEventListener('mousemove', (e) => {
      this.handleMouseMove(e);
    });
    
    document.addEventListener('keydown', (e) => {
      this.handleKeyDown(e);
    });
  }
  
  handleClick(e) {
    if (this.gameOver) return;
    
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    const col = Math.round(x / this.cellSize) - 1;
    const row = Math.round(y / this.cellSize) - 1;
    
    if (col >= 0 && col < this.boardSize && row >= 0 && row < this.boardSize) {
      this.placeStone(col, row);
    }
  }
  
  handleMouseMove(e) {
    if (this.gameOver) return;
    
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    const col = Math.round(x / this.cellSize) - 1;
    const row = Math.round(y / this.cellSize) - 1;
    
    if (col >= 0 && col < this.boardSize && row >= 0 && row < this.boardSize) {
      if (this.cursorX !== col || this.cursorY !== row) {
        this.cursorX = col;
        this.cursorY = row;
        this.showCursor = true;
        this.draw();
      }
    }
  }
  
  handleKeyDown(e) {
    if (this.gameOver) return;
    
    const key = e.key.toLowerCase();
    
    if (key === 'arrowup' || key === 'w') {
      e.preventDefault();
      this.cursorY = Math.max(0, this.cursorY - 1);
      this.showCursor = true;
      this.draw();
    } else if (key === 'arrowdown' || key === 's') {
      e.preventDefault();
      this.cursorY = Math.min(this.boardSize - 1, this.cursorY + 1);
      this.showCursor = true;
      this.draw();
    } else if (key === 'arrowleft' || key === 'a') {
      e.preventDefault();
      this.cursorX = Math.max(0, this.cursorX - 1);
      this.showCursor = true;
      this.draw();
    } else if (key === 'arrowright' || key === 'd') {
      e.preventDefault();
      this.cursorX = Math.min(this.boardSize - 1, this.cursorX + 1);
      this.showCursor = true;
      this.draw();
    } else if (key === 'enter' || key === ' ') {
      e.preventDefault();
      this.placeStone(this.cursorX, this.cursorY);
    }
  }
  
  placeStone(col, row) {
    if (this.board[col][row] !== null) return;
    
    const opponent = this.currentPlayer === 'black' ? 'white' : 'black';
    
    this.board[col][row] = this.currentPlayer;
    
    const capturedStones = this.checkCaptures(col, row, opponent);
    
    if (this.isKo(col, row)) {
      this.board[col][row] = null;
      return;
    }
    
    this.removeCapturedStones(capturedStones);
    
    if (this.isSuicide(col, row)) {
      this.board[col][row] = null;
      for (const stone of capturedStones) {
        this.board[stone.col][stone.row] = opponent;
      }
      return;
    }
    
    if (this.currentPlayer === 'black') {
      this.blackCaptures += capturedStones.length;
    } else {
      this.whiteCaptures += capturedStones.length;
    }
    
    this.moveHistory.push({
      col: col,
      row: row,
      player: this.currentPlayer,
      captures: capturedStones,
      koPoint: this.koPoint
    });
    
    this.koPoint = null;
    
    if (capturedStones.length === 1) {
      const captured = capturedStones[0];
      const liberties = this.countLiberties(captured.col, captured.row, opponent);
      if (liberties === 1) {
        this.koPoint = { col: captured.col, row: captured.row };
      }
    }
    
    this.passCount = 0;
    this.currentPlayer = opponent;
    this.showCursor = false;
    this.updateDisplay();
    this.draw();
    
    if (this.gameMode === 'ai' && this.currentPlayer === 'white' && !this.gameOver) {
      this.aiThinking = true;
      setTimeout(() => this.makeAIMove(), 500);
    }
  }
  
  checkCaptures(col, row, opponent) {
    const captured = [];
    const neighbors = this.getNeighbors(col, row);
    
    for (const neighbor of neighbors) {
      if (this.board[neighbor.col][neighbor.row] === opponent) {
        const group = this.getGroup(neighbor.col, neighbor.row);
        if (this.countLiberties(group[0].col, group[0].row, opponent) === 0) {
          captured.push(...group);
        }
      }
    }
    
    return captured;
  }
  
  getGroup(col, row) {
    const color = this.board[col][row];
    const group = [];
    const visited = new Set();
    const stack = [{ col, row }];
    
    while (stack.length > 0) {
      const current = stack.pop();
      const key = `${current.col},${current.row}`;
      
      if (visited.has(key)) continue;
      if (this.board[current.col][current.row] !== color) continue;
      
      visited.add(key);
      group.push(current);
      
      const neighbors = this.getNeighbors(current.col, current.row);
      for (const neighbor of neighbors) {
        stack.push(neighbor);
      }
    }
    
    return group;
  }
  
  countLiberties(col, row, color) {
    const group = this.getGroup(col, row);
    const liberties = new Set();
    
    for (const stone of group) {
      const neighbors = this.getNeighbors(stone.col, stone.row);
      for (const neighbor of neighbors) {
        if (this.board[neighbor.col][neighbor.row] === null) {
          liberties.add(`${neighbor.col},${neighbor.row}`);
        }
      }
    }
    
    return liberties.size;
  }
  
  getNeighbors(col, row) {
    const neighbors = [];
    const directions = [
      { col: 0, row: -1 },
      { col: 0, row: 1 },
      { col: -1, row: 0 },
      { col: 1, row: 0 }
    ];
    
    for (const dir of directions) {
      const newCol = col + dir.col;
      const newRow = row + dir.row;
      
      if (newCol >= 0 && newCol < this.boardSize && newRow >= 0 && newRow < this.boardSize) {
        neighbors.push({ col: newCol, row: newRow });
      }
    }
    
    return neighbors;
  }
  
  isSuicide(col, row) {
    const player = this.currentPlayer;
    const opponent = player === 'black' ? 'white' : 'black';
    
    const tempBoard = [];
    for (let i = 0; i < this.boardSize; i++) {
      tempBoard[i] = [];
      for (let j = 0; j < this.boardSize; j++) {
        tempBoard[i][j] = this.board[i][j];
      }
    }
    
    tempBoard[col][row] = player;
    
    const liberties = this.countLibertiesOnBoard(tempBoard, col, row, player);
    
    return liberties === 0;
  }
  
  countLibertiesOnBoard(board, col, row, color) {
    const group = this.getGroupOnBoard(board, col, row);
    const liberties = new Set();
    
    for (const stone of group) {
      const neighbors = this.getNeighbors(stone.col, stone.row);
      for (const neighbor of neighbors) {
        if (board[neighbor.col][neighbor.row] === null) {
          liberties.add(`${neighbor.col},${neighbor.row}`);
        }
      }
    }
    
    return liberties.size;
  }
  
  getGroupOnBoard(board, col, row) {
    const color = board[col][row];
    const group = [];
    const visited = new Set();
    const stack = [{ col, row }];
    
    while (stack.length > 0) {
      const current = stack.pop();
      const key = `${current.col},${current.row}`;
      
      if (visited.has(key)) continue;
      if (board[current.col][current.row] !== color) continue;
      
      visited.add(key);
      group.push(current);
      
      const neighbors = this.getNeighbors(current.col, current.row);
      for (const neighbor of neighbors) {
        stack.push(neighbor);
      }
    }
    
    return group;
  }
  
  isKo(col, row) {
    if (!this.koPoint) return false;
    return col === this.koPoint.col && row === this.koPoint.row;
  }
  
  removeCapturedStones(stones) {
    for (const stone of stones) {
      this.board[stone.col][stone.row] = null;
    }
  }
  
  pass() {
    if (this.gameOver) return;
    
    this.passCount++;
    this.moveHistory.push({
      pass: true,
      player: this.currentPlayer
    });
    
    if (this.passCount >= 2) {
      this.endGame();
      return;
    }
    
    this.currentPlayer = this.currentPlayer === 'black' ? 'white' : 'black';
    this.updateDisplay();
  }
  
  undo() {
    if (this.moveHistory.length === 0 || this.gameOver) return;
    
    const lastMove = this.moveHistory.pop();
    
    if (lastMove.pass) {
      this.passCount = Math.max(0, this.passCount - 1);
      this.currentPlayer = lastMove.player;
    } else {
      this.board[lastMove.col][lastMove.row] = null;
      
      for (const captured of lastMove.captures) {
        this.board[captured.col][captured.row] = lastMove.player === 'black' ? 'white' : 'black';
      }
      
      if (lastMove.player === 'black') {
        this.blackCaptures -= lastMove.captures.length;
      } else {
        this.whiteCaptures -= lastMove.captures.length;
      }
      
      this.koPoint = lastMove.koPoint;
      this.currentPlayer = lastMove.player;
    }
    
    this.updateDisplay();
    this.draw();
  }
  
  restart() {
    this.resetBoard();
    this.draw();
    document.getElementById('game-over-screen').style.display = 'none';
  }
  
  endGame() {
    this.gameOver = true;
    
    const blackScore = this.calculateScore('black');
    const whiteScore = this.calculateScore('white');
    
    const gameOverScreen = document.getElementById('game-over-screen');
    const gameOverTitle = document.getElementById('game-over-title');
    const gameOverMessage = document.getElementById('game-over-message');
    const finalScore = document.getElementById('final-score');
    
    if (blackScore > whiteScore) {
      gameOverTitle.textContent = '⚫ Black Wins!';
      gameOverTitle.style.color = '#2c3e50';
    } else if (whiteScore > blackScore) {
      gameOverTitle.textContent = '⚪ White Wins!';
      gameOverTitle.style.color = '#7f8c8d';
    } else {
      gameOverTitle.textContent = '🤝 Draw!';
      gameOverTitle.style.color = '#f39c12';
    }
    
    gameOverMessage.innerHTML = `Black: ${blackScore} | White: ${whiteScore}`;
    gameOverScreen.style.display = 'block';
  }
  
  calculateScore(player) {
    let score = 0;
    const opponent = player === 'black' ? 'white' : 'black';
    const visited = new Set();
    
    for (let i = 0; i < this.boardSize; i++) {
      for (let j = 0; j < this.boardSize; j++) {
        if (this.board[i][j] === player) {
          score++;
        } else if (this.board[i][j] === null) {
          const key = `${i},${j}`;
          if (!visited.has(key)) {
            const territory = this.getTerritory(i, j, visited);
            if (territory.owner === player) {
              score += territory.size;
            }
          }
        }
      }
    }
    
    if (player === 'black') {
      score += this.blackCaptures;
    } else {
      score += this.whiteCaptures + 6.5;
    }
    
    return Math.floor(score);
  }
  
  getTerritory(col, row, visited) {
    const territory = [];
    const stack = [{ col, row }];
    let touchesBlack = false;
    let touchesWhite = false;
    
    while (stack.length > 0) {
      const current = stack.pop();
      const key = `${current.col},${current.row}`;
      
      if (visited.has(key)) continue;
      
      if (this.board[current.col][current.row] === 'black') {
        touchesBlack = true;
        continue;
      }
      
      if (this.board[current.col][current.row] === 'white') {
        touchesWhite = true;
        continue;
      }
      
      visited.add(key);
      territory.push(current);
      
      const neighbors = this.getNeighbors(current.col, current.row);
      for (const neighbor of neighbors) {
        stack.push(neighbor);
      }
    }
    
    let owner = null;
    if (touchesBlack && !touchesWhite) {
      owner = 'black';
    } else if (touchesWhite && !touchesBlack) {
      owner = 'white';
    }
    
    return { size: territory.length, owner };
  }
  
  updateDisplay() {
    const blackScore = this.calculateScore('black');
    const whiteScore = this.calculateScore('white');
    
    document.getElementById('black-score').textContent = this.gameOver ? blackScore : this.blackCaptures;
    document.getElementById('white-score').textContent = this.gameOver ? whiteScore : this.whiteCaptures;
    document.getElementById('turn-indicator').textContent = this.currentPlayer === 'black' ? 'Black' : 'White';
    document.getElementById('capture-indicator').textContent = `B: ${this.blackCaptures} | W: ${this.whiteCaptures}`;
  }
  
  draw() {
    this.ctx.fillStyle = '#dcb35c';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.drawGrid();
    this.drawStones();
    this.drawStarPoints();
    
    if (this.showCursor) {
      this.drawCursor();
    }
  }
  
  drawGrid() {
    this.ctx.strokeStyle = '#000';
    this.ctx.lineWidth = 1;
    
    for (let i = 0; i < this.boardSize; i++) {
      const pos = this.cellSize * (i + 1);
      
      this.ctx.beginPath();
      this.ctx.moveTo(pos, this.cellSize);
      this.ctx.lineTo(pos, this.canvas.height - this.cellSize);
      this.ctx.stroke();
      
      this.ctx.beginPath();
      this.ctx.moveTo(this.cellSize, pos);
      this.ctx.lineTo(this.canvas.width - this.cellSize, pos);
      this.ctx.stroke();
    }
  }
  
  drawStones() {
    for (let i = 0; i < this.boardSize; i++) {
      for (let j = 0; j < this.boardSize; j++) {
        if (this.board[i][j] !== null) {
          const x = this.cellSize * (i + 1);
          const y = this.cellSize * (j + 1);
          const radius = this.cellSize * 0.45;
          
          this.ctx.beginPath();
          this.ctx.arc(x, y, radius, 0, Math.PI * 2);
          
          if (this.board[i][j] === 'black') {
            const gradient = this.ctx.createRadialGradient(x - radius * 0.3, y - radius * 0.3, 0, x, y, radius);
            gradient.addColorStop(0, '#555');
            gradient.addColorStop(1, '#000');
            this.ctx.fillStyle = gradient;
          } else {
            const gradient = this.ctx.createRadialGradient(x - radius * 0.3, y - radius * 0.3, 0, x, y, radius);
            gradient.addColorStop(0, '#fff');
            gradient.addColorStop(1, '#ccc');
            this.ctx.fillStyle = gradient;
          }
          
          this.ctx.fill();
          
          this.ctx.strokeStyle = this.board[i][j] === 'black' ? '#000' : '#999';
          this.ctx.lineWidth = 1;
          this.ctx.stroke();
        }
      }
    }
  }
  
  drawStarPoints() {
    const starPoints = this.getStarPoints();
    
    this.ctx.fillStyle = '#000';
    
    for (const point of starPoints) {
      const x = this.cellSize * (point.col + 1);
      const y = this.cellSize * (point.row + 1);
      
      this.ctx.beginPath();
      this.ctx.arc(x, y, 3, 0, Math.PI * 2);
      this.ctx.fill();
    }
  }
  
  getStarPoints() {
    const points = [];
    
    if (this.boardSize === 9) {
      points.push({ col: 2, row: 2 }, { col: 6, row: 2 });
      points.push({ col: 4, row: 4 });
      points.push({ col: 2, row: 6 }, { col: 6, row: 6 });
    } else if (this.boardSize === 13) {
      points.push({ col: 3, row: 3 }, { col: 9, row: 3 });
      points.push({ col: 6, row: 6 });
      points.push({ col: 3, row: 9 }, { col: 9, row: 9 });
    } else if (this.boardSize === 19) {
      points.push({ col: 3, row: 3 }, { col: 9, row: 3 }, { col: 15, row: 3 });
      points.push({ col: 3, row: 9 }, { col: 9, row: 9 }, { col: 15, row: 9 });
      points.push({ col: 3, row: 15 }, { col: 9, row: 15 }, { col: 15, row: 15 });
    }
    
    return points;
  }
  
  makeAIMove() {
    if (this.gameOver || this.currentPlayer !== 'white') {
      this.aiThinking = false;
      document.getElementById('ai-thinking').style.display = 'none';
      return;
    }
    
    document.getElementById('ai-thinking').style.display = 'inline';
    
    const moves = this.getValidMoves('white');
    
    if (moves.length === 0) {
      this.pass();
      this.aiThinking = false;
      document.getElementById('ai-thinking').style.display = 'none';
      return;
    }
    
    let bestMove;
    
    switch (this.aiDifficulty) {
      case 'easy':
        bestMove = this.getRandomMove(moves);
        break;
      case 'medium':
        bestMove = this.getMediumMove(moves);
        break;
      case 'hard':
        bestMove = this.getHardMove(moves);
        break;
    }
    
    if (bestMove) {
      this.placeStone(bestMove.col, bestMove.row);
    }
    
    this.aiThinking = false;
    document.getElementById('ai-thinking').style.display = 'none';
  }
  
  wouldCreateKo(col, row, player) {
    const opponent = player === 'black' ? 'white' : 'black';
    
    const tempBoard = [];
    for (let i = 0; i < this.boardSize; i++) {
      tempBoard[i] = [];
      for (let j = 0; j < this.boardSize; j++) {
        tempBoard[i][j] = this.board[i][j];
      }
    }
    
    tempBoard[col][row] = player;
    
    const capturedStones = this.checkCapturesOnBoard(tempBoard, col, row, opponent);
    
    for (const stone of capturedStones) {
      tempBoard[stone.col][stone.row] = null;
    }
    
    const liberties = this.countLibertiesOnBoard(tempBoard, col, row, player);
    
    if (liberties === 0) {
      return false;
    }
    
    if (capturedStones.length === 1) {
      const captured = capturedStones[0];
      const capturedLiberties = this.countLibertiesOnBoard(tempBoard, captured.col, captured.row, opponent);
      if (capturedLiberties === 1) {
        return true;
      }
    }
    
    return false;
  }
  
  getValidMoves(player) {
    const moves = [];
    const opponent = player === 'black' ? 'white' : 'black';
    
    for (let col = 0; col < this.boardSize; col++) {
      for (let row = 0; row < this.boardSize; row++) {
        if (this.board[col][row] !== null) continue;
        
        const tempBoard = [];
        for (let i = 0; i < this.boardSize; i++) {
          tempBoard[i] = [];
          for (let j = 0; j < this.boardSize; j++) {
            tempBoard[i][j] = this.board[i][j];
          }
        }
        
        tempBoard[col][row] = player;
        
        const capturedStones = this.checkCapturesOnBoard(tempBoard, col, row, opponent);
        
        for (const stone of capturedStones) {
          tempBoard[stone.col][stone.row] = null;
        }
        
        const liberties = this.countLibertiesOnBoard(tempBoard, col, row, player);
        
        if (liberties > 0) {
          if (!this.wouldCreateKo(col, row, player)) {
            moves.push({ col, row, score: 0 });
          }
        }
      }
    }
    
    return moves;
  }
  
  getRandomMove(moves) {
    return moves[Math.floor(Math.random() * moves.length)];
  }
  
  getMediumMove(moves) {
    for (const move of moves) {
      move.score = this.evaluateMove(move.col, move.row, 'white');
    }
    
    moves.sort((a, b) => b.score - a.score);
    
    const topMoves = moves.slice(0, Math.min(5, moves.length));
    return topMoves[Math.floor(Math.random() * topMoves.length)];
  }
  
  getHardMove(moves) {
    for (const move of moves) {
      move.score = this.evaluateMove(move.col, move.row, 'white');
    }
    
    moves.sort((a, b) => b.score - a.score);
    
    return moves[0];
  }
  
  evaluateMove(col, row, player) {
    let score = 0;
    const opponent = player === 'black' ? 'white' : 'black';
    
    const neighbors = this.getNeighbors(col, row);
    for (const neighbor of neighbors) {
      if (this.board[neighbor.col][neighbor.row] === opponent) {
        score += 10;
        
        const liberties = this.countLiberties(neighbor.col, neighbor.row, opponent);
        if (liberties === 1) {
          score += 20;
        } else {
          if (liberties === 2) {
            score += 10;
          }
        }
      }
    }
    
    const center = Math.floor(this.boardSize / 2);
    const distFromCenter = Math.abs(col - center) + Math.abs(row - center);
    score -= distFromCenter * 0.5;
    
    const starPoints = this.getStarPoints();
    for (const point of starPoints) {
      if (col === point.col && row === point.row) {
        score += 5;
      }
    }
    
    return score;
  }
  
  checkCapturesOnBoard(board, col, row, opponent) {
    const captured = [];
    const neighbors = this.getNeighbors(col, row);
    
    for (const neighbor of neighbors) {
      if (board[neighbor.col][neighbor.row] === opponent) {
        const group = this.getGroupOnBoard(board, neighbor.col, neighbor.row);
        if (this.countLibertiesOnBoard(board, group[0].col, group[0].row, opponent) === 0) {
          captured.push(...group);
        }
      }
    }
    
    return captured;
  }
  
  drawCursor() {
    const x = this.cellSize * (this.cursorX + 1);
    const y = this.cellSize * (this.cursorY + 1);
    const radius = this.cellSize * 0.45;
    
    this.ctx.beginPath();
    this.ctx.arc(x, y, radius, 0, Math.PI * 2);
    
    if (this.currentPlayer === 'black') {
      const gradient = this.ctx.createRadialGradient(x - radius * 0.3, y - radius * 0.3, 0, x, y, radius);
      gradient.addColorStop(0, 'rgba(85, 85, 85, 0.5)');
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0.5)');
      this.ctx.fillStyle = gradient;
    } else {
      const gradient = this.ctx.createRadialGradient(x - radius * 0.3, y - radius * 0.3, 0, x, y, radius);
      gradient.addColorStop(0, 'rgba(255, 255, 255, 0.5)');
      gradient.addColorStop(1, 'rgba(204, 204, 204, 0.5)');
      this.ctx.fillStyle = gradient;
    }
    
    this.ctx.fill();
    
    this.ctx.strokeStyle = this.currentPlayer === 'black' ? 'rgba(0, 0, 0, 0.5)' : 'rgba(153, 153, 153, 0.5)';
    this.ctx.lineWidth = 2;
    this.ctx.stroke();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  try {
    const game = new GoGame();
    console.log('GO game initialized successfully');
  } catch (error) {
    console.error('Error initializing GO game:', error);
  }
});
</script>
