---
layout: page
title: Chess Game
permalink: /games/chess/
---

<div class="chess-game-container">
  <h1>♟ Chess Game</h1>
  
  <div class="game-description">
    <p><strong>Objective:</strong> Checkmate your opponent's King to win!</p>
    <p><strong>How to Play:</strong> 
       <br>🖱️ <strong>Mouse:</strong> Click piece to select, click destination to move
       <br>⌨️ <strong>Keyboard:</strong> Arrow keys to move cursor, Enter to select/move
       <br>🤖 <strong>AI Mode:</strong> Play against computer (default)
    </p>
    <p><strong>Rules:</strong> 
       <br>• White moves first (you are White in AI mode)
       <br>• Capture opponent's pieces by landing on them
       <br>• Protect your King from check
       <br>• Checkmate wins the game
    </p>
    <p><strong>Controls:</strong> New Game (restart) | Undo (remove last move) | AI Difficulty</p>
  </div>
  
  <div class="game-wrapper">
    <div class="game-info">
      <div class="score-panel">
        <div class="score-item white-player">
          <div class="player-indicator white-piece">♔</div>
          <span class="score-label">White:</span>
          <span class="score-value" id="white-captures">0</span>
        </div>
        <div class="score-item black-player">
          <div class="player-indicator black-piece">♚</div>
          <span class="score-label">Black:</span>
          <span class="score-value" id="black-captures">0</span>
        </div>
        <div class="score-item">
          <span class="score-label">Turn:</span>
          <span class="score-value" id="turn-indicator">White</span>
          <span class="ai-thinking" id="ai-thinking" style="display: none;">AI Thinking...</span>
          <span class="check-indicator" id="check-indicator" style="display: none;">CHECK!</span>
        </div>
      </div>
      
      <div class="captured-pieces">
        <div class="captured-white" id="captured-white"></div>
        <div class="captured-black" id="captured-black"></div>
      </div>
      
      <div class="game-controls">
        <button id="undo-btn" class="game-btn">Undo</button>
        <button id="restart-btn" class="game-btn">New Game</button>
        <button id="home-btn" class="game-btn">Back to Games</button>
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
      <p id="game-over-message">Result: <strong id="game-result">Checkmate</strong></p>
      <button id="play-again-btn" class="restart-btn">Play Again</button>
    </div>
    
    <div class="board-container">
      <canvas id="chess-board" width="480" height="480" style="display: block;"></canvas>
    </div>
  </div>
</div>

<style>
.chess-game-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
}

.chess-game-container h1 {
  text-align: center;
  font-size: 2.5em;
  margin-bottom: 30px;
  color: #333;
}

.game-description {
  max-width: 600px;
  margin: 0 auto 30px;
  background: linear-gradient(135deg, #8b4513 0%, #a0522d 100%);
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
  color: #ffd700;
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
  border: 3px solid #8b4513;
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
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  margin-right: 10px;
}

.white-piece {
  background: linear-gradient(135deg, #f0d9b5 0%, #c9302c 100%);
  border-radius: 5px;
  color: #fff;
}

.black-piece {
  background: linear-gradient(135deg, #779556 0%, #4a4a4a 100%);
  border-radius: 5px;
  color: #fff;
}

.score-label {
  font-weight: bold;
  color: #a0a0a0;
}

.score-value {
  font-weight: bold;
  color: #ffd700;
}

.captured-pieces {
  background: #1a1a2e;
  padding: 15px 20px;
  border-radius: 10px;
  border: 3px solid #8b4513;
  color: white;
  margin-bottom: 15px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.captured-white, .captured-black {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  min-height: 40px;
  min-width: 150px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 5px;
  padding: 5px;
  font-size: 24px;
  align-items: center;
}

.captured-white {
  color: #fff;
}

.captured-black {
  color: #000;
  background: rgba(255, 255, 255, 0.3);
}

.captured-piece {
  display: inline-block;
  font-size: 28px;
  line-height: 1;
  text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
}

.check-indicator {
  margin-left: 10px;
  font-size: 0.9em;
  color: #ff4444;
  font-weight: bold;
  animation: pulse 1.5s ease-in-out infinite;
}

.ai-thinking {
  margin-left: 10px;
  font-size: 0.9em;
  color: #8b4513;
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
  background: linear-gradient(135deg, #8b4513 0%, #a0522d 100%);
  color: white;
}

.game-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 5px 15px rgba(139, 69, 19, 0.4);
}

.ai-difficulty-selector {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
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
  border: 2px solid #8b4513;
  border-radius: 5px;
  background: white;
  cursor: pointer;
}

.board-container {
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #deb887 0%, #f5deb3 100%);
  padding: 30px;
  border-radius: 15px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

#chess-board {
  display: block;
  background: #f0d9b5;
  border: 4px solid #8b4513;
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
  border: 3px solid #8b4513;
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
  background: linear-gradient(135deg, #8b4513 0%, #a0522d 100%);
  color: white;
}

.restart-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 5px 15px rgba(139, 69, 19, 0.4);
}

#game-result {
  font-size: 1.5em;
  color: #8b4513;
}

@media (max-width: 768px) {
  .chess-game-container h1 {
    font-size: 2em;
  }
  
  #chess-board {
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
class ChessGame {
  constructor() {
    this.canvas = document.getElementById('chess-board');
    if (!this.canvas) {
      console.error('Canvas element not found!');
      return;
    }
    
    this.ctx = this.canvas.getContext('2d');
    if (!this.ctx) {
      console.error('Canvas context not available!');
      return;
    }
    
    this.boardSize = 8;
    this.cellSize = 60;
    
    this.board = [];
    this.selectedPiece = null;
    this.currentPlayer = 'white';
    this.moveHistory = [];
    this.whiteCaptures = 0;
    this.blackCaptures = 0;
    this.gameOver = false;
    this.gameMode = 'ai';
    this.aiDifficulty = 'medium';
    this.aiThinking = false;
    this.castlingRights = {
      white: { kingSide: true, queenSide: true },
      black: { kingSide: true, queenSide: true }
    };
    this.enPassantTarget = null;
    
    console.log('ChessGame constructor called');
    console.log('Canvas:', this.canvas);
    console.log('Context:', this.ctx);
    
    this.init();
  }
  
  init() {
    console.log('Initializing game...');
    this.resetBoard();
    this.bindEvents();
    
    setTimeout(() => {
      console.log('Starting initial draw...');
      this.draw();
    }, 100);
  }
  
  resetBoard() {
    console.log('Resetting board...');
    
    this.board = [];
    for (let row = 0; row < 8; row++) {
      this.board[row] = [];
      for (let col = 0; col < 8; col++) {
        const piece = this.getInitialPiece(row, col);
        this.board[row][col] = piece;
        if (piece) {
          console.log(`Initial piece at (${row}, ${col}): ${piece.color} ${piece.type}`);
        }
      }
    }
    
    console.log('Board initialized with', this.board.length, 'rows');
    console.log('Total pieces:', this.board.flat().filter(p => p !== null).length);
    
    this.selectedPiece = null;
    this.currentPlayer = 'white';
    this.moveHistory = [];
    this.whiteCaptures = 0;
    this.blackCaptures = 0;
    this.gameOver = false;
    this.castlingRights = {
      white: { kingSide: true, queenSide: true },
      black: { kingSide: true, queenSide: true }
    };
    this.enPassantTarget = null;
    
    this.updateDisplay();
  }
  
  getInitialPiece(row, col) {
    const initialSetup = [
      ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
      ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
      ['', '', '', '', '', '', '', ''],
      ['', '', '', '', '', '', '', ''],
      ['', '', '', '', '', '', '', ''],
      ['', '', '', '', '', '', '', ''],
      ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
      ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    ];
    
    const piece = initialSetup[row][col];
    if (!piece) return null;
    
    const color = piece === piece.toUpperCase() ? 'white' : 'black';
    const type = piece.toUpperCase();
    
    return { color, type, row, col };
  }
  
  bindEvents() {
    const undoBtn = document.getElementById('undo-btn');
    const restartBtn = document.getElementById('restart-btn');
    const homeBtn = document.getElementById('home-btn');
    const aiDifficultySelect = document.getElementById('ai-difficulty');
    const playAgainBtn = document.getElementById('play-again-btn');
    
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
    
    if (aiDifficultySelect) {
      aiDifficultySelect.addEventListener('change', (e) => {
        this.aiDifficulty = e.target.value;
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
    
    if (playAgainBtn) {
      playAgainBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.restart();
      });
    }
    
    this.canvas.addEventListener('click', (e) => {
      this.handleClick(e);
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
    
    const col = Math.floor(x / this.cellSize);
    const row = Math.floor(y / this.cellSize);
    
    if (col < 0 || col >= 8 || row < 0 || row >= 8) return;
    
    const piece = this.board[row][col];
    
    if (this.selectedPiece) {
      if (this.selectedPiece.row === row && this.selectedPiece.col === col) {
        this.selectedPiece = null;
        this.draw();
        return;
      }
      
      if (this.isValidMove(this.selectedPiece, row, col)) {
        this.makeMove(this.selectedPiece, row, col);
      } else {
        this.selectedPiece = null;
        this.draw();
      }
    } else {
      if (piece && piece.color === this.currentPlayer) {
        this.selectedPiece = piece;
        this.draw();
      }
    }
  }
  
  handleKeyDown(e) {
    if (this.gameOver) return;
    
    const key = e.key.toLowerCase();
    
    if (key === 'escape') {
      this.selectedPiece = null;
      this.draw();
      return;
    }
    
    if (key === 'z' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      this.undo();
      return;
    }
  }
  
  isValidMove(piece, toRow, toCol) {
    const moves = this.getValidMoves(piece);
    return moves.some(move => move.row === toRow && move.col === toCol);
  }
  
  getValidMoves(piece) {
    const moves = [];
    const { color, type, row, col } = piece;
    
    if (type === 'P') {
      this.getPawnMoves(piece, moves);
    } else if (type === 'R') {
      this.getRookMoves(piece, moves);
    } else if (type === 'N') {
      this.getKnightMoves(piece, moves);
    } else if (type === 'B') {
      this.getBishopMoves(piece, moves);
    } else if (type === 'Q') {
      this.getQueenMoves(piece, moves);
    } else if (type === 'K') {
      this.getKingMoves(piece, moves);
    }
    
    const legalMoves = [];
    for (const move of moves) {
      const targetPiece = this.board[move.row][move.col];
      if (targetPiece && targetPiece.type === 'K') {
        continue;
      }
      
      if (!this.wouldLeaveKingInCheck(piece, move.row, move.col)) {
        legalMoves.push(move);
      }
    }
    
    return legalMoves;
  }
  
  getPseudoLegalMoves(piece) {
    const moves = [];
    const { type } = piece;
    
    if (type === 'P') {
      this.getPawnMoves(piece, moves);
    } else if (type === 'R') {
      this.getRookMoves(piece, moves);
    } else if (type === 'N') {
      this.getKnightMoves(piece, moves);
    } else if (type === 'B') {
      this.getBishopMoves(piece, moves);
    } else if (type === 'Q') {
      this.getRookMoves(piece, moves);
      this.getBishopMoves(piece, moves);
    } else if (type === 'K') {
      this.getKingMoves(piece, moves);
    }
    
    return moves;
  }
  
  wouldLeaveKingInCheck(piece, toRow, toCol) {
    const { row: fromRow, col: fromCol, color } = piece;
    
    const tempPiece = this.board[toRow][toCol];
    this.board[toRow][toCol] = piece;
    this.board[fromRow][fromCol] = null;
    
    let capturedPawn = null;
    if (piece.type === 'P' && this.enPassantTarget && 
        toCol === this.enPassantTarget.col && 
        toRow === this.enPassantTarget.row) {
      const capturedPawnRow = color === 'white' ? toRow + 1 : toRow - 1;
      capturedPawn = this.board[capturedPawnRow][toCol];
      this.board[capturedPawnRow][toCol] = null;
    }
    
    const inCheck = this.isInCheck(color);
    
    this.board[fromRow][fromCol] = piece;
    this.board[toRow][toCol] = tempPiece;
    
    if (capturedPawn) {
      const capturedPawnRow = color === 'white' ? toRow + 1 : toRow - 1;
      this.board[capturedPawnRow][toCol] = capturedPawn;
    }
    
    return inCheck;
  }
  
  getPawnMoves(piece, moves) {
    const { color, row, col } = piece;
    const direction = color === 'white' ? -1 : 1;
    const startRow = color === 'white' ? 6 : 1;
    
    const forwardRow = row + direction;
    if (forwardRow >= 0 && forwardRow < 8) {
      if (!this.board[forwardRow][col]) {
        moves.push({ row: forwardRow, col });
        
        if (row === startRow) {
          const doubleForwardRow = row + 2 * direction;
          if (doubleForwardRow >= 0 && doubleForwardRow < 8 && !this.board[doubleForwardRow][col]) {
            moves.push({ row: doubleForwardRow, col });
          }
        }
      }
    }
    
    const captureCols = [col - 1, col + 1];
    for (const captureCol of captureCols) {
      if (captureCol >= 0 && captureCol < 8) {
        const captureRow = row + direction;
        if (captureRow >= 0 && captureRow < 8) {
          const targetPiece = this.board[captureRow][captureCol];
          if (targetPiece && targetPiece.color !== color) {
            moves.push({ row: captureRow, col: captureCol });
          }
        }
      }
    }
    
    if (this.enPassantTarget && 
        row + direction === this.enPassantTarget.row && 
        Math.abs(col - this.enPassantTarget.col) === 1) {
      moves.push({ row: row + direction, col: this.enPassantTarget.col });
    }
  }
  
  getRookMoves(piece, moves) {
    const { row, col } = piece;
    const directions = [
      { dr: -1, dc: 0 },
      { dr: 1, dc: 0 },
      { dr: 0, dc: -1 },
      { dr: 0, dc: 1 }
    ];
    
    for (const dir of directions) {
      let newRow = row + dir.dr;
      let newCol = col + dir.dc;
      
      while (newRow >= 0 && newRow < 8 && newCol >= 0 && newCol < 8) {
        const targetPiece = this.board[newRow][newCol];
        
        if (!targetPiece) {
          moves.push({ row: newRow, col: newCol });
        } else if (targetPiece.color !== piece.color) {
          moves.push({ row: newRow, col: newCol });
          break;
        } else {
          break;
        }
        
        newRow += dir.dr;
        newCol += dir.dc;
      }
    }
  }
  
  getKnightMoves(piece, moves) {
    const { row, col } = piece;
    const offsets = [
      { dr: -2, dc: -1 }, { dr: -2, dc: 1 },
      { dr: -1, dc: -2 }, { dr: -1, dc: 2 },
      { dr: 1, dc: -2 }, { dr: 1, dc: 2 },
      { dr: 2, dc: -1 }, { dr: 2, dc: 1 }
    ];
    
    for (const offset of offsets) {
      const newRow = row + offset.dr;
      const newCol = col + offset.dc;
      
      if (newRow >= 0 && newRow < 8 && newCol >= 0 && newCol < 8) {
        const targetPiece = this.board[newRow][newCol];
        if (!targetPiece || targetPiece.color !== piece.color) {
          moves.push({ row: newRow, col: newCol });
        }
      }
    }
  }
  
  getBishopMoves(piece, moves) {
    const { row, col } = piece;
    const directions = [
      { dr: -1, dc: -1 }, { dr: -1, dc: 1 },
      { dr: 1, dc: -1 }, { dr: 1, dc: 1 }
    ];
    
    for (const dir of directions) {
      let newRow = row + dir.dr;
      let newCol = col + dir.dc;
      
      while (newRow >= 0 && newRow < 8 && newCol >= 0 && newCol < 8) {
        const targetPiece = this.board[newRow][newCol];
        
        if (!targetPiece) {
          moves.push({ row: newRow, col: newCol });
        } else if (targetPiece.color !== piece.color) {
          moves.push({ row: newRow, col: newCol });
          break;
        } else {
          break;
        }
        
        newRow += dir.dr;
        newCol += dir.dc;
      }
    }
  }
  
  getQueenMoves(piece, moves) {
    this.getRookMoves(piece, moves);
    this.getBishopMoves(piece, moves);
  }
  
  getKingMoves(piece, moves) {
    const { row, col, color } = piece;
    const offsets = [
      { dr: -1, dc: -1 }, { dr: -1, dc: 0 }, { dr: -1, dc: 1 },
      { dr: 0, dc: -1 }, { dr: 0, dc: 1 },
      { dr: 1, dc: -1 }, { dr: 1, dc: 0 }, { dr: 1, dc: 1 }
    ];
    
    for (const offset of offsets) {
      const newRow = row + offset.dr;
      const newCol = col + offset.dc;
      
      if (newRow >= 0 && newRow < 8 && newCol >= 0 && newCol < 8) {
        const targetPiece = this.board[newRow][newCol];
        if (!targetPiece || targetPiece.color !== piece.color) {
          moves.push({ row: newRow, col: newCol });
        }
      }
    }
    
    if (!this.isInCheck(color)) {
      if (this.castlingRights[color].kingSide && col === 4) {
        if (!this.board[row][5] && !this.board[row][6]) {
          const rook = this.board[row][7];
          if (rook && rook.type === 'R') {
            if (!this.isSquareAttacked(row, 5, color) && !this.isSquareAttacked(row, 6, color)) {
              moves.push({ row, col: 6 });
            }
          }
        }
      }
      
      if (this.castlingRights[color].queenSide && col === 4) {
        if (!this.board[row][1] && !this.board[row][2] && !this.board[row][3]) {
          const rook = this.board[row][0];
          if (rook && rook.type === 'R') {
            if (!this.isSquareAttacked(row, 2, color) && !this.isSquareAttacked(row, 3, color)) {
              moves.push({ row, col: 2 });
            }
          }
        }
      }
    }
  }
  
  isSquareAttacked(row, col, defendingColor) {
    const attackingColor = defendingColor === 'white' ? 'black' : 'white';
    
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const piece = this.board[r][c];
        if (piece && piece.color === attackingColor) {
          const moves = this.getPseudoLegalMoves(piece);
          if (moves.some(move => move.row === row && move.col === col)) {
            return true;
          }
        }
      }
    }
    
    return false;
  }
  
  makeMove(piece, toRow, toCol) {
    const { row: fromRow, col: fromCol } = piece;
    const capturedPiece = this.board[toRow][toCol];
    
    this.board[toRow][toCol] = piece;
    this.board[fromRow][fromCol] = null;
    
    piece.row = toRow;
    piece.col = toCol;
    
    let capturedPawn = null;
    const oldEnPassantTarget = this.enPassantTarget;
    let promoted = false;
    
    if (piece.type === 'P') {
      if ((piece.color === 'white' && toRow === 0) || (piece.color === 'black' && toRow === 7)) {
        piece.type = 'Q';
        promoted = true;
      }
      
      if (oldEnPassantTarget && 
          toCol === oldEnPassantTarget.col && 
          toRow === oldEnPassantTarget.row) {
        const capturedPawnRow = piece.color === 'white' ? toRow + 1 : toRow - 1;
        capturedPawn = this.board[capturedPawnRow][toCol];
        if (capturedPawn && capturedPawn.type === 'P') {
          this.board[capturedPawnRow][toCol] = null;
        }
      }
    }
    
    if (piece.type === 'K' && Math.abs(toCol - fromCol) === 2) {
      if (this.isInCheck(piece.color)) {
        this.board[fromRow][fromCol] = piece;
        this.board[toRow][toCol] = capturedPiece;
        piece.row = fromRow;
        piece.col = fromCol;
        return;
      }
      
      const rookCol = toCol > fromCol ? 7 : 0;
      const rook = this.board[fromRow][rookCol];
      
      if (!rook || rook.type !== 'R') {
        this.board[fromRow][fromCol] = piece;
        this.board[toRow][toCol] = capturedPiece;
        piece.row = fromRow;
        piece.col = fromCol;
        return;
      }
      
      const passThroughCol = (fromCol + toCol) / 2;
      const passThroughPiece = this.board[fromRow][passThroughCol];
      if (passThroughPiece) {
        this.board[fromRow][fromCol] = piece;
        this.board[toRow][toCol] = capturedPiece;
        piece.row = fromRow;
        piece.col = fromCol;
        return;
      }
      
      this.board[fromRow][rookCol] = null;
      this.board[fromRow][(fromCol + toCol) / 2] = rook;
      rook.row = fromRow;
      rook.col = (fromCol + toCol) / 2;
      this.castlingRights[piece.color].kingSide = false;
      this.castlingRights[piece.color].queenSide = false;
    }
    
    if (piece.type === 'P' && Math.abs(toRow - fromRow) === 2) {
      this.enPassantTarget = { row: (fromRow + toRow) / 2, col: fromCol };
    } else {
      this.enPassantTarget = null;
    }
    
    if (capturedPiece) {
      if (piece.color === 'white') {
        this.whiteCaptures++;
        this.addCapturedPiece('white', capturedPiece);
      } else {
        this.blackCaptures++;
        this.addCapturedPiece('black', capturedPiece);
      }
    }
    
    if (capturedPawn) {
      if (piece.color === 'white') {
        this.whiteCaptures++;
        this.addCapturedPiece('white', capturedPawn);
      } else {
        this.blackCaptures++;
        this.addCapturedPiece('black', capturedPawn);
      }
    }
    
    this.moveHistory.push({
      piece: piece,
      fromRow,
      fromCol,
      toRow,
      toCol,
      captured: capturedPiece,
      capturedPawn: capturedPawn,
      oldEnPassantTarget: oldEnPassantTarget,
      promoted: promoted
    });
    
    this.currentPlayer = this.currentPlayer === 'white' ? 'black' : 'white';
    this.selectedPiece = null;
    this.updateDisplay();
    this.draw();
    
    if (this.isCheckmate(this.currentPlayer)) {
      this.gameOver = true;
      this.endGame('checkmate');
      return;
    }
    
    if (this.hasNoLegalMoves(this.currentPlayer)) {
      this.gameOver = true;
      this.endGame('stalemate');
      return;
    }
    
    if (!this.gameOver && this.gameMode === 'ai' && this.currentPlayer === 'black') {
      this.aiThinking = true;
      document.getElementById('ai-thinking').style.display = 'inline';
      setTimeout(() => this.makeAIMove(), 500);
    }
  }
  
  undo() {
    if (this.moveHistory.length === 0 || this.gameOver) return;
    
    const undoOneMove = () => {
      if (this.moveHistory.length === 0) return false;
      
      const lastMove = this.moveHistory.pop();
      const { piece, fromRow, fromCol, toRow, toCol, captured, capturedPawn, oldEnPassantTarget, promoted } = lastMove;
      
      this.board[fromRow][fromCol] = piece;
      this.board[toRow][toCol] = captured;
      
      piece.row = fromRow;
      piece.col = fromCol;
      
      if (promoted) {
        piece.type = 'P';
      }
      
      if (capturedPawn) {
        const capturedPawnRow = piece.color === 'white' ? toRow + 1 : toRow - 1;
        this.board[capturedPawnRow][toCol] = capturedPawn;
      }
      
      this.enPassantTarget = oldEnPassantTarget;
      
      if (captured) {
        if (piece.color === 'white') {
          this.whiteCaptures--;
          this.removeCapturedPiece('white', captured);
        } else {
          this.blackCaptures--;
          this.removeCapturedPiece('black', captured);
        }
      }
      
      if (capturedPawn) {
        if (piece.color === 'white') {
          this.whiteCaptures--;
          this.removeCapturedPiece('white', capturedPawn);
        } else {
          this.blackCaptures--;
          this.removeCapturedPiece('black', capturedPawn);
        }
      }
      
      this.currentPlayer = this.currentPlayer === 'white' ? 'black' : 'white';
      return true;
    };
    
    if (this.gameMode === 'ai') {
      undoOneMove();
      undoOneMove();
    } else {
      undoOneMove();
    }
    
    this.selectedPiece = null;
    this.updateDisplay();
    this.draw();
  }
  
  addCapturedPiece(color, piece) {
    const container = document.getElementById(`captured-${color}`);
    const pieceSymbols = {
      'white': { 'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙' },
      'black': { 'K': '♚', 'Q': '♛', 'R': '♜', 'B': '♝', 'N': '♞', 'P': '♟' }
    };
    
    const pieceElement = document.createElement('span');
    pieceElement.textContent = pieceSymbols[piece.color][piece.type];
    pieceElement.className = 'captured-piece';
    pieceElement.dataset.type = piece.type;
    pieceElement.dataset.color = piece.color;
    container.appendChild(pieceElement);
  }
  
  removeCapturedPiece(color, piece) {
    const container = document.getElementById(`captured-${color}`);
    const pieces = container.querySelectorAll('.captured-piece');
    
    for (const pieceElement of pieces) {
      if (pieceElement.dataset.type === piece.type && pieceElement.dataset.color === piece.color) {
        container.removeChild(pieceElement);
        break;
      }
    }
  }
  
  restart() {
    this.moveHistory = [];
    this.whiteCaptures = 0;
    this.blackCaptures = 0;
    this.gameOver = false;
    this.currentPlayer = 'white';
    this.selectedPiece = null;
    this.aiThinking = false;
    
    document.getElementById('game-over-screen').style.display = 'none';
    document.getElementById('captured-white').innerHTML = '';
    document.getElementById('captured-black').innerHTML = '';
    
    const aiThinking = document.getElementById('ai-thinking');
    if (aiThinking) {
      aiThinking.style.display = 'none';
    }
    
    this.resetBoard();
    this.draw();
  }
  
  updateDisplay() {
    document.getElementById('white-captures').textContent = this.whiteCaptures;
    document.getElementById('black-captures').textContent = this.blackCaptures;
    document.getElementById('turn-indicator').textContent = this.currentPlayer === 'white' ? 'White' : 'Black';
    
    const inCheck = this.isInCheck(this.currentPlayer);
    document.getElementById('check-indicator').style.display = inCheck ? 'inline' : 'none';
  }
  
  isInCheck(color) {
    const king = this.findKing(color);
    if (!king) return false;
    
    const opponentColor = color === 'white' ? 'black' : 'white';
    
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const piece = this.board[row][col];
        if (piece && piece.color === opponentColor) {
          const moves = this.getPseudoLegalMoves(piece);
          if (moves.some(move => move.row === king.row && move.col === king.col)) {
            return true;
          }
        }
      }
    }
    
    return false;
  }
  
  findKing(color) {
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const piece = this.board[row][col];
        if (piece && piece.color === color && piece.type === 'K') {
          return piece;
        }
      }
    }
    return null;
  }
  
  isCheckmate(color) {
    if (!this.isInCheck(color)) return false;
    
    return this.hasNoLegalMoves(color);
  }
  
  hasNoLegalMoves(color) {
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const piece = this.board[row][col];
        if (piece && piece.color === color) {
          const moves = this.getValidMoves(piece);
          if (moves.length > 0) {
            return false;
          }
        }
      }
    }
    
    return true;
  }
  
  makeAIMove() {
    if (this.gameOver || this.currentPlayer !== 'black') {
      this.aiThinking = false;
      const aiThinking = document.getElementById('ai-thinking');
      if (aiThinking) {
        aiThinking.style.display = 'none';
      }
      return;
    }
    
    const pieces = this.getPlayerPieces('black');
    const allMoves = [];
    
    for (const piece of pieces) {
      const moves = this.getValidMoves(piece);
      for (const move of moves) {
        allMoves.push({ piece, toRow: move.row, toCol: move.col });
      }
    }
    
    if (allMoves.length === 0) {
      this.gameOver = true;
      if (this.isInCheck('black')) {
        this.endGame('checkmate');
      } else {
        this.endGame('stalemate');
      }
      return;
    }
    
    let bestMove;
    
    try {
      switch (this.aiDifficulty) {
        case 'easy':
          bestMove = this.getRandomMove(allMoves);
          break;
        case 'medium':
          bestMove = this.getMediumMove(allMoves);
          break;
        case 'hard':
          bestMove = this.getHardMove(allMoves);
          break;
      }
    } catch (error) {
      console.error('AI move error:', error);
      bestMove = this.getRandomMove(allMoves);
    }
    
    if (bestMove) {
      this.makeMove(bestMove.piece, bestMove.toRow, bestMove.toCol);
    }
    
    this.aiThinking = false;
    const aiThinking = document.getElementById('ai-thinking');
    if (aiThinking) {
      aiThinking.style.display = 'none';
    }
  }
  
  getPlayerPieces(color) {
    const pieces = [];
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const piece = this.board[row][col];
        if (piece && piece.color === color) {
          pieces.push(piece);
        }
      }
    }
    return pieces;
  }
  
  getRandomMove(moves) {
    return moves[Math.floor(Math.random() * moves.length)];
  }
  
  getMediumMove(moves) {
    for (const move of moves) {
      move.score = this.evaluateMove(move);
    }
    
    moves.sort((a, b) => b.score - a.score);
    
    const topMoves = moves.slice(0, Math.min(5, moves.length));
    return topMoves[Math.floor(Math.random() * topMoves.length)];
  }
  
  getHardMove(moves) {
    for (const move of moves) {
      move.score = this.evaluateMove(move);
    }
    
    moves.sort((a, b) => b.score - a.score);
    return moves[0];
  }
  
  evaluateMove(move) {
    let score = 0;
    const { piece, toRow, toCol } = move;
    const targetPiece = this.board[toRow][toCol];
    
    if (targetPiece) {
      const pieceValues = { 'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0 };
      score += pieceValues[targetPiece.type] * 10;
    }
    
    const centerRow = 3.5;
    const centerCol = 3.5;
    const distFromCenter = Math.abs(toRow - centerRow) + Math.abs(toCol - centerCol);
    score -= distFromCenter * 0.5;
    
    const opponentColor = piece.color === 'white' ? 'black' : 'white';
    const opponentKing = this.findKing(opponentColor);
    if (opponentKing) {
      const distToKing = Math.abs(toRow - opponentKing.row) + Math.abs(toCol - opponentKing.col);
      score -= distToKing * 0.3;
    }
    
    return score;
  }
  
  endGame(result) {
    this.gameOver = true;
    
    const aiThinking = document.getElementById('ai-thinking');
    if (aiThinking) {
      aiThinking.style.display = 'none';
    }
    
    const gameOverScreen = document.getElementById('game-over-screen');
    const gameOverTitle = document.getElementById('game-over-title');
    const gameOverMessage = document.getElementById('game-over-message');
    const gameResult = document.getElementById('game-result');
    
    if (result === 'checkmate') {
      const winner = this.currentPlayer === 'white' ? 'Black' : 'White';
      gameOverTitle.textContent = 'Checkmate!';
      gameOverMessage.innerHTML = `${winner} wins!`;
    } else {
      gameOverTitle.textContent = 'Stalemate!';
      gameOverMessage.innerHTML = 'Draw - No legal moves available';
    }
    
    gameOverScreen.style.display = 'block';
  }
  
  draw() {
    console.log('Drawing board...');
    console.log('Canvas dimensions:', this.canvas.width, 'x', this.canvas.height);
    console.log('Cell size:', this.cellSize);
    console.log('Board state:', this.board);
    
    this.ctx.fillStyle = '#f0d9b5';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.drawBoard();
    this.drawPieces();
    this.drawSelection();
    
    console.log('Draw complete');
  }
  
  drawBoard() {
    console.log('Drawing board squares...');
    
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const x = col * this.cellSize;
        const y = row * this.cellSize;
        
        if ((row + col) % 2 === 0) {
          this.ctx.fillStyle = '#f0d9b5';
        } else {
          this.ctx.fillStyle = '#b58863';
        }
        this.ctx.fillRect(x, y, this.cellSize, this.cellSize);
        
        this.ctx.strokeStyle = '#8b4513';
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(x, y, this.cellSize, this.cellSize);
      }
    }
    
    console.log('Board squares drawn');
  }
  
  drawPieces() {
    console.log('Drawing pieces...');
    let pieceCount = 0;
    
    const pieceSymbols = {
      'white': { 'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙' },
      'black': { 'K': '♚', 'Q': '♛', 'R': '♜', 'B': '♝', 'N': '♞', 'P': '♟' }
    };
    
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const piece = this.board[row][col];
        if (piece) {
          pieceCount++;
          const x = col * this.cellSize + this.cellSize / 2;
          const y = row * this.cellSize + this.cellSize / 2;
          
          const fontSize = this.cellSize * 0.8;
          this.ctx.font = `bold ${fontSize}px Arial, sans-serif`;
          this.ctx.textAlign = 'center';
          this.ctx.textBaseline = 'middle';
          
          const symbol = pieceSymbols[piece.color][piece.type];
          
          if (piece.color === 'white') {
            this.ctx.strokeStyle = '#000';
            this.ctx.lineWidth = 2;
            this.ctx.strokeText(symbol, x, y);
            this.ctx.fillStyle = '#fff';
            this.ctx.fillText(symbol, x, y);
          } else {
            this.ctx.fillStyle = '#000';
            this.ctx.fillText(symbol, x, y);
          }
          
          console.log(`Drawing ${piece.color} ${piece.type} at (${row}, ${col}): ${symbol}`);
        }
      }
    }
    
    console.log(`Drew ${pieceCount} pieces`);
  }
  
  drawSelection() {
    if (this.selectedPiece) {
      const x = this.selectedPiece.col * this.cellSize;
      const y = this.selectedPiece.row * this.cellSize;
      
      this.ctx.strokeStyle = '#ffeb3b';
      this.ctx.lineWidth = 4;
      this.ctx.strokeRect(x, y, this.cellSize, this.cellSize);
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  try {
    const game = new ChessGame();
    console.log('Chess game initialized successfully');
  } catch (error) {
    console.error('Error initializing Chess game:', error);
  }
});
</script>
