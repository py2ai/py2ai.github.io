---
layout: page
title: Tetris
permalink: /games/tetris/
---

<div class="tetris-container">
  <h1>🧱 Tetris</h1>
  
  <div class="game-description">
    <p><strong>Objective:</strong> Rotate and move falling blocks to complete horizontal lines and score points!</p>
    <p><strong>How to Play:</strong> Use arrow keys (←→) to move, (↑) to rotate, (↓) to speed up. On mobile, use on-screen buttons.</p>
    <p><strong>Rules:</strong> Complete lines to clear them and earn points. Game ends when blocks reach the top!</p>
    <p><strong>Scoring:</strong> 100 points per line cleared. Bonus points for multiple lines at once!</p>
  </div>
  
  <div class="game-wrapper">
    <div class="game-area">
      <div class="game-main">
        <div class="score-display">
          <span class="score-left">Score: <strong id="current-score">0</strong></span>
          <button id="sound-toggle" class="sound-toggle" title="Toggle Sound">🔊</button>
          <span class="score-right">Top: <strong id="high-score">0</strong></span>
        </div>
        
        <canvas id="game-canvas" width="300" height="600"></canvas>
      </div>
      
      <div class="next-piece">
        <canvas id="next-canvas" width="100" height="100"></canvas>
      </div>
    </div>
    
    <div id="game-over-screen" class="game-over-screen" style="display: none;">
      <h2>Game Over!</h2>
      <p>Your Score: <strong id="final-score">0</strong></p>
      <button id="restart-btn" class="restart-btn">Play Again</button>
      <button id="home-btn" class="home-btn">Back to Games</button>
    </div>
    
    <div class="mobile-controls">
      <div class="control-header">
        <button id="lock-toggle" class="lock-toggle" title="Lock/Unlock Controls">🔓</button>
        <span id="lock-status" class="lock-status">Unlocked</span>
      </div>
      <div class="control-row">
        <button class="control-btn rotate-btn" data-action="rotate">↻</button>
      </div>
      <div class="control-row">
        <button class="control-btn left-btn" data-action="left">◀</button>
        <button class="control-btn down-btn" data-action="down">▼</button>
        <button class="control-btn right-btn" data-action="right">▶</button>
      </div>
    </div>
  </div>
</div>

<style>
.tetris-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
}

.tetris-container h1 {
  text-align: center;
  font-size: 2.5em;
  margin-bottom: 30px;
  color: #333;
}

.game-description {
  max-width: 600px;
  margin: 0 auto 30px;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  padding: 20px 25px;
  border-radius: 15px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.game-description p {
  margin: 10px 0;
  font-size: 1.05em;
  color: #333;
  line-height: 1.6;
}

.game-description strong {
  color: #667eea;
}

.game-wrapper {
  max-width: 500px;
  margin: 0 auto 40px;
  position: relative;
}

.score-display {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #1a1a1a;
  padding: 12px 20px;
  border-radius: 10px 10px 0 0;
  border: 4px solid #667eea;
  border-bottom: none;
  color: white;
  font-size: 1.1em;
  font-weight: bold;
  width: 308px;
  margin: 0 auto;
  max-width: 100%;
  box-sizing: border-box;
}

.score-left, .score-right {
  flex: 1;
}

.score-left {
  text-align: left;
}

.score-right {
  text-align: right;
}

.sound-toggle {
  background: none;
  border: none;
  font-size: 1.2em;
  cursor: pointer;
  padding: 0 10px;
  color: white;
  transition: transform 0.2s ease;
}

.sound-toggle:hover {
  transform: scale(1.2);
}

.sound-toggle:active {
  transform: scale(0.9);
}

.game-area {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-bottom: 20px;
}

.game-main {
  display: flex;
  flex-direction: column;
}

#game-canvas {
  display: block;
  background: #1a1a1a;
  border: 4px solid #667eea;
  border-top: none;
  border-radius: 0 0 10px 10px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.next-piece {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding-top: 50px;
}

#next-canvas {
  background: #1a1a1a;
  border: 2px solid #667eea;
  border-radius: 10px;
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
}

.game-over-screen h2 {
  font-size: 2em;
  margin-bottom: 20px;
  color: #333;
}

.restart-btn, .home-btn {
  padding: 12px 30px;
  font-size: 1.1em;
  border: none;
  border-radius: 25px;
  cursor: pointer;
  font-weight: bold;
  transition: all 0.3s ease;
  margin: 5px;
}

.restart-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.home-btn {
  background: #6c757d;
  color: white;
}

.restart-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.home-btn:hover {
  background: #5a6268;
}

#final-score {
  font-size: 1.5em;
  color: #667eea;
}

.mobile-controls {
  display: block;
  position: fixed;
  bottom: 20px;
  right: 20px;
  margin-top: 20px;
  padding: 20px;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
  border-radius: 20px;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  cursor: move;
  user-select: none;
  touch-action: none;
  z-index: 1000;
}

.mobile-controls.locked {
  cursor: default;
}

.control-header {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  margin-bottom: 10px;
}

.lock-toggle {
  background: rgba(255, 255, 255, 0.2);
  border: none;
  border-radius: 50%;
  width: 35px;
  height: 35px;
  font-size: 18px;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.lock-toggle:hover {
  background: rgba(255, 255, 255, 0.3);
  transform: scale(1.1);
}

.lock-toggle:active {
  transform: scale(0.95);
}

.lock-status {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.8);
  margin-top: 4px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.control-row {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-bottom: 10px;
}

.control-row:last-child {
  margin-bottom: 0;
}

.control-btn {
  width: 70px;
  height: 70px;
  font-size: 28px;
  border: none;
  border-radius: 15px;
  background: linear-gradient(145deg, #ffffff 0%, #f0f0f0 100%);
  color: #667eea;
  cursor: pointer;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2), inset 0 2px 0 rgba(255, 255, 255, 0.8);
  transition: all 0.15s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.control-btn:active {
  transform: scale(0.92);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.8);
}

.control-btn:hover {
  background: linear-gradient(145deg, #f8f8f8 0%, #e8e8e8 100%);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25), inset 0 2px 0 rgba(255, 255, 255, 0.8);
}

@media (max-width: 1024px) {
  .tetris-container h1 {
    font-size: 2em;
  }
  
  #game-canvas, #next-canvas {
    width: 100%;
    height: auto;
  }
  
  .score-display {
    width: 100%;
  }
  
  .game-description {
    padding: 15px 20px;
    font-size: 0.95em;
  }
  
  .game-description p {
    margin: 8px 0;
    font-size: 0.95em;
  }
}
</style>

<script>
class TetrisGame {
  constructor() {
    this.canvas = document.getElementById('game-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.nextCanvas = document.getElementById('next-canvas');
    this.nextCtx = this.nextCanvas.getContext('2d');
    
    this.cols = 10;
    this.rows = 20;
    this.blockSize = 30;
    
    this.board = [];
    this.currentPiece = null;
    this.nextPiece = null;
    
    this.score = 0;
    this.highScore = 0;
    this.gameLoop = null;
    this.dropInterval = 1000;
    this.lastDrop = 0;
    this.isGameRunning = false;
    this.soundEnabled = true;
    this.audioContext = null;
    
    this.colors = [
      '#00f0f0',
      '#0000f0',
      '#f0a000',
      '#f0f000',
      '#00f000',
      '#0000f0',
      '#a000f0',
      '#f0f0f0'
    ];
    
    this.shapes = [
      [[1, 1, 1, 1]],
      [[1, 0, 0], [1, 1, 1]],
      [[0, 0, 1], [1, 1, 1]],
      [[1, 1], [1, 1]],
      [[0, 1, 1], [1, 1, 0]],
      [[1, 1, 0], [0, 1, 1]],
      [[0, 1, 0], [1, 1, 1]]
    ];
    
    this.init();
  }
  
  init() {
    this.loadHighScore();
    this.bindEvents();
    this.startGame();
  }
  
  bindEvents() {
    const restartBtn = document.getElementById('restart-btn');
    const homeBtn = document.getElementById('home-btn');
    const soundToggle = document.getElementById('sound-toggle');
    
    if (!restartBtn || !homeBtn || !soundToggle) {
      console.error('Tetris game buttons not found in DOM');
      return;
    }
    
    restartBtn.addEventListener('click', (e) => {
      e.preventDefault();
      this.initAudioContext();
      this.restartGame();
    });
    
    homeBtn.addEventListener('click', (e) => {
      e.preventDefault();
      window.location.href = '/games';
    });
    
    soundToggle.addEventListener('click', (e) => {
      e.preventDefault();
      this.initAudioContext();
      this.toggleSound();
    });
    
    document.addEventListener('keydown', (e) => {
      this.initAudioContext();
      this.handleKeyPress(e);
    });
    
    document.querySelectorAll('.control-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        this.initAudioContext();
        const action = btn.dataset.action;
        this.handleMobileControl(action);
      });
      
      btn.addEventListener('touchstart', (e) => {
        e.preventDefault();
        this.initAudioContext();
        const action = btn.dataset.action;
        this.handleMobileControl(action);
      });
    });
    
    this.setupDraggableControls();
  }
  
  handleKeyPress(e) {
    if (!this.isGameRunning) return;
    
    switch (e.key) {
      case 'ArrowLeft':
      case 'a':
        this.movePiece(-1, 0);
        break;
      case 'ArrowRight':
      case 'd':
        this.movePiece(1, 0);
        break;
      case 'ArrowDown':
      case 's':
        this.movePiece(0, 1);
        break;
      case 'ArrowUp':
      case 'w':
        this.rotatePiece();
        break;
    }
    
    e.preventDefault();
  }
  
  handleMobileControl(action) {
    if (!this.isGameRunning) return;
    
    switch (action) {
      case 'left':
        this.movePiece(-1, 0);
        break;
      case 'right':
        this.movePiece(1, 0);
        break;
      case 'down':
        this.movePiece(0, 1);
        break;
      case 'rotate':
        this.rotatePiece();
        break;
    }
  }
  
  startGame() {
    this.resetGame();
    
    document.getElementById('game-over-screen').style.display = 'none';
    
    this.isGameRunning = true;
    this.lastDrop = Date.now();
    this.gameLoop = requestAnimationFrame(() => this.update());
  }
  
  resetGame() {
    this.board = Array(this.rows).fill(null).map(() => Array(this.cols).fill(0));
    this.score = 0;
    this.dropInterval = 1000;
    this.updateScoreDisplay();
    this.spawnPiece();
    this.draw();
  }
  
  spawnPiece() {
    if (!this.nextPiece) {
      this.nextPiece = this.createPiece();
    }
    
    this.currentPiece = this.nextPiece;
    this.nextPiece = this.createPiece();
    
    this.currentPiece.x = Math.floor(this.cols / 2) - Math.floor(this.currentPiece.shape[0].length / 2);
    this.currentPiece.y = 0;
    
    if (this.collision(0, 0)) {
      this.gameOver();
    }
    
    this.drawNextPiece();
  }
  
  createPiece() {
    const shapeIndex = Math.floor(Math.random() * this.shapes.length);
    const colorIndex = Math.floor(Math.random() * this.colors.length);
    
    return {
      shape: this.shapes[shapeIndex],
      color: this.colors[colorIndex],
      x: 0,
      y: 0
    };
  }
  
  movePiece(dx, dy) {
    if (!this.collision(dx, dy)) {
      this.currentPiece.x += dx;
      this.currentPiece.y += dy;
      this.draw();
    }
  }
  
  rotatePiece() {
    const rotated = this.currentPiece.shape[0].map((_, i) =>
      this.currentPiece.shape.map(row => row[i]).reverse()
    );
    
    const originalShape = this.currentPiece.shape;
    this.currentPiece.shape = rotated;
    
    if (this.collision(0, 0)) {
      this.currentPiece.shape = originalShape;
    } else {
      this.draw();
    }
  }
  
  collision(dx, dy) {
    for (let y = 0; y < this.currentPiece.shape.length; y++) {
      for (let x = 0; x < this.currentPiece.shape[y].length; x++) {
        if (this.currentPiece.shape[y][x]) {
          const newX = this.currentPiece.x + x + dx;
          const newY = this.currentPiece.y + y + dy;
          
          if (newX < 0 || newX >= this.cols || newY >= this.rows) {
            return true;
          }
          
          if (newY >= 0 && this.board[newY][newX]) {
            return true;
          }
        }
      }
    }
    return false;
  }
  
  lockPiece() {
    for (let y = 0; y < this.currentPiece.shape.length; y++) {
      for (let x = 0; x < this.currentPiece.shape[y].length; x++) {
        if (this.currentPiece.shape[y][x]) {
          const boardY = this.currentPiece.y + y;
          const boardX = this.currentPiece.x + x;
          
          if (boardY >= 0) {
            this.board[boardY][boardX] = this.currentPiece.color;
          }
        }
      }
    }
    
    this.clearLines();
    this.spawnPiece();
  }
  
  clearLines() {
    let linesCleared = 0;
    
    for (let y = this.rows - 1; y >= 0; y--) {
      if (this.board[y].every(cell => cell !== 0)) {
        this.board.splice(y, 1);
        this.board.unshift(Array(this.cols).fill(0));
        linesCleared++;
        y++;
      }
    }
    
    if (linesCleared > 0) {
      this.playClearSound();
      const points = [0, 100, 300, 500, 800];
      this.score += points[Math.min(linesCleared, 4)];
      this.updateScoreDisplay();
      this.increaseSpeed();
    }
  }
  
  increaseSpeed() {
    if (this.dropInterval > 200) {
      this.dropInterval -= 50;
    }
  }
  
  update() {
    if (!this.isGameRunning) return;
    
    const now = Date.now();
    if (now - this.lastDrop > this.dropInterval) {
      if (!this.collision(0, 1)) {
        this.currentPiece.y++;
      } else {
        this.playDropSound();
        this.lockPiece();
      }
      this.lastDrop = now;
    }
    
    this.draw();
    this.gameLoop = requestAnimationFrame(() => this.update());
  }
  
  draw() {
    this.ctx.fillStyle = '#1a1a1a';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.drawBoard();
    this.drawPiece();
  }
  
  drawBoard() {
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        if (this.board[y][x]) {
          this.drawBlock(x, y, this.board[y][x]);
        }
      }
    }
  }
  
  drawPiece() {
    for (let y = 0; y < this.currentPiece.shape.length; y++) {
      for (let x = 0; x < this.currentPiece.shape[y].length; x++) {
        if (this.currentPiece.shape[y][x]) {
          this.drawBlock(this.currentPiece.x + x, this.currentPiece.y + y, this.currentPiece.color);
        }
      }
    }
  }
  
  drawBlock(x, y, color) {
    this.ctx.fillStyle = color;
    this.ctx.fillRect(x * this.blockSize, y * this.blockSize, this.blockSize - 1, this.blockSize - 1);
    
    this.ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    this.ctx.fillRect(x * this.blockSize, y * this.blockSize, this.blockSize - 1, 4);
    this.ctx.fillRect(x * this.blockSize, y * this.blockSize, 4, this.blockSize - 1);
  }
  
  drawNextPiece() {
    this.nextCtx.fillStyle = '#1a1a1a';
    this.nextCtx.fillRect(0, 0, this.nextCanvas.width, this.nextCanvas.height);
    
    const offsetX = (this.nextCanvas.width - this.nextPiece.shape[0].length * 25) / 2;
    const offsetY = (this.nextCanvas.height - this.nextPiece.shape.length * 25) / 2;
    
    for (let y = 0; y < this.nextPiece.shape.length; y++) {
      for (let x = 0; x < this.nextPiece.shape[y].length; x++) {
        if (this.nextPiece.shape[y][x]) {
          this.nextCtx.fillStyle = this.nextPiece.color;
          this.nextCtx.fillRect(offsetX + x * 25, offsetY + y * 25, 24, 24);
          
          this.nextCtx.fillStyle = 'rgba(255, 255, 255, 0.3)';
          this.nextCtx.fillRect(offsetX + x * 25, offsetY + y * 25, 24, 3);
          this.nextCtx.fillRect(offsetX + x * 25, offsetY + y * 25, 3, 24);
        }
      }
    }
  }
  
  updateScoreDisplay() {
    document.getElementById('current-score').textContent = this.score;
    
    if (this.score > this.highScore) {
      this.highScore = this.score;
      document.getElementById('high-score').textContent = this.highScore;
      this.saveHighScore();
    }
  }
  
  gameOver() {
    this.isGameRunning = false;
    cancelAnimationFrame(this.gameLoop);
    
    document.getElementById('final-score').textContent = this.score;
    document.getElementById('game-over-screen').style.display = 'block';
  }
  
  restartGame() {
    document.getElementById('game-over-screen').style.display = 'none';
    this.resetGame();
    this.isGameRunning = true;
    this.lastDrop = Date.now();
    this.gameLoop = requestAnimationFrame(() => this.update());
  }
  
  saveHighScore() {
    localStorage.setItem('tetrisHighScore', this.highScore.toString());
  }
  
  loadHighScore() {
    const savedHighScore = localStorage.getItem('tetrisHighScore');
    if (savedHighScore) {
      this.highScore = parseInt(savedHighScore);
      document.getElementById('high-score').textContent = this.highScore;
    }
  }
  
  initAudioContext() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
  }
  
  toggleSound() {
    this.soundEnabled = !this.soundEnabled;
    const soundToggle = document.getElementById('sound-toggle');
    soundToggle.textContent = this.soundEnabled ? '🔊' : '🔇';
  }
  
  playDropSound() {
    if (!this.soundEnabled) return;
    
    try {
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      
      if (this.audioContext.state === 'suspended') {
        this.audioContext.resume();
      }
      
      const oscillator = this.audioContext.createOscillator();
      const gainNode = this.audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(this.audioContext.destination);
      
      oscillator.frequency.value = 200;
      oscillator.type = 'square';
      
      gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.1);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.1);
    } catch (error) {
      console.error('Error playing sound:', error);
    }
  }
  
  playClearSound() {
    if (!this.soundEnabled) return;
    
    try {
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      
      if (this.audioContext.state === 'suspended') {
        this.audioContext.resume();
      }
      
      const oscillator = this.audioContext.createOscillator();
      const gainNode = this.audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(this.audioContext.destination);
      
      oscillator.frequency.value = 1000;
      oscillator.type = 'square';
      
      gainNode.gain.setValueAtTime(0.15, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.2);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.2);
    } catch (error) {
      console.error('Error playing sound:', error);
    }
  }
  
  setupDraggableControls() {
    const controls = document.querySelector('.mobile-controls');
    const lockToggle = document.getElementById('lock-toggle');
    const lockStatus = document.getElementById('lock-status');
    if (!controls || !lockToggle || !lockStatus) return;
    
    let isLocked = false;
    let isDragging = false;
    let startX, startY, initialX, initialY;
    
    const updateLockState = () => {
      isLocked = !isLocked;
      controls.classList.toggle('locked', isLocked);
      lockToggle.textContent = isLocked ? '🔒' : '🔓';
      lockStatus.textContent = isLocked ? 'Locked' : 'Unlocked';
    };
    
    lockToggle.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      updateLockState();
    });
    
    lockToggle.addEventListener('touchstart', (e) => {
      e.preventDefault();
      e.stopPropagation();
      updateLockState();
    });
    
    const startDrag = (e) => {
      if (isLocked) return;
      if (e.target === lockToggle) return;
      
      isDragging = true;
      const clientX = e.type === 'mousedown' ? e.clientX : e.touches[0].clientX;
      const clientY = e.type === 'mousedown' ? e.clientY : e.touches[0].clientY;
      startX = clientX;
      startY = clientY;
      initialX = controls.offsetLeft;
      initialY = controls.offsetTop;
      controls.style.cursor = 'grabbing';
    };
    
    const drag = (e) => {
      if (!isDragging || isLocked) return;
      e.preventDefault();
      
      const clientX = e.type === 'mousemove' ? e.clientX : e.touches[0].clientX;
      const clientY = e.type === 'mousemove' ? e.clientY : e.touches[0].clientY;
      
      const deltaX = clientX - startX;
      const deltaY = clientY - startY;
      
      let newX = initialX + deltaX;
      let newY = initialY + deltaY;
      
      const maxX = window.innerWidth - controls.offsetWidth;
      const maxY = window.innerHeight - controls.offsetHeight;
      
      newX = Math.max(0, Math.min(newX, maxX));
      newY = Math.max(0, Math.min(newY, maxY));
      
      controls.style.left = newX + 'px';
      controls.style.top = newY + 'px';
      controls.style.right = 'auto';
      controls.style.bottom = 'auto';
    };
    
    const endDrag = () => {
      isDragging = false;
      if (!isLocked) {
        controls.style.cursor = 'move';
      }
    };
    
    controls.addEventListener('mousedown', startDrag);
    controls.addEventListener('touchstart', startDrag, { passive: false });
    
    document.addEventListener('mousemove', drag);
    document.addEventListener('touchmove', drag, { passive: false });
    
    document.addEventListener('mouseup', endDrag);
    document.addEventListener('touchend', endDrag);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  try {
    const game = new TetrisGame();
    console.log('Tetris game initialized successfully');
  } catch (error) {
    console.error('Error initializing Tetris game:', error);
  }
});
</script>
