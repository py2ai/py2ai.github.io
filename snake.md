---
layout: page
title: Snake Game
permalink: /games/snake/
---

<div class="snake-game-container">
  <h1>🐍 Snake Game</h1>
  
  <div class="game-description">
    <p><strong>Objective:</strong> Guide the snake to eat food and grow as long as possible without hitting yourself!</p>
    <p><strong>Rules:</strong> The snake wraps around walls - use this to your advantage!</p>
    <p><strong>Scoring:</strong> +10 points for each food eaten. The game speeds up as you score more!</p>
  </div>
  
  <div class="game-wrapper">
    <div class="game-info">
      <div class="score-display">
        <span>Score: <strong id="current-score">0</strong></span>
        <span>High Score: <strong id="high-score">0</strong></span>
      </div>
    </div>
    
    <div id="game-start-screen" class="start-screen">
      <h2>Enter Your Name</h2>
      <input type="text" id="player-name" placeholder="Your name" maxlength="20" />
      <button id="start-btn" class="start-btn">Start Game</button>
      <div class="instructions">
        <p class="instructions-title">How to Play:</p>
        <p class="desktop-instructions">🖥️ <strong>Desktop:</strong> Use Arrow Keys or WASD to move</p>
        <p class="mobile-instructions">📱 <strong>Mobile:</strong> Tap the buttons below or swipe on the game</p>
      </div>
    </div>
    
    <div id="game-over-screen" class="game-over-screen" style="display: none;">
      <h2>Game Over!</h2>
      <p>Your Score: <strong id="final-score">0</strong></p>
      <button id="restart-btn" class="restart-btn">Play Again</button>
      <button id="home-btn" class="home-btn">Back to Games</button>
    </div>
    
    <canvas id="game-canvas" width="400" height="400"></canvas>
    
    <div class="mobile-controls">
      <div class="control-row">
        <button class="control-btn up-btn" data-direction="up">▲</button>
      </div>
      <div class="control-row">
        <button class="control-btn left-btn" data-direction="left">◀</button>
        <button class="control-btn down-btn" data-direction="down">▼</button>
        <button class="control-btn right-btn" data-direction="right">▶</button>
      </div>
    </div>
  </div>
  
  <div class="leaderboard">
    <h2>🏆 Top 20 Leaderboard</h2>
    <div id="leaderboard-list" class="leaderboard-list">
      <p class="no-scores">No scores yet. Be the first!</p>
    </div>
  </div>
</div>

<style>
.snake-game-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
}

.snake-game-container h1 {
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

.game-info {
  margin-bottom: 15px;
}

.score-display {
  display: flex;
  justify-content: space-between;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 15px 25px;
  border-radius: 10px;
  color: white;
  font-size: 1.2em;
  font-weight: bold;
}

#game-canvas {
  display: block;
  margin: 0 auto;
  background: #1a1a1a;
  border: 4px solid #667eea;
  border-radius: 10px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.start-screen, .game-over-screen {
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

.start-screen h2, .game-over-screen h2 {
  font-size: 2em;
  margin-bottom: 20px;
  color: #333;
}

#player-name {
  width: 100%;
  padding: 15px;
  font-size: 1.1em;
  border: 2px solid #ddd;
  border-radius: 10px;
  margin-bottom: 20px;
  box-sizing: border-box;
}

#player-name:focus {
  outline: none;
  border-color: #667eea;
}

.start-btn, .restart-btn, .home-btn {
  padding: 12px 30px;
  font-size: 1.1em;
  border: none;
  border-radius: 25px;
  cursor: pointer;
  font-weight: bold;
  transition: all 0.3s ease;
  margin: 5px;
}

.start-btn, .restart-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.home-btn {
  background: #6c757d;
  color: white;
}

.start-btn:hover, .restart-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.home-btn:hover {
  background: #5a6268;
}

.instructions {
  margin-top: 20px;
  text-align: left;
}

.instructions-title {
  font-size: 1.1em;
  font-weight: bold;
  color: #333;
  margin-bottom: 10px;
}

.desktop-instructions,
.mobile-instructions {
  font-size: 0.95em;
  color: #666;
  margin: 8px 0;
  line-height: 1.5;
}

.desktop-instructions strong,
.mobile-instructions strong {
  color: #667eea;
}

#final-score {
  font-size: 1.5em;
  color: #667eea;
}

.leaderboard {
  max-width: 600px;
  margin: 0 auto;
  background: white;
  border-radius: 20px;
  padding: 30px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

.leaderboard h2 {
  text-align: center;
  font-size: 2em;
  margin-bottom: 25px;
  color: #333;
}

.leaderboard-list {
  max-height: 600px;
  overflow-y: auto;
}

.leaderboard-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  margin-bottom: 10px;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 10px;
  transition: transform 0.2s ease;
}

.leaderboard-item:hover {
  transform: translateX(5px);
}

.leaderboard-item:nth-child(1) {
  background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
}

.leaderboard-item:nth-child(2) {
  background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
}

.leaderboard-item:nth-child(3) {
  background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
}

.rank {
  font-size: 1.5em;
  font-weight: bold;
  width: 40px;
  text-align: center;
}

.player-name {
  flex: 1;
  font-weight: bold;
  color: #333;
  margin-left: 15px;
}

.player-score {
  font-size: 1.3em;
  font-weight: bold;
  color: #667eea;
}

.no-scores {
  text-align: center;
  color: #999;
  font-size: 1.1em;
  padding: 20px;
}

.mobile-controls {
  display: none;
  margin-top: 20px;
  padding: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 15px;
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
  width: 60px;
  height: 60px;
  font-size: 24px;
  border: none;
  border-radius: 10px;
  background: white;
  color: #667eea;
  cursor: pointer;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.control-btn:active {
  transform: scale(0.95);
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}

.control-btn:hover {
  background: #f0f0f0;
}

@media (max-width: 600px) {
  .snake-game-container h1 {
    font-size: 2em;
  }
  
  #game-canvas {
    width: 100%;
    height: auto;
  }
  
  .mobile-controls {
    display: block;
  }
  
  .control-btn {
    width: 70px;
    height: 70px;
    font-size: 28px;
  }
  
  .game-description {
    padding: 15px 20px;
    font-size: 0.95em;
  }
  
  .game-description p {
    margin: 8px 0;
    font-size: 0.95em;
  }
  
  .instructions-title {
    font-size: 1em;
  }
  
  .desktop-instructions,
  .mobile-instructions {
    font-size: 0.9em;
    margin: 6px 0;
  }
  
  .leaderboard {
    padding: 20px;
  }
  
  .leaderboard-item {
    padding: 12px 15px;
  }
}
</style>

<script>
class SnakeGame {
  constructor() {
    this.canvas = document.getElementById('game-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.gridSize = 20;
    this.tileCount = this.canvas.width / this.gridSize;
    
    this.snake = [];
    this.food = {};
    this.direction = 'right';
    this.nextDirection = 'right';
    this.score = 0;
    this.highScore = 0;
    this.gameLoop = null;
    this.gameSpeed = 100;
    this.isGameRunning = false;
    
    this.init();
  }
  
  init() {
    this.loadHighScore();
    this.loadLeaderboard();
    this.loadPlayerName();
    this.bindEvents();
  }
  
  bindEvents() {
    document.getElementById('start-btn').addEventListener('click', () => this.startGame());
    document.getElementById('restart-btn').addEventListener('click', () => this.restartGame());
    document.getElementById('home-btn').addEventListener('click', () => {
      window.location.href = '/games';
    });
    
    document.addEventListener('keydown', (e) => this.handleKeyPress(e));
    
    // Mobile control buttons
    document.querySelectorAll('.control-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        const direction = btn.dataset.direction;
        this.handleMobileControl(direction);
      });
      
      btn.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const direction = btn.dataset.direction;
        this.handleMobileControl(direction);
      });
    });
    
    // Swipe gesture support
    this.setupSwipeGestures();
  }
  
  handleMobileControl(direction) {
    if (!this.isGameRunning) return;
    
    switch (direction) {
      case 'up':
        if (this.direction !== 'down') {
          this.nextDirection = 'up';
        }
        break;
      case 'down':
        if (this.direction !== 'up') {
          this.nextDirection = 'down';
        }
        break;
      case 'left':
        if (this.direction !== 'right') {
          this.nextDirection = 'left';
        }
        break;
      case 'right':
        if (this.direction !== 'left') {
          this.nextDirection = 'right';
        }
        break;
    }
  }
  
  setupSwipeGestures() {
    let touchStartX = 0;
    let touchStartY = 0;
    let touchEndX = 0;
    let touchEndY = 0;
    
    this.canvas.addEventListener('touchstart', (e) => {
      touchStartX = e.changedTouches[0].screenX;
      touchStartY = e.changedTouches[0].screenY;
    }, { passive: true });
    
    this.canvas.addEventListener('touchend', (e) => {
      if (!this.isGameRunning) return;
      
      touchEndX = e.changedTouches[0].screenX;
      touchEndY = e.changedTouches[0].screenY;
      
      this.handleSwipe(touchStartX, touchStartY, touchEndX, touchEndY);
    }, { passive: true });
  }
  
  handleSwipe(startX, startY, endX, endY) {
    const deltaX = endX - startX;
    const deltaY = endY - startY;
    const minSwipeDistance = 30;
    
    if (Math.abs(deltaX) < minSwipeDistance && Math.abs(deltaY) < minSwipeDistance) {
      return;
    }
    
    if (Math.abs(deltaX) > Math.abs(deltaY)) {
      // Horizontal swipe
      if (deltaX > 0 && this.direction !== 'left') {
        this.nextDirection = 'right';
      } else if (deltaX < 0 && this.direction !== 'right') {
        this.nextDirection = 'left';
      }
    } else {
      // Vertical swipe
      if (deltaY > 0 && this.direction !== 'up') {
        this.nextDirection = 'down';
      } else if (deltaY < 0 && this.direction !== 'down') {
        this.nextDirection = 'up';
      }
    }
  }
  
  handleKeyPress(e) {
    if (!this.isGameRunning) return;
    
    const key = e.key.toLowerCase();
    
    if ((key === 'arrowup' || key === 'w') && this.direction !== 'down') {
      this.nextDirection = 'up';
    } else if ((key === 'arrowdown' || key === 's') && this.direction !== 'up') {
      this.nextDirection = 'down';
    } else if ((key === 'arrowleft' || key === 'a') && this.direction !== 'right') {
      this.nextDirection = 'left';
    } else if ((key === 'arrowright' || key === 'd') && this.direction !== 'left') {
      this.nextDirection = 'right';
    }
    
    e.preventDefault();
  }
  
  startGame() {
    let playerName = document.getElementById('player-name').value.trim();
    
    if (!playerName) {
      const savedName = localStorage.getItem('snakePlayerName');
      if (savedName) {
        playerName = savedName;
      } else {
        alert('Please enter your name!');
        return;
      }
    }
    
    this.playerName = playerName;
    localStorage.setItem('snakePlayerName', playerName);
    
    this.resetGame();
    
    document.getElementById('game-start-screen').style.display = 'none';
    document.getElementById('game-over-screen').style.display = 'none';
    
    this.isGameRunning = true;
    this.gameLoop = setInterval(() => this.update(), this.gameSpeed);
  }
  
  resetGame() {
    this.snake = [
      { x: 10, y: 10 },
      { x: 9, y: 10 },
      { x: 8, y: 10 }
    ];
    this.direction = 'right';
    this.nextDirection = 'right';
    this.score = 0;
    this.updateScoreDisplay();
    this.placeFood();
  }
  
  update() {
    this.direction = this.nextDirection;
    
    const head = { ...this.snake[0] };
    
    switch (this.direction) {
      case 'up':
        head.y--;
        break;
      case 'down':
        head.y++;
        break;
      case 'left':
        head.x--;
        break;
      case 'right':
        head.x++;
        break;
    }
    
    // Wrap around walls
    if (head.x < 0) {
      head.x = this.tileCount - 1;
    } else if (head.x >= this.tileCount) {
      head.x = 0;
    }
    
    if (head.y < 0) {
      head.y = this.tileCount - 1;
    } else if (head.y >= this.tileCount) {
      head.y = 0;
    }
    
    if (this.checkCollision(head)) {
      this.gameOver();
      return;
    }
    
    this.snake.unshift(head);
    
    if (head.x === this.food.x && head.y === this.food.y) {
      this.score += 10;
      this.updateScoreDisplay();
      this.placeFood();
      this.increaseSpeed();
    } else {
      this.snake.pop();
    }
    
    this.draw();
  }
  
  checkCollision(head) {
    for (let i = 0; i < this.snake.length; i++) {
      if (head.x === this.snake[i].x && head.y === this.snake[i].y) {
        return true;
      }
    }
    
    return false;
  }
  
  placeFood() {
    let validPosition = false;
    
    while (!validPosition) {
      this.food = {
        x: Math.floor(Math.random() * this.tileCount),
        y: Math.floor(Math.random() * this.tileCount)
      };
      
      validPosition = true;
      
      for (let segment of this.snake) {
        if (this.food.x === segment.x && this.food.y === segment.y) {
          validPosition = false;
          break;
        }
      }
    }
  }
  
  increaseSpeed() {
    if (this.gameSpeed > 50) {
      this.gameSpeed -= 2;
      clearInterval(this.gameLoop);
      this.gameLoop = setInterval(() => this.update(), this.gameSpeed);
    }
  }
  
  draw() {
    this.ctx.fillStyle = '#1a1a1a';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.ctx.fillStyle = '#667eea';
    this.snake.forEach((segment, index) => {
      if (index === 0) {
        this.ctx.fillStyle = '#764ba2';
      } else {
        this.ctx.fillStyle = '#667eea';
      }
      
      this.ctx.fillRect(
        segment.x * this.gridSize + 1,
        segment.y * this.gridSize + 1,
        this.gridSize - 2,
        this.gridSize - 2
      );
    });
    
    this.ctx.fillStyle = '#ff6b6b';
    this.ctx.beginPath();
    this.ctx.arc(
      this.food.x * this.gridSize + this.gridSize / 2,
      this.food.y * this.gridSize + this.gridSize / 2,
      this.gridSize / 2 - 2,
      0,
      Math.PI * 2
    );
    this.ctx.fill();
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
    clearInterval(this.gameLoop);
    
    document.getElementById('final-score').textContent = this.score;
    document.getElementById('game-over-screen').style.display = 'block';
    
    this.saveScore();
    this.loadLeaderboard();
  }
  
  restartGame() {
    document.getElementById('game-over-screen').style.display = 'none';
    this.gameSpeed = 100;
    this.resetGame();
    this.isGameRunning = true;
    this.gameLoop = setInterval(() => this.update(), this.gameSpeed);
  }
  
  saveHighScore() {
    localStorage.setItem('snakeHighScore', this.highScore.toString());
  }
  
  loadHighScore() {
    const savedHighScore = localStorage.getItem('snakeHighScore');
    if (savedHighScore) {
      this.highScore = parseInt(savedHighScore);
      document.getElementById('high-score').textContent = this.highScore;
    }
  }
  
  loadPlayerName() {
    const savedName = localStorage.getItem('snakePlayerName');
    if (savedName) {
      document.getElementById('player-name').value = savedName;
    }
  }
  
  saveScore() {
    let leaderboard = this.getLeaderboard();
    
    leaderboard.push({
      name: this.playerName,
      score: this.score,
      date: new Date().toISOString()
    });
    
    leaderboard.sort((a, b) => b.score - a.score);
    leaderboard = leaderboard.slice(0, 20);
    
    localStorage.setItem('snakeLeaderboard', JSON.stringify(leaderboard));
  }
  
  getLeaderboard() {
    const savedLeaderboard = localStorage.getItem('snakeLeaderboard');
    return savedLeaderboard ? JSON.parse(savedLeaderboard) : [];
  }
  
  loadLeaderboard() {
    const leaderboard = this.getLeaderboard();
    const leaderboardList = document.getElementById('leaderboard-list');
    
    if (leaderboard.length === 0) {
      leaderboardList.innerHTML = '<p class="no-scores">No scores yet. Be the first!</p>';
      return;
    }
    
    leaderboardList.innerHTML = leaderboard.map((entry, index) => `
      <div class="leaderboard-item">
        <span class="rank">${index + 1}</span>
        <span class="player-name">${this.escapeHtml(entry.name)}</span>
        <span class="player-score">${entry.score}</span>
      </div>
    `).join('');
  }
  
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new SnakeGame();
});
</script>
