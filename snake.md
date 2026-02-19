---
layout: page
title: Snake Game
permalink: /games/snake/
---

<div class="snake-game-container">
  <h1>🐍 Snake Game</h1>
  
  <div class="game-description">
    <p><strong>Objective:</strong> Guide the snake to eat food and grow as long as possible without hitting yourself!</p>
    <p><strong>How to Play:</strong> Use arrow keys (↑↓←→) or WASD to control the snake's direction. On mobile, use the on-screen buttons or swipe on the canvas.</p>
    <p><strong>Rules:</strong> The snake wraps around walls - use this to your advantage! Avoid hitting your own body.</p>
    <p><strong>Scoring:</strong> +10 points for each food eaten. The game speeds up as you score more!</p>
  </div>
  
  <div class="game-wrapper">
    <div class="score-display">
      <span class="score-left">Score: <strong id="current-score">0</strong></span>
      <button id="sound-toggle" class="sound-toggle" title="Toggle Sound">🔊</button>
      <span class="score-right">Top: <strong id="high-score">0</strong></span>
    </div>
    
    <div id="game-over-screen" class="game-over-screen" style="display: none;">
      <h2>Game Over!</h2>
      <p>Your Score: <strong id="final-score">0</strong></p>
      <button id="restart-btn" class="restart-btn">Play Again</button>
      <button id="home-btn" class="home-btn">Back to Games</button>
    </div>
    
    <canvas id="game-canvas" width="400" height="400"></canvas>
    
    <div class="mobile-controls">
      <div class="control-header">
        <button id="lock-toggle" class="lock-toggle" title="Lock/Unlock Controls">🔓</button>
        <span id="lock-status" class="lock-status">Unlocked</span>
      </div>
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

.score-display {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #0d0d0d;
  padding: 12px 20px;
  border-radius: 10px 10px 0 0;
  border: 4px solid #667eea;
  border-bottom: none;
  color: white;
  font-size: 1.1em;
  font-weight: bold;
  width: 408px;
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

#game-canvas {
  display: block;
  margin: 0 auto;
  background: #1a1a1a;
  border: 4px solid #667eea;
  border-top: none;
  border-radius: 0 0 10px 10px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
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
  .snake-game-container h1 {
    font-size: 2em;
  }
  
  #game-canvas {
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
    this.gameSpeed = 200;
    this.isGameRunning = false;
    this.soundEnabled = true;
    this.audioContext = null;
    
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
      console.error('Snake game buttons not found in DOM');
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
        const direction = btn.dataset.direction;
        this.handleMobileControl(direction);
      });
      
      btn.addEventListener('touchstart', (e) => {
        e.preventDefault();
        this.initAudioContext();
        const direction = btn.dataset.direction;
        this.handleMobileControl(direction);
      });
    });
    
    this.setupSwipeGestures();
    this.setupDraggableControls();
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
    
    this.canvas.addEventListener('touchstart', (e) => {
      touchStartX = e.changedTouches[0].screenX;
      touchStartY = e.changedTouches[0].screenY;
    }, { passive: true });
    
    this.canvas.addEventListener('touchend', (e) => {
      if (!this.isGameRunning) return;
      
      const touchEndX = e.changedTouches[0].screenX;
      const touchEndY = e.changedTouches[0].screenY;
      
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
      if (deltaX > 0 && this.direction !== 'left') {
        this.nextDirection = 'right';
      } else if (deltaX < 0 && this.direction !== 'right') {
        this.nextDirection = 'left';
      }
    } else {
      if (deltaY > 0 && this.direction !== 'up') {
        this.nextDirection = 'down';
      } else if (deltaY < 0 && this.direction !== 'down') {
        this.nextDirection = 'up';
      }
    }
  }
  
  handleKeyPress(e) {
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
    this.resetGame();
    
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
      this.increaseSpeed();
      this.playEatSound();
      this.placeFood();
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
    
    this.snake.forEach((segment, index) => {
      if (index === 0) {
        this.ctx.fillStyle = '#32CD32';
      } else {
        this.ctx.fillStyle = '#228B22';
      }
      
      this.ctx.fillRect(
        segment.x * this.gridSize + 1,
        segment.y * this.gridSize + 1,
        this.gridSize - 2,
        this.gridSize - 2
      );
    });
    
    this.ctx.fillStyle = '#ff0000';
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
  }
  
  restartGame() {
    document.getElementById('game-over-screen').style.display = 'none';
    this.gameSpeed = 200;
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
  
  playEatSound() {
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
      
      oscillator.frequency.value = 800;
      oscillator.type = 'square';
      
      gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.1);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.1);
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
    const game = new SnakeGame();
    console.log('Snake game initialized successfully');
  } catch (error) {
    console.error('Error initializing Snake game:', error);
  }
});
</script>
