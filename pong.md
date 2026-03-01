---
layout: page
title: Pong
permalink: /games/pong/
---

<div class="pong-container">
  <h1>🏓 Pong</h1>
  
  <div class="game-description">
    <p><strong>Objective:</strong> Score points by getting the ball past your opponent's paddle!</p>
    <p><strong>How to Play:</strong> Use arrow keys (↑↓) or W/S to move your paddle. On mobile, use on-screen buttons.</p>
    <p><strong>Rules:</strong> First to 5 points wins the level. Complete all 10 levels to win the game!</p>
    <p><strong>Levels:</strong> Ball speed increases with each level. Can you beat all 10 levels?</p>
  </div>
  
  <div class="game-wrapper">
    <div class="score-display">
      <span class="score-left">Player: <strong id="player-score">0</strong></span>
      <span class="score-center">Level: <strong id="current-level">1</strong></span>
      <button id="sound-toggle" class="sound-toggle" title="Toggle Sound">🔊</button>
      <span class="score-right">AI: <strong id="ai-score">0</strong></span>
    </div>
    
    <div class="game-area">
      <canvas id="game-canvas" width="600" height="400"></canvas>
    </div>
    
    <div id="game-over-screen" class="game-over-screen" style="display: none;">
      <h2 id="winner-text">You Won!</h2>
      <p>Final Score: <strong id="final-score">0 - 0</strong></p>
      <button id="restart-btn" class="restart-btn">Play Again</button>
      <button id="home-btn" class="home-btn">Back to Games</button>
    </div>
    
    <div class="mobile-controls">
      <div class="control-grid">
        <div></div>
        <button class="control-btn up-btn" data-action="up">▲</button>
        <div class="lock-container">
          <button id="lock-toggle" class="lock-toggle" title="Lock/Unlock Controls">🔓</button>
          <span id="lock-status" class="lock-status">UNLOCKED</span>
        </div>
        <div></div>
        <button class="control-btn down-btn" data-action="down">▼</button>
        <div></div>
      </div>
    </div>
  </div>
</div>

<style>
.pong-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
}

.pong-container h1 {
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
  max-width: 650px;
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
  width: 608px;
  margin: 0 auto;
  max-width: 100%;
  box-sizing: border-box;
}

.score-left, .score-right, .score-center {
  flex: 1;
}

.score-left {
  text-align: left;
}

.score-center {
  text-align: center;
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
  margin-bottom: 20px;
}

#game-canvas {
  display: block;
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
  padding: 12px;
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

.control-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  grid-template-rows: repeat(2, auto);
  gap: 8px;
  align-items: center;
}

.lock-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.lock-toggle {
  background: rgba(255, 255, 255, 0.2);
  border: none;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  font-size: 16px;
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
  font-size: 9px;
  color: rgba(255, 255, 255, 0.8);
  margin-top: 2px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
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
  .pong-container h1 {
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
class PongGame {
  constructor() {
    this.canvas = document.getElementById('game-canvas');
    this.ctx = this.canvas.getContext('2d');
    
    this.paddleWidth = 10;
    this.paddleHeight = 80;
    this.ballSize = 15;
    
    this.player = {
      x: 10,
      y: this.canvas.height / 2 - this.paddleHeight / 2,
      score: 0,
      speed: 8
    };
    
    this.ai = {
      x: this.canvas.width - 20,
      y: this.canvas.height / 2 - this.paddleHeight / 2,
      score: 0,
      speed: 5
    };
    
    this.ball = {
      x: this.canvas.width / 2,
      y: this.canvas.height / 2,
      speedX: 3,
      speedY: 3,
      baseSpeed: 3
    };
    
    this.winningScore = 5;
    this.currentLevel = 1;
    this.maxLevels = 10;
    this.isGameRunning = false;
    this.gameLoop = null;
    this.soundEnabled = true;
    this.audioContext = null;
    
    this.keys = {
      up: false,
      down: false
    };
    
    this.init();
  }
  
  init() {
    this.bindEvents();
    this.startGame();
  }
  
  bindEvents() {
    const restartBtn = document.getElementById('restart-btn');
    const homeBtn = document.getElementById('home-btn');
    const soundToggle = document.getElementById('sound-toggle');
    
    if (!restartBtn || !homeBtn || !soundToggle) {
      console.error('Pong game buttons not found in DOM');
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
      this.handleKeyPress(e, true);
    });
    
    document.addEventListener('keyup', (e) => {
      this.handleKeyPress(e, false);
    });
    
    document.querySelectorAll('.control-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        this.initAudioContext();
        const action = btn.dataset.action;
        this.handleMobileControl(action, true);
      });
      
      btn.addEventListener('touchstart', (e) => {
        e.preventDefault();
        this.initAudioContext();
        const action = btn.dataset.action;
        this.handleMobileControl(action, true);
      });
      
      btn.addEventListener('touchend', (e) => {
        e.preventDefault();
        const action = btn.dataset.action;
        this.handleMobileControl(action, false);
      });
    });
    
    this.setupDraggableControls();
  }
  
  handleKeyPress(e, isPressed) {
    const key = e.key.toLowerCase();
    
    if (key === 'arrowup' || key === 'w') {
      this.keys.up = isPressed;
    } else if (key === 'arrowdown' || key === 's') {
      this.keys.down = isPressed;
    }
    
    e.preventDefault();
  }
  
  handleMobileControl(action, isPressed) {
    if (action === 'up') {
      this.keys.up = isPressed;
    } else if (action === 'down') {
      this.keys.down = isPressed;
    }
  }
  
  startGame() {
    this.resetGame();
    
    document.getElementById('game-over-screen').style.display = 'none';
    
    this.isGameRunning = true;
    this.gameLoop = requestAnimationFrame(() => this.update());
  }
  
  resetGame() {
    this.player.y = this.canvas.height / 2 - this.paddleHeight / 2;
    this.ai.y = this.canvas.height / 2 - this.paddleHeight / 2;
    this.player.score = 0;
    this.ai.score = 0;
    this.currentLevel = 1;
    
    this.resetBall();
    this.updateScoreDisplay();
  }
  
  resetBall() {
    this.ball.x = this.canvas.width / 2;
    this.ball.y = this.canvas.height / 2;
    const speedMultiplier = 1 + (this.currentLevel - 1) * 0.3;
    this.ball.speedX = this.ball.baseSpeed * speedMultiplier * (Math.random() > 0.5 ? 1 : -1);
    this.ball.speedY = this.ball.baseSpeed * speedMultiplier * (Math.random() > 0.5 ? 1 : -1);
  }
  
  update() {
    if (!this.isGameRunning) return;
    
    this.movePlayer();
    this.moveAI();
    this.moveBall();
    this.checkCollisions();
    this.draw();
    
    this.gameLoop = requestAnimationFrame(() => this.update());
  }
  
  movePlayer() {
    if (this.keys.up && this.player.y > 0) {
      this.player.y -= this.player.speed;
    }
    if (this.keys.down && this.player.y < this.canvas.height - this.paddleHeight) {
      this.player.y += this.player.speed;
    }
  }
  
  moveAI() {
    const aiCenter = this.ai.y + this.paddleHeight / 2;
    const targetY = this.ball.y - this.paddleHeight / 2;
    
    if (aiCenter < this.ball.y - 10) {
      this.ai.y += this.ai.speed;
    } else if (aiCenter > this.ball.y + 10) {
      this.ai.y -= this.ai.speed;
    }
    
    this.ai.y = Math.max(0, Math.min(this.ai.y, this.canvas.height - this.paddleHeight));
  }
  
  moveBall() {
    this.ball.x += this.ball.speedX;
    this.ball.y += this.ball.speedY;
    
    if (this.ball.y <= 0 || this.ball.y >= this.canvas.height - this.ballSize) {
      this.ball.speedY *= -1;
      this.playBounceSound();
    }
  }
  
  checkCollisions() {
    if (this.ball.x <= this.player.x + this.paddleWidth) {
      if (this.ball.y >= this.player.y && this.ball.y <= this.player.y + this.paddleHeight) {
        this.ball.speedX *= -1.1;
        this.ball.x = this.player.x + this.paddleWidth + 1;
        this.playHitSound();
      }
    }
    
    if (this.ball.x >= this.ai.x - this.ballSize) {
      if (this.ball.y >= this.ai.y && this.ball.y <= this.ai.y + this.paddleHeight) {
        this.ball.speedX *= -1.1;
        this.ball.x = this.ai.x - this.ballSize - 1;
        this.playHitSound();
      }
    }
    
    if (this.ball.x < 0) {
      this.ai.score++;
      this.playScoreSound();
      this.updateScoreDisplay();
      this.checkWinner();
      this.resetBall();
    }
    
    if (this.ball.x > this.canvas.width) {
      this.player.score++;
      this.playScoreSound();
      this.updateScoreDisplay();
      this.checkWinner();
      this.resetBall();
    }
  }
  
  checkWinner() {
    if (this.player.score >= this.winningScore) {
      if (this.currentLevel < this.maxLevels) {
        this.currentLevel++;
        this.player.score = 0;
        this.ai.score = 0;
        this.resetBall();
        this.updateScoreDisplay();
      } else {
        this.gameOver();
      }
    } else if (this.ai.score >= this.winningScore) {
      this.gameOver();
    }
  }
  
  gameOver() {
    this.isGameRunning = false;
    cancelAnimationFrame(this.gameLoop);
    
    let winnerText;
    if (this.currentLevel >= this.maxLevels && this.player.score >= this.winningScore) {
      winnerText = '🎉 You Won All Levels! 🎉';
    } else if (this.player.score >= this.winningScore) {
      winnerText = 'You Won!';
    } else {
      winnerText = 'AI Won!';
    }
    document.getElementById('winner-text').textContent = winnerText;
    document.getElementById('final-score').textContent = `${this.player.score} - ${this.ai.score}`;
    document.getElementById('game-over-screen').style.display = 'block';
  }
  
  restartGame() {
    document.getElementById('game-over-screen').style.display = 'none';
    this.resetGame();
    this.isGameRunning = true;
    this.gameLoop = requestAnimationFrame(() => this.update());
  }
  
  draw() {
    this.ctx.fillStyle = '#1a1a1a';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.drawNet();
    this.drawPaddle(this.player.x, this.player.y, '#667eea');
    this.drawPaddle(this.ai.x, this.ai.y, '#764ba2');
    this.drawBall();
  }
  
  drawNet() {
    this.ctx.strokeStyle = '#333';
    this.ctx.setLineDash([10, 10]);
    this.ctx.beginPath();
    this.ctx.moveTo(this.canvas.width / 2, 0);
    this.ctx.lineTo(this.canvas.width / 2, this.canvas.height);
    this.ctx.stroke();
    this.ctx.setLineDash([]);
  }
  
  drawPaddle(x, y, color) {
    this.ctx.fillStyle = color;
    this.ctx.fillRect(x, y, this.paddleWidth, this.paddleHeight);
    
    this.ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    this.ctx.fillRect(x, y, this.paddleWidth, 4);
  }
  
  drawBall() {
    this.ctx.fillStyle = '#fff';
    this.ctx.beginPath();
    this.ctx.arc(this.ball.x, this.ball.y, this.ballSize / 2, 0, Math.PI * 2);
    this.ctx.fill();
  }
  
  updateScoreDisplay() {
    document.getElementById('player-score').textContent = this.player.score;
    document.getElementById('ai-score').textContent = this.ai.score;
    document.getElementById('current-level').textContent = this.currentLevel;
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
  
  playHitSound() {
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
      
      oscillator.frequency.value = 400;
      oscillator.type = 'square';
      
      gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.1);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.1);
    } catch (error) {
      console.error('Error playing sound:', error);
    }
  }
  
  playBounceSound() {
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
      
      oscillator.frequency.value = 300;
      oscillator.type = 'sine';
      
      gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.08);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.08);
    } catch (error) {
      console.error('Error playing sound:', error);
    }
  }
  
  playScoreSound() {
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
      
      oscillator.frequency.value = 600;
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
      lockStatus.textContent = isLocked ? 'LOCKED' : 'UNLOCKED';
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
    const game = new PongGame();
    console.log('Pong game initialized successfully');
  } catch (error) {
    console.error('Error initializing Pong game:', error);
  }
});
</script>
