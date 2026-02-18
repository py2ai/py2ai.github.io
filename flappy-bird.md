---
layout: page
title: Flappy Bird
permalink: /games/flappy-bird/
---

<div class="flappy-bird-container">
  <h1>🐦 Flappy Bird</h1>
  
  <div class="game-description">
    <p><strong>Objective:</strong> Help the bird fly through the pipes without hitting them!</p>
    <p><strong>Rules:</strong> Tap or press space to flap and gain height. Don't hit the pipes or ground!</p>
    <p><strong>Scoring:</strong> +1 point for each pipe passed. Try to beat your high score!</p>
  </div>
  
  <div class="game-wrapper">
    <div class="game-info">
      <div class="score-display">
        <span>Score: <strong id="current-score">0</strong></span>
        <span>High Score: <strong id="high-score">0</strong></span>
      </div>
    </div>
    
    <div id="game-start-screen" class="start-screen">
      <h2>Flappy Bird</h2>
      <button id="start-btn" class="start-btn">Start Game</button>
      <div class="instructions">
        <p class="instructions-title">How to Play:</p>
        <p class="desktop-instructions">🖥️ <strong>Desktop:</strong> Press Space or Click to flap</p>
        <p class="mobile-instructions">📱 <strong>Mobile:</strong> Tap anywhere to flap</p>
      </div>
    </div>
    
    <div id="game-over-screen" class="game-over-screen" style="display: none;">
      <h2>Game Over!</h2>
      <p>Your Score: <strong id="final-score">0</strong></p>
      <button id="restart-btn" class="restart-btn">Play Again</button>
      <button id="home-btn" class="home-btn">Back to Games</button>
    </div>
    
    <canvas id="game-canvas" width="400" height="600"></canvas>
  </div>
</div>

<style>
.flappy-bird-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
}

.flappy-bird-container h1 {
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
  background: linear-gradient(180deg, #87CEEB 0%, #E0F6FF 100%);
  border: 4px solid #667eea;
  border-radius: 10px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  cursor: pointer;
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

@media (max-width: 600px) {
  .flappy-bird-container h1 {
    font-size: 2em;
  }
  
  #game-canvas {
    width: 100%;
    height: auto;
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
}
</style>

<script>
class FlappyBirdGame {
  constructor() {
    this.canvas = document.getElementById('game-canvas');
    this.ctx = this.canvas.getContext('2d');
    
    this.bird = {
      x: 80,
      y: 300,
      width: 40,
      height: 30,
      velocity: 0,
      gravity: 0.3,
      jumpStrength: -6
    };
    
    this.pipes = [];
    this.pipeWidth = 70;
    this.pipeGap = 240;
    this.pipeSpeed = 1.5,
    this.pipeSpawnRate = 150;
    this.frameCount = 0;
    
    this.score = 0;
    this.highScore = 0;
    this.gameLoop = null;
    this.isGameRunning = false;
    
    this.init();
  }
  
  init() {
    this.loadHighScore();
    this.bindEvents();
  }
  
  bindEvents() {
    const startBtn = document.getElementById('start-btn');
    const restartBtn = document.getElementById('restart-btn');
    const homeBtn = document.getElementById('home-btn');
    
    if (!startBtn || !restartBtn || !homeBtn) {
      console.error('Flappy Bird game buttons not found in DOM');
      return;
    }
    
    console.log('Flappy Bird - Adding event listeners to buttons');
    
    startBtn.addEventListener('click', (e) => {
      console.log('Flappy Bird - Start button clicked');
      e.preventDefault();
      this.startGame();
    });
    
    restartBtn.addEventListener('click', (e) => {
      console.log('Flappy Bird - Restart button clicked');
      e.preventDefault();
      this.restartGame();
    });
    
    homeBtn.addEventListener('click', (e) => {
      console.log('Flappy Bird - Home button clicked');
      e.preventDefault();
      window.location.href = '/games';
    });
    
    // Desktop controls
    document.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && this.isGameRunning) {
        e.preventDefault();
        this.flap();
      }
    });
    
    this.canvas.addEventListener('click', () => {
      if (this.isGameRunning) {
        this.flap();
      }
    });
    
    // Mobile controls
    this.canvas.addEventListener('touchstart', (e) => {
      if (this.isGameRunning) {
        e.preventDefault();
        this.flap();
      }
    }, { passive: false });
  }
  
  startGame() {
    this.resetGame();
    
    document.getElementById('game-start-screen').style.display = 'none';
    document.getElementById('game-over-screen').style.display = 'none';
    
    this.isGameRunning = true;
    this.draw();
    this.gameLoop = setInterval(() => this.update(), 20);
  }
  
  resetGame() {
    this.bird = {
      x: 80,
      y: 300,
      width: 40,
      height: 30,
      velocity: 0,
      gravity: 0.3,
      jumpStrength: -6
    };
    
    this.pipes = [];
    this.frameCount = 0;
    this.score = 0;
    this.updateScoreDisplay();
  }
  
  flap() {
    this.bird.velocity = this.bird.jumpStrength;
  }
  
  update() {
    this.frameCount++;
    
    // Update bird
    this.bird.velocity += this.bird.gravity;
    this.bird.y += this.bird.velocity;
    
    // Spawn pipes
    if (this.frameCount % this.pipeSpawnRate === 0) {
      this.spawnPipe();
    }
    
    // Update pipes
    this.pipes.forEach(pipe => {
      pipe.x -= this.pipeSpeed;
      
      // Score when passing pipe
      if (!pipe.passed && pipe.x + this.pipeWidth < this.bird.x) {
        pipe.passed = true;
        this.score++;
        this.updateScoreDisplay();
      }
    });
    
    // Remove off-screen pipes
    this.pipes = this.pipes.filter(pipe => pipe.x + this.pipeWidth > 0);
    
    // Check collisions
    if (this.checkCollision()) {
      this.gameOver();
      return;
    }
    
    this.draw();
  }
  
  spawnPipe() {
    const minHeight = 80;
    const maxHeight = this.canvas.height - this.pipeGap - minHeight - 50;
    const topHeight = Math.random() * (maxHeight - minHeight) + minHeight;
    
    this.pipes.push({
      x: this.canvas.width,
      topHeight: topHeight,
      bottomY: topHeight + this.pipeGap,
      passed: false
    });
  }
  
  checkCollision() {
    // Ground collision
    if (this.bird.y + this.bird.height >= this.canvas.height - 20) {
      return true;
    }
    
    // Ceiling collision - allow bird to go slightly above
    if (this.bird.y < -this.bird.height) {
      return true;
    }
    
    // Pipe collision - with smaller hitbox for easier gameplay
    const hitboxPadding = 5;
    
    for (let pipe of this.pipes) {
      // Check if bird is within pipe's x range (with padding)
      if (this.bird.x + this.bird.width - hitboxPadding > pipe.x && 
          this.bird.x + hitboxPadding < pipe.x + this.pipeWidth) {
        // Check top pipe collision (with padding)
        if (this.bird.y + hitboxPadding < pipe.topHeight) {
          return true;
        }
        
        // Check bottom pipe collision (with padding)
        if (this.bird.y + this.bird.height - hitboxPadding > pipe.bottomY) {
          return true;
        }
      }
    }
    
    return false;
  }
  
  draw() {
    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Draw ground
    this.ctx.fillStyle = '#8B4513';
    this.ctx.fillRect(0, this.canvas.height - 20, this.canvas.width, 20);
    
    // Draw grass
    this.ctx.fillStyle = '#228B22';
    this.ctx.fillRect(0, this.canvas.height - 25, this.canvas.width, 5);
    
    // Draw pipes
    this.pipes.forEach(pipe => {
      // Top pipe
      this.ctx.fillStyle = '#4CAF50';
      this.ctx.fillRect(pipe.x, 0, this.pipeWidth, pipe.topHeight);
      
      // Top pipe cap
      this.ctx.fillStyle = '#45a049';
      this.ctx.fillRect(pipe.x - 5, pipe.topHeight - 20, this.pipeWidth + 10, 20);
      
      // Bottom pipe
      this.ctx.fillStyle = '#4CAF50';
      this.ctx.fillRect(pipe.x, pipe.bottomY, this.pipeWidth, this.canvas.height - pipe.bottomY);
      
      // Bottom pipe cap
      this.ctx.fillStyle = '#45a049';
      this.ctx.fillRect(pipe.x - 5, pipe.bottomY, this.pipeWidth + 10, 20);
    });
    
    // Draw bird
    this.ctx.fillStyle = '#FFC107';
    this.ctx.beginPath();
    this.ctx.ellipse(
      this.bird.x + this.bird.width / 2,
      this.bird.y + this.bird.height / 2,
      this.bird.width / 2,
      this.bird.height / 2,
      0,
      0,
      Math.PI * 2
    );
    this.ctx.fill();
    
    // Bird eye
    this.ctx.fillStyle = 'white';
    this.ctx.beginPath();
    this.ctx.arc(
      this.bird.x + this.bird.width * 0.7,
      this.bird.y + this.bird.height * 0.3,
      8,
      0,
      Math.PI * 2
    );
    this.ctx.fill();
    
    // Bird pupil
    this.ctx.fillStyle = 'black';
    this.ctx.beginPath();
    this.ctx.arc(
      this.bird.x + this.bird.width * 0.75,
      this.bird.y + this.bird.height * 0.3,
      4,
      0,
      Math.PI * 2
    );
    this.ctx.fill();
    
    // Bird beak
    this.ctx.fillStyle = '#FF5722';
    this.ctx.beginPath();
    this.ctx.moveTo(this.bird.x + this.bird.width, this.bird.y + this.bird.height / 2);
    this.ctx.lineTo(this.bird.x + this.bird.width + 15, this.bird.y + this.bird.height / 2 + 5);
    this.ctx.lineTo(this.bird.x + this.bird.width, this.bird.y + this.bird.height / 2 + 10);
    this.ctx.closePath();
    this.ctx.fill();
    
    // Bird wings
    this.ctx.fillStyle = '#4CAF50';
    const wingOffset = Math.sin(this.frameCount * 0.3) * 5;
    
    // Left wing
    this.ctx.beginPath();
    this.ctx.ellipse(
      this.bird.x + this.bird.width * 0.3,
      this.bird.y + this.bird.height * 0.6 + wingOffset,
      12,
      8,
      -0.3,
      0,
      Math.PI * 2
    );
    this.ctx.fill();
    
    // Right wing
    this.ctx.beginPath();
    this.ctx.ellipse(
      this.bird.x + this.bird.width * 0.5,
      this.bird.y + this.bird.height * 0.6 + wingOffset,
      12,
      8,
      0.3,
      0,
      Math.PI * 2
    );
    this.ctx.fill();
    
    // Bird tail
    this.ctx.fillStyle = '#4CAF50';
    const tailOffset = Math.sin(this.frameCount * 0.2) * 3;
    
    // Tail feathers
    this.ctx.beginPath();
    this.ctx.moveTo(this.bird.x, this.bird.y + this.bird.height / 2);
    this.ctx.lineTo(this.bird.x - 15, this.bird.y + this.bird.height / 2 - 5 + tailOffset);
    this.ctx.lineTo(this.bird.x - 15, this.bird.y + this.bird.height / 2 + 5 + tailOffset);
    this.ctx.closePath();
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
    this.resetGame();
    this.isGameRunning = true;
    this.draw();
    this.gameLoop = setInterval(() => this.update(), 20);
  }
  
  saveHighScore() {
    localStorage.setItem('flappyBirdHighScore', this.highScore.toString());
  }
  
  loadHighScore() {
    const savedHighScore = localStorage.getItem('flappyBirdHighScore');
    if (savedHighScore) {
      this.highScore = parseInt(savedHighScore);
      document.getElementById('high-score').textContent = this.highScore;
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  try {
    const game = new FlappyBirdGame();
    console.log('Flappy Bird game initialized successfully');
  } catch (error) {
    console.error('Error initializing Flappy Bird game:', error);
  }
});
</script>
