---
layout: page
title: Copter Game
permalink: /games/copter/
---

<div class="copter-game-container">
  <h1>🚁 Copter Game</h1>
  
  <div class="game-description">
    <p><strong>Objective:</strong> Fly the helicopter through obstacles and survive as long as possible!</p>
    <p><strong>How to Play:</strong> 
       <br>🖱️ <strong>Mouse:</strong> Click or hold to fly up, release to fall down
       <br>⌨️ <strong>Keyboard:</strong> Press or hold SPACE to fly up, release to fall down
       <br>📱 <strong>Touch:</strong> Tap or hold to fly up, release to fall down
    </p>
    <p><strong>Rules:</strong> 
       <br>• Avoid obstacles (buildings and barriers)
       <br>• Don't hit the ground or ceiling
       <br>• Score increases as you survive longer
       <br>• Try to beat your high score!
    </p>
    <p><strong>Controls:</strong> Click/Tap/Space to fly up | Restart button to play again</p>
  </div>
  
  <div class="game-wrapper">
    <div class="game-info">
      <div class="score-panel">
        <div class="score-item">
          <span class="score-label">Score:</span>
          <span class="score-value" id="score">0</span>
        </div>
        <div class="score-item">
          <span class="score-label">High Score:</span>
          <span class="score-value" id="high-score">0</span>
        </div>
      </div>
      
      <div class="game-controls">
        <button id="restart-btn" class="game-btn">Restart</button>
        <button id="home-btn" class="game-btn">Back to Games</button>
      </div>
      
      <div class="difficulty-selector">
        <label for="difficulty">Difficulty:</label>
        <select id="difficulty">
          <option value="easy">Easy</option>
          <option value="medium" selected>Medium</option>
          <option value="hard">Hard</option>
        </select>
      </div>
    </div>
    
    <div id="game-over-screen" class="game-over-screen" style="display: none;">
      <h2>Game Over!</h2>
      <p>Your Score: <strong id="final-score">0</strong></p>
      <p>High Score: <strong id="final-high-score">0</strong></p>
      <button id="play-again-btn" class="restart-btn">Play Again</button>
    </div>
    
    <div class="board-container">
      <canvas id="copter-board" width="800" height="500"></canvas>
    </div>
  </div>
</div>

<style>
.copter-game-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
}

.copter-game-container h1 {
  text-align: center;
  font-size: 2.5em;
  margin-bottom: 30px;
  color: #333;
}

.game-description {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 25px;
  border-radius: 15px;
  margin-bottom: 30px;
  color: white;
  box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

.game-description p {
  margin: 10px 0;
  line-height: 1.6;
}

.game-description strong {
  color: #ffd700;
}

.game-wrapper {
  display: flex;
  gap: 30px;
  align-items: flex-start;
  justify-content: center;
  flex-wrap: wrap;
}

.game-info {
  flex: 1;
  min-width: 250px;
  max-width: 300px;
}

.score-panel {
  background: white;
  padding: 20px;
  border-radius: 10px;
  margin-bottom: 20px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.score-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid #eee;
}

.score-item:last-child {
  border-bottom: none;
}

.score-label {
  font-weight: bold;
  color: #666;
  font-size: 1.1em;
}

.score-value {
  font-size: 1.5em;
  font-weight: bold;
  color: #667eea;
}

.game-controls {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 20px;
}

.game-btn {
  padding: 12px 20px;
  font-size: 1em;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  font-weight: bold;
}

.game-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.game-btn:active {
  transform: translateY(0);
}

.difficulty-selector {
  background: white;
  padding: 15px;
  border-radius: 10px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.difficulty-selector label {
  display: block;
  font-weight: bold;
  color: #333;
  margin-bottom: 10px;
}

.difficulty-selector select {
  width: 100%;
  padding: 10px;
  font-size: 1em;
  border: 2px solid #667eea;
  border-radius: 5px;
  background: white;
  cursor: pointer;
}

.board-container {
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #87CEEB 0%, #98D8C8 100%);
  padding: 30px;
  border-radius: 15px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

#copter-board {
  display: block;
  background: #87CEEB;
  border: 4px solid #333;
  border-radius: 5px;
  cursor: pointer;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}

.game-over-screen {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: white;
  padding: 40px;
  border-radius: 15px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  text-align: center;
  z-index: 1000;
  min-width: 300px;
}

.game-over-screen h2 {
  font-size: 2em;
  color: #e74c3c;
  margin-bottom: 20px;
}

.game-over-screen p {
  font-size: 1.2em;
  color: #333;
  margin: 10px 0;
}

.game-over-screen strong {
  color: #667eea;
  font-size: 1.3em;
}

.restart-btn {
  margin-top: 20px;
  padding: 15px 30px;
  font-size: 1.1em;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  font-weight: bold;
}

.restart-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.restart-btn:active {
  transform: translateY(0);
}

@media (max-width: 768px) {
  .game-wrapper {
    flex-direction: column;
    align-items: center;
  }
  
  .game-info {
    width: 100%;
    max-width: none;
  }
  
  #copter-board {
    width: 100%;
    height: auto;
  }
}
</style>

<script>
class CopterGame {
  constructor() {
    this.canvas = document.getElementById('copter-board');
    this.ctx = this.canvas.getContext('2d');
    
    this.copter = {
      x: 100,
      y: 250,
      width: 60,
      height: 30,
      velocity: 0,
      gravity: 0.3,
      lift: -4
    };
    
    this.obstacles = [];
    this.score = 0;
    this.highScore = parseInt(localStorage.getItem('copterHighScore')) || 0;
    this.gameOver = false;
    this.gameStarted = false;
    this.isFlyingUp = false;
    this.obstacleSpeed = 3;
    this.obstacleFrequency = 150;
    this.frameCount = 0;
    
    this.difficultySettings = {
      easy: { speed: 2, frequency: 200, gravity: 0.2 },
      medium: { speed: 3, frequency: 150, gravity: 0.3 },
      hard: { speed: 4, frequency: 100, gravity: 0.4 }
    };
    
    this.currentDifficulty = 'medium';
    
    this.init();
  }
  
  init() {
    this.bindEvents();
    this.updateHighScoreDisplay();
    this.draw();
  }
  
  bindEvents() {
    const restartBtn = document.getElementById('restart-btn');
    const homeBtn = document.getElementById('home-btn');
    const playAgainBtn = document.getElementById('play-again-btn');
    const difficultySelect = document.getElementById('difficulty');
    
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
    
    if (playAgainBtn) {
      playAgainBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.restart();
      });
    }
    
    if (difficultySelect) {
      difficultySelect.addEventListener('change', (e) => {
        this.currentDifficulty = e.target.value;
        this.applyDifficulty();
      });
    }
    
    this.canvas.addEventListener('mousedown', (e) => {
      e.preventDefault();
      this.handleInputStart();
    });
    
    this.canvas.addEventListener('mouseup', (e) => {
      e.preventDefault();
      this.handleInputEnd();
    });
    
    this.canvas.addEventListener('mouseleave', (e) => {
      e.preventDefault();
      this.handleInputEnd();
    });
    
    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.handleInputStart();
    });
    
    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      this.handleInputEnd();
    });
    
    document.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        this.handleInputStart();
      }
    });
    
    document.addEventListener('keyup', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        this.handleInputEnd();
      }
    });
  }
  
  handleInputStart() {
    if (this.gameOver) {
      this.restart();
      return;
    }
    
    if (!this.gameStarted) {
      this.gameStarted = true;
      this.gameLoop();
    }
    
    this.isFlyingUp = true;
  }
  
  handleInputEnd() {
    this.isFlyingUp = false;
  }
  
  applyDifficulty() {
    const settings = this.difficultySettings[this.currentDifficulty];
    this.obstacleSpeed = settings.speed;
    this.obstacleFrequency = settings.frequency;
    this.copter.gravity = settings.gravity;
  }
  
  update() {
    if (this.isFlyingUp) {
      this.copter.velocity = this.copter.lift;
    } else {
      this.copter.velocity += this.copter.gravity;
    }
    
    this.copter.y += this.copter.velocity;
    
    if (this.copter.y < 0) {
      this.copter.y = 0;
      this.copter.velocity = 0;
    }
    
    if (this.copter.y + this.copter.height > this.canvas.height) {
      this.endGame();
      return;
    }
    
    this.frameCount++;
    
    if (this.frameCount % this.obstacleFrequency === 0) {
      this.addObstacle();
    }
    
    this.updateObstacles();
    this.checkCollisions();
    
    if (!this.gameOver) {
      this.score++;
      this.updateScoreDisplay();
    }
  }
  
  addObstacle() {
    const minHeight = 50;
    const maxHeight = this.canvas.height - 150;
    const gapHeight = 150;
    const topHeight = Math.random() * (maxHeight - minHeight) + minHeight;
    const bottomY = topHeight + gapHeight;
    const bottomHeight = this.canvas.height - bottomY;
    
    this.obstacles.push({
      x: this.canvas.width,
      topHeight: topHeight,
      bottomY: bottomY,
      bottomHeight: bottomHeight,
      width: 60,
      passed: false
    });
  }
  
  updateObstacles() {
    for (let i = this.obstacles.length - 1; i >= 0; i--) {
      const obstacle = this.obstacles[i];
      obstacle.x -= this.obstacleSpeed;
      
      if (obstacle.x + obstacle.width < 0) {
        this.obstacles.splice(i, 1);
      }
    }
  }
  
  checkCollisions() {
    const copterBox = {
      x: this.copter.x + 5,
      y: this.copter.y + 5,
      width: this.copter.width - 10,
      height: this.copter.height - 10
    };
    
    for (const obstacle of this.obstacles) {
      const topBox = {
        x: obstacle.x,
        y: 0,
        width: obstacle.width,
        height: obstacle.topHeight
      };
      
      const bottomBox = {
        x: obstacle.x,
        y: obstacle.bottomY,
        width: obstacle.width,
        height: obstacle.bottomHeight
      };
      
      if (this.checkCollision(copterBox, topBox) || this.checkCollision(copterBox, bottomBox)) {
        this.endGame();
        return;
      }
    }
  }
  
  checkCollision(box1, box2) {
    return box1.x < box2.x + box2.width &&
           box1.x + box1.width > box2.x &&
           box1.y < box2.y + box2.height &&
           box1.y + box1.height > box2.y;
  }
  
  draw() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.drawBackground();
    this.drawObstacles();
    this.drawCopter();
    
    if (!this.gameStarted) {
      this.drawStartMessage();
    }
  }
  
  drawBackground() {
    const gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
    gradient.addColorStop(0, '#87CEEB');
    gradient.addColorStop(1, '#98D8C8');
    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.ctx.fillStyle = '#228B22';
    this.ctx.fillRect(0, this.canvas.height - 20, this.canvas.width, 20);
    
    this.ctx.fillStyle = '#fff';
    this.ctx.font = 'bold 20px Arial';
    this.ctx.textAlign = 'center';
    for (let i = 0; i < 10; i++) {
      const x = (i * 100 + this.frameCount * 2) % (this.canvas.width + 100) - 50;
      const y = 50 + Math.sin((this.frameCount + i * 20) * 0.02) * 10;
      this.ctx.fillText('☁️', x, y);
    }
  }
  
  drawCopter() {
    const { x, y, width, height } = this.copter;
    const centerX = x + width / 2;
    const centerY = y + height / 2;
    
    this.ctx.save();
    this.ctx.translate(centerX, centerY);
    this.ctx.scale(-1, 1);
    this.ctx.rotate(this.copter.velocity * 0.005);
    this.ctx.translate(-centerX, -centerY);
    
    this.ctx.fillStyle = '#e74c3c';
    this.ctx.beginPath();
    this.ctx.ellipse(centerX, centerY, width / 2, height / 2, 0, 0, Math.PI * 2);
    this.ctx.fill();
    
    this.ctx.fillStyle = '#c0392b';
    this.ctx.beginPath();
    this.ctx.ellipse(centerX - 10, centerY + 5, 15, 10, 0, 0, Math.PI * 2);
    this.ctx.fill();
    
    this.ctx.fillStyle = '#3498db';
    this.ctx.beginPath();
    this.ctx.ellipse(centerX - 15, centerY - 3, 12, 8, 0, 0, Math.PI * 2);
    this.ctx.fill();
    
    this.ctx.fillStyle = '#2c3e50';
    this.ctx.fillRect(x + width, centerY - 3, 20, 6);
    
    this.ctx.fillStyle = '#95a5a6';
    this.ctx.beginPath();
    this.ctx.ellipse(centerX, centerY - 15, 5, 15, 0, 0, Math.PI * 2);
    this.ctx.fill();
    
    this.ctx.fillStyle = '#7f8c8d';
    this.ctx.beginPath();
    this.ctx.ellipse(centerX, centerY - 15, 3, 12, 0, 0, Math.PI * 2);
    this.ctx.fill();
    
    const bladeAngle = this.frameCount * 0.3;
    this.ctx.save();
    this.ctx.translate(centerX, centerY - 15);
    this.ctx.rotate(bladeAngle);
    
    this.ctx.fillStyle = '#34495e';
    this.ctx.fillRect(-35, -2, 70, 4);
    this.ctx.fillRect(-2, -35, 4, 70);
    
    this.ctx.restore();
    
    this.ctx.fillStyle = '#e74c3c';
    this.ctx.beginPath();
    this.ctx.moveTo(x + width + 20, centerY - 3);
    this.ctx.lineTo(x + width + 35, centerY - 3);
    this.ctx.lineTo(x + width + 35, centerY + 3);
    this.ctx.lineTo(x + width + 20, centerY + 3);
    this.ctx.closePath();
    this.ctx.fill();
    
    this.ctx.fillStyle = '#c0392b';
    this.ctx.beginPath();
    this.ctx.moveTo(x + width + 35, centerY - 3);
    this.ctx.lineTo(x + width + 40, centerY - 8);
    this.ctx.lineTo(x + width + 40, centerY + 8);
    this.ctx.lineTo(x + width + 35, centerY + 3);
    this.ctx.closePath();
    this.ctx.fill();
    
    this.ctx.restore();
  }
  
  drawObstacles() {
    for (const obstacle of this.obstacles) {
      this.ctx.fillStyle = '#2c3e50';
      this.ctx.fillRect(obstacle.x, 0, obstacle.width, obstacle.topHeight);
      
      this.ctx.fillStyle = '#34495e';
      this.ctx.fillRect(obstacle.x + 5, 0, obstacle.width - 10, obstacle.topHeight - 5);
      
      this.ctx.fillStyle = '#2c3e50';
      this.ctx.fillRect(obstacle.x, obstacle.bottomY, obstacle.width, obstacle.bottomHeight);
      
      this.ctx.fillStyle = '#34495e';
      this.ctx.fillRect(obstacle.x + 5, obstacle.bottomY + 5, obstacle.width - 10, obstacle.bottomHeight - 5);
      
      this.ctx.fillStyle = '#e74c3c';
      this.ctx.fillRect(obstacle.x, obstacle.topHeight - 10, obstacle.width, 10);
      this.ctx.fillRect(obstacle.x, obstacle.bottomY, obstacle.width, 10);
    }
  }
  
  drawStartMessage() {
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.ctx.fillStyle = '#fff';
    this.ctx.font = 'bold 36px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('🚁 Copter Game', this.canvas.width / 2, this.canvas.height / 2 - 40);
    
    this.ctx.font = '24px Arial';
    this.ctx.fillText('Click, Tap, or Press Space to Start', this.canvas.width / 2, this.canvas.height / 2 + 20);
    
    this.ctx.font = '18px Arial';
    this.ctx.fillText('Hold to fly up, release to fall down', this.canvas.width / 2, this.canvas.height / 2 + 60);
  }
  
  updateScoreDisplay() {
    document.getElementById('score').textContent = Math.floor(this.score / 10);
  }
  
  updateHighScoreDisplay() {
    document.getElementById('high-score').textContent = Math.floor(this.highScore / 10);
  }
  
  endGame() {
    this.gameOver = true;
    this.gameStarted = false;
    
    const currentScore = Math.floor(this.score / 10);
    
    if (currentScore > this.highScore) {
      this.highScore = currentScore;
      localStorage.setItem('copterHighScore', this.highScore);
      this.updateHighScoreDisplay();
    }
    
    document.getElementById('final-score').textContent = currentScore;
    document.getElementById('final-high-score').textContent = Math.floor(this.highScore / 10);
    document.getElementById('game-over-screen').style.display = 'block';
  }
  
  restart() {
    this.copter.y = 250;
    this.copter.velocity = 0;
    this.obstacles = [];
    this.score = 0;
    this.frameCount = 0;
    this.gameOver = false;
    this.gameStarted = false;
    this.isFlyingUp = false;
    
    this.applyDifficulty();
    this.updateScoreDisplay();
    
    document.getElementById('game-over-screen').style.display = 'none';
    
    this.draw();
  }
  
  gameLoop() {
    if (!this.gameStarted || this.gameOver) {
      return;
    }
    
    this.update();
    this.draw();
    
    requestAnimationFrame(() => this.gameLoop());
  }
}

const game = new CopterGame();
</script>
