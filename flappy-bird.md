---
layout: page
title: Flappy Bird
permalink: /games/flappy-bird/
---

<div class="flappy-bird-container">
  <h1>🐦 Flappy Bird</h1>
  
  <div class="game-description">
    <p><strong>Objective:</strong> Help the bird fly through the gaps between pipes and survive as long as possible!</p>
    <p><strong>How to Play:</strong> Click or tap anywhere on the canvas, or press Space/Up Arrow to flap wings and fly up. The bird falls due to gravity when not flapping.</p>
    <p><strong>Rules:</strong> Avoid hitting the pipes (top and bottom) and the ground/ceiling. Each pipe passed earns you 1 point.</p>
    <p><strong>Scoring:</strong> +1 point for each pipe passed. Try to beat your high score!</p>
  </div>
  
  <div class="game-wrapper">
    <div class="score-display">
      <button id="sound-toggle" class="sound-toggle" title="Toggle Sound">🔊</button>
    </div>
    
    <div id="game-over-screen" class="game-over-screen" style="display: none;">
      <h2>Game Over!</h2>
      <p>Your Score: <strong id="final-score">0</strong></p>
      <button id="restart-btn" class="restart-btn">Play Again</button>
      <button id="home-btn" class="home-btn">Back to Games</button>
    </div>
    
    <canvas id="game-canvas" width="400" height="500"></canvas>
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

.score-display {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 12px 20px;
  border-radius: 10px 10px 0 0;
  border: 4px solid #667eea;
  border-bottom: none;
  color: white;
  font-size: 1.1em;
  font-weight: bold;
  width: 400px;
  margin: 0 auto;
  max-width: 100%;
  box-sizing: border-box;
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
  background: linear-gradient(180deg, #87CEEB 0%, #E0F6FF 100%);
  border: 4px solid #667eea;
  border-top: none;
  border-radius: 0 0 10px 10px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
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

@media (max-width: 1024px) {
  .flappy-bird-container h1 {
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
class FlappyBirdGame {
  constructor() {
    this.canvas = document.getElementById('game-canvas');
    this.ctx = this.canvas.getContext('2d');
    
    this.bird = {
      x: 80,
      y: 150,
      width: 35,
      height: 25,
      velocity: 0,
      gravity: 0.175,
      jumpStrength: -3.5,
      wingAngle: 0
    };
    
    this.pipes = [];
    this.pipeWidth = 60;
    this.pipeGap = 160;
    this.pipeSpeed = 2.5;
    this.pipeSpawnRate = 180;
    this.frameCount = 0;
    
    this.score = 0;
    this.highScore = 0;
    this.isGameRunning = false;
    this.gameLoop = null;
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
      console.error('Flappy Bird game buttons not found in DOM');
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
    
    this.canvas.addEventListener('click', (e) => {
      e.preventDefault();
      this.initAudioContext();
      this.flap();
    });
    
    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.initAudioContext();
      this.flap();
    }, { passive: false });
  }
  
  handleKeyPress(e) {
    if (e.code === 'Space' || e.code === 'ArrowUp') {
      e.preventDefault();
      this.flap();
    }
  }
  
  startGame() {
    this.resetGame();
    
    document.getElementById('game-over-screen').style.display = 'none';
    
    this.isGameRunning = true;
    this.draw();
    this.gameLoop = setInterval(() => this.update(), 20);
  }
  
  resetGame() {
    this.bird = {
      x: 80,
      y: 150,
      width: 35,
      height: 25,
      velocity: 0,
      gravity: 0.175,
      jumpStrength: -3.5,
      wingAngle: 0
    };
    
    this.pipes = [];
    this.score = 0;
    this.frameCount = 0;
    this.updateScoreDisplay();
  }
  
  update() {
    if (!this.isGameRunning) return;
    
    this.frameCount++;
    
    this.bird.velocity += this.bird.gravity;
    this.bird.y += this.bird.velocity;
    this.bird.wingAngle += 0.3;
    
    if (this.frameCount % this.pipeSpawnRate === 0) {
      this.spawnPipe();
    }
    
    this.pipes.forEach(pipe => {
      pipe.x -= this.pipeSpeed;
      
      if (!pipe.passed && pipe.x + this.pipeWidth < this.bird.x) {
        pipe.passed = true;
        this.score++;
        this.playScoreSound();
        this.updateScoreDisplay();
      }
    });
    
    this.pipes = this.pipes.filter(pipe => pipe.x + this.pipeWidth > 0);
    
    if (this.checkCollision()) {
      this.gameOver();
      return;
    }
    
    this.draw();
  }
  
  spawnPipe() {
    const minHeight = 50;
    const maxHeight = this.canvas.height - this.pipeGap - minHeight - 50;
    const topHeight = Math.floor(Math.random() * (maxHeight - minHeight + 1)) + minHeight;
    
    this.pipes.push({
      x: this.canvas.width,
      topHeight: topHeight,
      bottomY: topHeight + this.pipeGap,
      passed: false
    });
  }
  
  checkCollision() {
    if (this.bird.y < 0 || this.bird.y + this.bird.height > this.canvas.height) {
      return true;
    }
    
    for (let pipe of this.pipes) {
      if (this.bird.x + this.bird.width > pipe.x && this.bird.x < pipe.x + this.pipeWidth) {
        if (this.bird.y < pipe.topHeight || this.bird.y + this.bird.height > pipe.bottomY) {
          return true;
        }
      }
    }
    
    return false;
  }
  
  flap() {
    if (!this.isGameRunning) return;
    
    this.bird.velocity = this.bird.jumpStrength;
  }
  
  draw() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.drawBackground();
    this.drawPipes();
    this.drawBird();
    this.drawScore();
  }
  
  drawScore() {
    this.ctx.font = 'bold 24px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'top';
    
    this.ctx.fillStyle = 'white';
    this.ctx.strokeStyle = 'black';
    this.ctx.lineWidth = 3;
    
    this.ctx.strokeText(this.score.toString(), this.canvas.width / 2, 15);
    this.ctx.fillText(this.score.toString(), this.canvas.width / 2, 15);
    
    this.ctx.font = 'bold 16px Arial';
    this.ctx.textAlign = 'left';
    this.ctx.strokeText('Top Score: ' + this.highScore.toString(), 10, 15);
    this.ctx.fillText('Top Score: ' + this.highScore.toString(), 10, 15);
  }
  
  drawBackground() {
    const gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
    gradient.addColorStop(0, '#87CEEB');
    gradient.addColorStop(1, '#E0F6FF');
    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.ctx.fillStyle = '#90EE90';
    this.ctx.fillRect(0, this.canvas.height - 50, this.canvas.width, 50);
  }
  
  drawPipes() {
    this.pipes.forEach(pipe => {
      const pipeGradient = this.ctx.createLinearGradient(pipe.x, 0, pipe.x + this.pipeWidth, 0);
      pipeGradient.addColorStop(0, '#228B22');
      pipeGradient.addColorStop(0.5, '#32CD32');
      pipeGradient.addColorStop(1, '#228B22');
      
      this.ctx.fillStyle = pipeGradient;
      
      this.ctx.fillRect(pipe.x, 0, this.pipeWidth, pipe.topHeight);
      this.ctx.fillRect(pipe.x, pipe.bottomY, this.pipeWidth, this.canvas.height - pipe.bottomY);
      
      this.ctx.fillStyle = '#1a5c1a';
      this.ctx.fillRect(pipe.x - 5, pipe.topHeight - 30, this.pipeWidth + 10, 30);
      this.ctx.fillRect(pipe.x - 5, pipe.bottomY, this.pipeWidth + 10, 30);
    });
  }
  
  drawBird() {
    const x = this.bird.x;
    const y = this.bird.y;
    const width = this.bird.width;
    const height = this.bird.height;
    
    this.ctx.save();
    this.ctx.translate(x + width / 2, y + height / 2);
    
    const rotation = Math.min(Math.max(this.bird.velocity * 0.05, -0.5), 0.5);
    this.ctx.rotate(rotation);
    
    this.ctx.fillStyle = '#FFD700';
    this.ctx.beginPath();
    this.ctx.ellipse(0, 0, width / 2, height / 2, 0, 0, Math.PI * 2);
    this.ctx.fill();
    
    this.ctx.fillStyle = '#FFA500';
    this.ctx.beginPath();
    this.ctx.ellipse(5, 2, width / 2.5, height / 2.5, 0, 0, Math.PI * 2);
    this.ctx.fill();
    
    this.ctx.fillStyle = 'white';
    this.ctx.beginPath();
    this.ctx.arc(8, -3, 5, 0, Math.PI * 2);
    this.ctx.fill();
    
    this.ctx.fillStyle = 'black';
    this.ctx.beginPath();
    this.ctx.arc(9, -3, 2.5, 0, Math.PI * 2);
    this.ctx.fill();
    
    this.ctx.fillStyle = '#FF6B6B';
    this.ctx.beginPath();
    this.ctx.moveTo(width / 2, 0);
    this.ctx.lineTo(width / 2 + 10, 3);
    this.ctx.lineTo(width / 2, 6);
    this.ctx.closePath();
    this.ctx.fill();
    
    const wingOffset = Math.sin(this.bird.wingAngle) * 5;
    this.ctx.fillStyle = '#32CD32';
    this.ctx.beginPath();
    this.ctx.ellipse(-5, 5 + wingOffset, 10, 6, -0.3, 0, Math.PI * 2);
    this.ctx.fill();
    
    this.ctx.fillStyle = '#228B22';
    this.ctx.beginPath();
    this.ctx.moveTo(-width / 2, 5);
    this.ctx.lineTo(-width / 2 - 12, 8);
    this.ctx.lineTo(-width / 2 - 10, 12);
    this.ctx.lineTo(-width / 2 - 8, 8);
    this.ctx.closePath();
    this.ctx.fill();
    
    this.ctx.restore();
  }
  
  updateScoreDisplay() {
    if (this.score > this.highScore) {
      this.highScore = this.score;
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
      
      gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.15);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.15);
    } catch (error) {
      console.error('Error playing sound:', error);
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
