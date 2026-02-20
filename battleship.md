---
layout: page
title: Battleship Game
permalink: /games/battleship/
---

<div class="battleship-game-container">
  <h1>⚓ Battleship Game</h1>
  
  <div class="game-description">
    <p><strong>Objective:</strong> Sink all enemy ships before they sink yours!</p>
    <p><strong>How to Play:</strong> Click on the enemy grid to fire. Place your ships on your grid before starting.</p>
    <p><strong>Rules:</strong> Take turns firing at each other's grid. First to sink all opponent's ships wins!</p>
    <p><strong>Scoring:</strong> Track your hits and misses. Sink all 5 ships to win!</p>
  </div>
  
  <div class="game-wrapper">
    <div class="game-info">
      <div class="score-panel">
        <div class="score-item">
          <span class="score-label">Your Hits:</span>
          <span class="score-value" id="player-hits">0</span>
        </div>
        <div class="score-item">
          <span class="score-label">Enemy Hits:</span>
          <span class="score-value" id="enemy-hits">0</span>
        </div>
        <div class="score-item">
          <span class="score-label">Turn:</span>
          <span class="score-value" id="turn-indicator">Your Turn</span>
        </div>
      </div>
      
      <div class="ship-legend">
        <h3>Ships to Sink:</h3>
        <div class="ship-item">
          <span class="ship-icon ship-5"></span>
          <span>Carrier (5)</span>
        </div>
        <div class="ship-item">
          <span class="ship-icon ship-4"></span>
          <span>Battleship (4)</span>
        </div>
        <div class="ship-item">
          <span class="ship-icon ship-3"></span>
          <span>Cruiser (3)</span>
        </div>
        <div class="ship-item">
          <span class="ship-icon ship-2"></span>
          <span>Destroyer (2)</span>
        </div>
        <div class="ship-item">
          <span class="ship-icon ship-1"></span>
          <span>Submarine (1)</span>
        </div>
      </div>
    </div>
    
    <div id="setup-screen" class="setup-screen">
      <h2>Place Your Ships</h2>
      <p>Click on your grid to place ships. Click again to rotate.</p>
      <div class="current-ship">
        <span>Current Ship: </span>
        <span id="current-ship-name">Carrier (5)</span>
      </div>
      <button id="random-placement" class="setup-btn">Random Placement</button>
      <button id="start-game" class="setup-btn" disabled>Start Game</button>
    </div>
    
    <div id="game-over-screen" class="game-over-screen" style="display: none;">
      <h2 id="game-over-title">Game Over!</h2>
      <p id="game-over-message">Your Score: <strong id="final-score">0</strong></p>
      <button id="restart-btn" class="restart-btn">Play Again</button>
      <button id="home-btn" class="home-btn">Back to Games</button>
    </div>
    
    <div class="grids-container">
      <div class="grid-wrapper">
        <h3>Your Fleet</h3>
        <canvas id="player-grid" width="300" height="300"></canvas>
      </div>
      <div class="grid-wrapper">
        <h3>Enemy Fleet</h3>
        <canvas id="enemy-grid" width="300" height="300"></canvas>
      </div>
    </div>
  </div>
</div>

<style>
.battleship-game-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
}

.battleship-game-container h1 {
  text-align: center;
  font-size: 2.5em;
  margin-bottom: 30px;
  color: #333;
}

.game-description {
  max-width: 600px;
  margin: 0 auto 30px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
  max-width: 800px;
  margin: 0 auto 40px;
  position: relative;
}

.game-info {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 20px;
  gap: 20px;
}

.score-panel {
  flex: 1;
  background: #1a1a2e;
  padding: 15px 20px;
  border-radius: 10px;
  border: 3px solid #667eea;
  color: white;
}

.score-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 8px 0;
  font-size: 1.1em;
}

.score-label {
  font-weight: bold;
  color: #a0a0a0;
}

.score-value {
  font-weight: bold;
  color: #667eea;
}

.ship-legend {
  flex: 1;
  background: #1a1a2e;
  padding: 15px 20px;
  border-radius: 10px;
  border: 3px solid #667eea;
  color: white;
}

.ship-legend h3 {
  margin: 0 0 15px 0;
  font-size: 1.2em;
  color: #667eea;
  text-align: center;
}

.ship-item {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 8px 0;
}

.ship-icon {
  width: 30px;
  height: 10px;
  background: #32CD32;
  border-radius: 2px;
}

.ship-1 { width: 20px; }
.ship-2 { width: 30px; }
.ship-3 { width: 40px; }
.ship-4 { width: 50px; }
.ship-5 { width: 60px; }

.setup-screen {
  background: rgba(255, 255, 255, 0.95);
  padding: 30px;
  border-radius: 15px;
  text-align: center;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
  margin-bottom: 20px;
  border: 3px solid #667eea;
}

.setup-screen h2 {
  font-size: 1.8em;
  margin-bottom: 15px;
  color: #333;
}

.setup-screen p {
  font-size: 1.1em;
  color: #666;
  margin-bottom: 15px;
}

.current-ship {
  font-size: 1.2em;
  font-weight: bold;
  color: #667eea;
  margin-bottom: 20px;
}

.setup-btn {
  padding: 12px 30px;
  font-size: 1.1em;
  border: none;
  border-radius: 25px;
  cursor: pointer;
  font-weight: bold;
  transition: all 0.3s ease;
  margin: 5px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.setup-btn:hover:not(:disabled) {
  transform: scale(1.05);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.setup-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.grids-container {
  display: flex;
  justify-content: center;
  gap: 30px;
  flex-wrap: wrap;
}

.grid-wrapper {
  text-align: center;
}

.grid-wrapper h3 {
  margin-bottom: 15px;
  color: #333;
  font-size: 1.3em;
}

#player-grid, #enemy-grid {
  display: block;
  margin: 0 auto;
  background: #1a1a2e;
  border: 3px solid #667eea;
  border-radius: 10px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  cursor: pointer;
}

#enemy-grid {
  cursor: crosshair;
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
  border: 3px solid #667eea;
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

@media (max-width: 768px) {
  .game-info {
    flex-direction: column;
  }
  
  .grids-container {
    flex-direction: column;
    align-items: center;
  }
  
  .battleship-game-container h1 {
    font-size: 2em;
  }
  
  #player-grid, #enemy-grid {
    width: 100%;
    height: auto;
  }
}
</style>

<script>
class BattleshipGame {
  constructor() {
    this.playerCanvas = document.getElementById('player-grid');
    this.enemyCanvas = document.getElementById('enemy-grid');
    this.playerCtx = this.playerCanvas.getContext('2d');
    this.enemyCtx = this.enemyCanvas.getContext('2d');
    
    this.gridSize = 10;
    this.cellSize = 30;
    
    this.playerShips = [];
    this.enemyShips = [];
    this.playerHits = [];
    this.playerMisses = [];
    this.enemyHits = [];
    this.enemyMisses = [];
    
    this.shipsToPlace = [
      { name: 'Carrier', size: 5 },
      { name: 'Battleship', size: 4 },
      { name: 'Cruiser', size: 3 },
      { name: 'Destroyer', size: 2 },
      { name: 'Submarine', size: 1 }
    ];
    
    this.currentShipIndex = 0;
    this.isHorizontal = true;
    this.gamePhase = 'setup';
    this.isPlayerTurn = true;
    this.playerHitsCount = 0;
    this.enemyHitsCount = 0;
    this.soundEnabled = true;
    this.audioContext = null;
    
    this.init();
  }
  
  init() {
    this.bindEvents();
    this.drawGrids();
    this.updateCurrentShipDisplay();
  }
  
  bindEvents() {
    const restartBtn = document.getElementById('restart-btn');
    const homeBtn = document.getElementById('home-btn');
    const randomPlacementBtn = document.getElementById('random-placement');
    const startGameBtn = document.getElementById('start-game');
    
    if (restartBtn) {
      restartBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.initAudioContext();
        this.restartGame();
      });
    }
    
    if (homeBtn) {
      homeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        window.location.href = '/games';
      });
    }
    
    if (randomPlacementBtn) {
      randomPlacementBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.initAudioContext();
        this.randomPlacement();
      });
    }
    
    if (startGameBtn) {
      startGameBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.initAudioContext();
        this.startGame();
      });
    }
    
    this.playerCanvas.addEventListener('click', (e) => {
      this.initAudioContext();
      this.handlePlayerGridClick(e);
    });
    
    this.enemyCanvas.addEventListener('click', (e) => {
      this.initAudioContext();
      this.handleEnemyGridClick(e);
    });
    
    document.addEventListener('keydown', (e) => {
      if (e.key === 'r' || e.key === 'R') {
        this.isHorizontal = !this.isHorizontal;
      }
    });
  }
  
  drawGrids() {
    this.drawGrid(this.playerCtx, this.playerShips, this.enemyHits, this.enemyMisses, true);
    this.drawGrid(this.enemyCtx, this.enemyShips, this.playerHits, this.playerMisses, false);
  }
  
  drawGrid(ctx, ships, hits, misses, showShips) {
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    for (let i = 0; i < this.gridSize; i++) {
      for (let j = 0; j < this.gridSize; j++) {
        const x = i * this.cellSize;
        const y = j * this.cellSize;
        
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, this.cellSize, this.cellSize);
        
        const isHit = hits.some(h => h.x === i && h.y === j);
        const isMiss = misses.some(m => m.x === i && m.y === j);
        const isShip = ships.some(s => s.positions.some(p => p.x === i && p.y === j));
        
        if (isHit) {
          ctx.fillStyle = '#ff0000';
          ctx.beginPath();
          ctx.arc(x + this.cellSize / 2, y + this.cellSize / 2, this.cellSize / 3, 0, Math.PI * 2);
          ctx.fill();
        } else if (isMiss) {
          ctx.fillStyle = '#ffffff';
          ctx.beginPath();
          ctx.arc(x + this.cellSize / 2, y + this.cellSize / 2, this.cellSize / 4, 0, Math.PI * 2);
          ctx.fill();
        } else if (showShips && isShip) {
          ctx.fillStyle = '#32CD32';
          ctx.fillRect(x + 2, y + 2, this.cellSize - 4, this.cellSize - 4);
        }
      }
    }
  }
  
  handlePlayerGridClick(e) {
    if (this.gamePhase !== 'setup') return;
    
    const rect = this.playerCanvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / this.cellSize);
    const y = Math.floor((e.clientY - rect.top) / this.cellSize);
    
    if (x < 0 || x >= this.gridSize || y < 0 || y >= this.gridSize) return;
    
    if (this.canPlaceShip(x, y, this.shipsToPlace[this.currentShipIndex].size, this.isHorizontal)) {
      this.placeShip(x, y, this.shipsToPlace[this.currentShipIndex].size, this.isHorizontal, true);
      this.currentShipIndex++;
      
      if (this.currentShipIndex >= this.shipsToPlace.length) {
        document.getElementById('start-game').disabled = false;
        document.getElementById('current-ship-name').textContent = 'All ships placed!';
      } else {
        this.updateCurrentShipDisplay();
      }
      
      this.drawGrids();
    }
  }
  
  handleEnemyGridClick(e) {
    if (this.gamePhase !== 'playing' || !this.isPlayerTurn) return;
    
    const rect = this.enemyCanvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / this.cellSize);
    const y = Math.floor((e.clientY - rect.top) / this.cellSize);
    
    if (x < 0 || x >= this.gridSize || y < 0 || y >= this.gridSize) return;
    
    const alreadyFired = this.playerHits.some(h => h.x === x && h.y === y) || 
                        this.playerMisses.some(m => m.x === x && m.y === y);
    
    if (alreadyFired) return;
    
    const isHit = this.enemyShips.some(s => s.positions.some(p => p.x === x && p.y === y));
    
    if (isHit) {
      this.playerHits.push({ x, y });
      this.playerHitsCount++;
      this.playHitSound();
      this.checkSunkShip(this.enemyShips, x, y);
    } else {
      this.playerMisses.push({ x, y });
      this.playMissSound();
    }
    
    this.updateScoreDisplay();
    this.drawGrids();
    
    if (this.checkWinCondition()) {
      return;
    }
    
    this.isPlayerTurn = false;
    document.getElementById('turn-indicator').textContent = 'Enemy Turn';
    
    setTimeout(() => this.enemyTurn(), 1000);
  }
  
  canPlaceShip(x, y, size, horizontal) {
    for (let i = 0; i < size; i++) {
      const checkX = horizontal ? x + i : x;
      const checkY = horizontal ? y : y + i;
      
      if (checkX < 0 || checkX >= this.gridSize || checkY < 0 || checkY >= this.gridSize) {
        return false;
      }
      
      if (this.playerShips.some(s => s.positions.some(p => p.x === checkX && p.y === checkY))) {
        return false;
      }
    }
    
    return true;
  }
  
  placeShip(x, y, size, horizontal, isPlayer) {
    const positions = [];
    
    for (let i = 0; i < size; i++) {
      positions.push({
        x: horizontal ? x + i : x,
        y: horizontal ? y : y + i
      });
    }
    
    const ship = {
      name: this.shipsToPlace[this.currentShipIndex].name,
      size: size,
      positions: positions,
      hits: 0
    };
    
    if (isPlayer) {
      this.playerShips.push(ship);
    } else {
      this.enemyShips.push(ship);
    }
  }
  
  randomPlacement() {
    this.playerShips = [];
    this.currentShipIndex = 0;
    
    for (const ship of this.shipsToPlace) {
      let placed = false;
      
      while (!placed) {
        const horizontal = Math.random() < 0.5;
        const x = Math.floor(Math.random() * this.gridSize);
        const y = Math.floor(Math.random() * this.gridSize);
        
        if (this.canPlaceShip(x, y, ship.size, horizontal)) {
          this.placeShip(x, y, ship.size, horizontal, true);
          placed = true;
        }
      }
      
      this.currentShipIndex++;
    }
    
    document.getElementById('start-game').disabled = false;
    document.getElementById('current-ship-name').textContent = 'All ships placed!';
    this.drawGrids();
  }
  
  startGame() {
    this.placeEnemyShips();
    this.gamePhase = 'playing';
    document.getElementById('setup-screen').style.display = 'none';
    document.getElementById('game-over-screen').style.display = 'none';
  }
  
  placeEnemyShips() {
    this.enemyShips = [];
    
    for (const ship of this.shipsToPlace) {
      let placed = false;
      
      while (!placed) {
        const horizontal = Math.random() < 0.5;
        const x = Math.floor(Math.random() * this.gridSize);
        const y = Math.floor(Math.random() * this.gridSize);
        
        if (this.canPlaceEnemyShip(x, y, ship.size, horizontal)) {
          this.placeEnemyShip(x, y, ship.size, horizontal);
          placed = true;
        }
      }
    }
  }
  
  canPlaceEnemyShip(x, y, size, horizontal) {
    for (let i = 0; i < size; i++) {
      const checkX = horizontal ? x + i : x;
      const checkY = horizontal ? y : y + i;
      
      if (checkX < 0 || checkX >= this.gridSize || checkY < 0 || checkY >= this.gridSize) {
        return false;
      }
      
      if (this.enemyShips.some(s => s.positions.some(p => p.x === checkX && p.y === checkY))) {
        return false;
      }
    }
    
    return true;
  }
  
  placeEnemyShip(x, y, size, horizontal) {
    const positions = [];
    
    for (let i = 0; i < size; i++) {
      positions.push({
        x: horizontal ? x + i : x,
        y: horizontal ? y : y + i
      });
    }
    
    const ship = {
      name: this.shipsToPlace.find(s => s.size === size).name,
      size: size,
      positions: positions,
      hits: 0
    };
    
    this.enemyShips.push(ship);
  }
  
  enemyTurn() {
    if (this.gamePhase !== 'playing') return;
    
    let x, y;
    let validShot = false;
    
    while (!validShot) {
      x = Math.floor(Math.random() * this.gridSize);
      y = Math.floor(Math.random() * this.gridSize);
      
      const alreadyFired = this.enemyHits.some(h => h.x === x && h.y === y) || 
                          this.enemyMisses.some(m => m.x === x && m.y === y);
      
      if (!alreadyFired) {
        validShot = true;
      }
    }
    
    const isHit = this.playerShips.some(s => s.positions.some(p => p.x === x && p.y === y));
    
    if (isHit) {
      this.enemyHits.push({ x, y });
      this.enemyHitsCount++;
      this.checkSunkShip(this.playerShips, x, y);
    } else {
      this.enemyMisses.push({ x, y });
    }
    
    this.updateScoreDisplay();
    this.drawGrids();
    
    if (this.checkWinCondition()) {
      return;
    }
    
    this.isPlayerTurn = true;
    document.getElementById('turn-indicator').textContent = 'Your Turn';
  }
  
  checkSunkShip(ships, x, y) {
    for (const ship of ships) {
      if (ship.positions.some(p => p.x === x && p.y === y)) {
        ship.hits++;
        
        if (ship.hits === ship.size) {
          this.playSunkSound();
        }
        break;
      }
    }
  }
  
  checkWinCondition() {
    const allEnemyShipsSunk = this.enemyShips.every(s => s.hits === s.size);
    const allPlayerShipsSunk = this.playerShips.every(s => s.hits === s.size);
    
    if (allEnemyShipsSunk) {
      this.gameOver(true);
      return true;
    }
    
    if (allPlayerShipsSunk) {
      this.gameOver(false);
      return true;
    }
    
    return false;
  }
  
  gameOver(playerWon) {
    this.gamePhase = 'gameover';
    
    const gameOverScreen = document.getElementById('game-over-screen');
    const gameOverTitle = document.getElementById('game-over-title');
    const gameOverMessage = document.getElementById('game-over-message');
    const finalScore = document.getElementById('final-score');
    
    if (playerWon) {
      gameOverTitle.textContent = '🎉 Victory!';
      gameOverTitle.style.color = '#32CD32';
      gameOverMessage.innerHTML = 'You sank all enemy ships!<br>Score: <strong id="final-score">' + this.playerHitsCount + '</strong>';
    } else {
      gameOverTitle.textContent = '💥 Defeat!';
      gameOverTitle.style.color = '#ff0000';
      gameOverMessage.innerHTML = 'Your fleet has been destroyed!<br>Enemy Score: <strong id="final-score">' + this.enemyHitsCount + '</strong>';
    }
    
    gameOverScreen.style.display = 'block';
  }
  
  updateScoreDisplay() {
    document.getElementById('player-hits').textContent = this.playerHitsCount;
    document.getElementById('enemy-hits').textContent = this.enemyHitsCount;
  }
  
  updateCurrentShipDisplay() {
    if (this.currentShipIndex < this.shipsToPlace.length) {
      const ship = this.shipsToPlace[this.currentShipIndex];
      document.getElementById('current-ship-name').textContent = `${ship.name} (${ship.size})`;
    }
  }
  
  restartGame() {
    this.playerShips = [];
    this.enemyShips = [];
    this.playerHits = [];
    this.playerMisses = [];
    this.enemyHits = [];
    this.enemyMisses = [];
    this.currentShipIndex = 0;
    this.isHorizontal = true;
    this.gamePhase = 'setup';
    this.isPlayerTurn = true;
    this.playerHitsCount = 0;
    this.enemyHitsCount = 0;
    
    document.getElementById('setup-screen').style.display = 'block';
    document.getElementById('game-over-screen').style.display = 'none';
    document.getElementById('start-game').disabled = true;
    document.getElementById('turn-indicator').textContent = 'Your Turn';
    
    this.updateCurrentShipDisplay();
    this.updateScoreDisplay();
    this.drawGrids();
  }
  
  initAudioContext() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
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
      
      gainNode.gain.setValueAtTime(0.15, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.15);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.15);
    } catch (error) {
      console.error('Error playing hit sound:', error);
    }
  }
  
  playMissSound() {
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
      oscillator.type = 'sine';
      
      gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.1);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.1);
    } catch (error) {
      console.error('Error playing miss sound:', error);
    }
  }
  
  playSunkSound() {
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
      
      oscillator.frequency.setValueAtTime(600, this.audioContext.currentTime);
      oscillator.frequency.exponentialRampToValueAtTime(200, this.audioContext.currentTime + 0.3);
      oscillator.type = 'sawtooth';
      
      gainNode.gain.setValueAtTime(0.2, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.3);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.3);
    } catch (error) {
      console.error('Error playing sunk sound:', error);
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  try {
    const game = new BattleshipGame();
    console.log('Battleship game initialized successfully');
  } catch (error) {
    console.error('Error initializing Battleship game:', error);
  }
});
</script>
