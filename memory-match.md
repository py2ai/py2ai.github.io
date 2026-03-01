---
layout: page
title: Memory Match
permalink: /games/memory-match/
---

<div class="memory-container">
  <h1>🎴 Memory Match</h1>
  
  <div class="game-description">
    <p><strong>Objective:</strong> Find all matching pairs of cards in the fewest moves possible!</p>
    <p><strong>How to Play:</strong> Click or tap cards to flip them. Find matching pairs to clear them.</p>
    <p><strong>Rules:</strong> Only two cards can be flipped at once. Match all pairs to win!</p>
    <p><strong>Scoring:</strong> Fewer moves = better score. Track your best!</p>
  </div>
  
  <div class="game-wrapper">
    <div class="score-display">
      <span class="score-left">Moves: <strong id="current-moves">0</strong></span>
      <button id="sound-toggle" class="sound-toggle" title="Toggle Sound">🔊</button>
      <span class="score-right">Best: <strong id="best-moves">0</strong></span>
    </div>
    
    <div id="game-over-screen" class="game-over-screen" style="display: none;">
      <h2>You Won!</h2>
      <p>Total Moves: <strong id="final-moves">0</strong></p>
      <button id="restart-btn" class="restart-btn">Play Again</button>
      <button id="home-btn" class="home-btn">Back to Games</button>
    </div>
    
    <div id="game-board" class="game-board"></div>
  </div>
</div>

<style>
.memory-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
}

.memory-container h1 {
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
  width: 100%;
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

.game-board {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  background: #1a1a1a;
  padding: 15px;
  border: 4px solid #667eea;
  border-top: none;
  border-radius: 0 0 10px 10px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  width: 100%;
  box-sizing: border-box;
}

.card {
  aspect-ratio: 1;
  background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);
  border-radius: 10px;
  cursor: pointer;
  position: relative;
  transform-style: preserve-3d;
  transition: transform 0.3s ease;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.card:hover {
  transform: scale(1.05);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
}

.card.flipped {
  transform: rotateY(180deg);
}

.card.matched {
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.3s ease;
}

.card-front, .card-back {
  position: absolute;
  width: 100%;
  height: 100%;
  backface-visibility: hidden;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2.5em;
}

.card-front {
  background: linear-gradient(145deg, #ffffff 0%, #f0f0f0 100%);
  transform: rotateY(180deg);
  box-shadow: inset 0 2px 0 rgba(255, 255, 255, 0.8);
}

.card-back {
  background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);
  box-shadow: inset 0 2px 0 rgba(255, 255, 255, 0.3);
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

#final-moves {
  font-size: 1.5em;
  color: #667eea;
}

@media (max-width: 1024px) {
  .memory-container h1 {
    font-size: 2em;
  }
  
  .game-board {
    width: 100%;
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
class MemoryMatchGame {
  constructor() {
    this.gameBoard = document.getElementById('game-board');
    this.cards = [];
    this.flippedCards = [];
    this.matchedPairs = 0;
    this.moves = 0;
    this.bestMoves = 0;
    this.isLocked = false;
    this.soundEnabled = true;
    this.audioContext = null;
    
    this.emojis = ['🍎', '🍊', '🍋', '🍌', '🍍', '🍎', '🍏', '🍐'];
    
    this.init();
  }
  
  init() {
    this.loadBestMoves();
    this.bindEvents();
    this.startGame();
  }
  
  bindEvents() {
    const restartBtn = document.getElementById('restart-btn');
    const homeBtn = document.getElementById('home-btn');
    const soundToggle = document.getElementById('sound-toggle');
    
    if (!restartBtn || !homeBtn || !soundToggle) {
      console.error('Memory Match game buttons not found in DOM');
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
  }
  
  startGame() {
    this.resetGame();
    
    document.getElementById('game-over-screen').style.display = 'none';
  }
  
  resetGame() {
    this.cards = [];
    this.flippedCards = [];
    this.matchedPairs = 0;
    this.moves = 0;
    this.isLocked = false;
    
    this.updateMovesDisplay();
    this.createCards();
  }
  
  createCards() {
    this.gameBoard.innerHTML = '';
    
    const pairs = [...this.emojis, ...this.emojis];
    this.shuffleArray(pairs);
    
    pairs.forEach((emoji, index) => {
      const card = document.createElement('div');
      card.className = 'card';
      card.dataset.index = index;
      card.dataset.emoji = emoji;
      
      const front = document.createElement('div');
      front.className = 'card-front';
      front.textContent = emoji;
      
      const back = document.createElement('div');
      back.className = 'card-back';
      back.textContent = '?';
      
      card.appendChild(front);
      card.appendChild(back);
      
      card.addEventListener('click', (e) => {
        e.preventDefault();
        this.initAudioContext();
        this.flipCard(card);
      });
      
      card.addEventListener('touchstart', (e) => {
        e.preventDefault();
        this.initAudioContext();
        this.flipCard(card);
      });
      
      this.gameBoard.appendChild(card);
      this.cards.push(card);
    });
  }
  
  shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }
  
  flipCard(card) {
    if (this.isLocked) return;
    if (card.classList.contains('flipped')) return;
    if (card.classList.contains('matched')) return;
    
    this.playFlipSound();
    card.classList.add('flipped');
    this.flippedCards.push(card);
    
    if (this.flippedCards.length === 2) {
      this.moves++;
      this.updateMovesDisplay();
      this.checkForMatch();
    }
  }
  
  checkForMatch() {
    this.isLocked = true;
    
    const [card1, card2] = this.flippedCards;
    const isMatch = card1.dataset.emoji === card2.dataset.emoji;
    
    if (isMatch) {
      this.playMatchSound();
      setTimeout(() => {
        card1.classList.add('matched');
        card2.classList.add('matched');
        this.matchedPairs++;
        this.flippedCards = [];
        this.isLocked = false;
        
        if (this.matchedPairs === this.emojis.length) {
          this.gameWon();
        }
      }, 500);
    } else {
      this.playMismatchSound();
      setTimeout(() => {
        card1.classList.remove('flipped');
        card2.classList.remove('flipped');
        this.flippedCards = [];
        this.isLocked = false;
      }, 1000);
    }
  }
  
  gameWon() {
    this.playWinSound();
    
    if (this.moves < this.bestMoves || this.bestMoves === 0) {
      this.bestMoves = this.moves;
      this.saveBestMoves();
    }
    
    document.getElementById('final-moves').textContent = this.moves;
    document.getElementById('game-over-screen').style.display = 'block';
  }
  
  restartGame() {
    document.getElementById('game-over-screen').style.display = 'none';
    this.resetGame();
  }
  
  updateMovesDisplay() {
    document.getElementById('current-moves').textContent = this.moves;
  }
  
  saveBestMoves() {
    localStorage.setItem('memoryMatchBestMoves', this.bestMoves.toString());
  }
  
  loadBestMoves() {
    const savedBestMoves = localStorage.getItem('memoryMatchBestMoves');
    if (savedBestMoves) {
      this.bestMoves = parseInt(savedBestMoves);
      document.getElementById('best-moves').textContent = this.bestMoves;
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
  
  playFlipSound() {
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
      
      oscillator.frequency.value = 500;
      oscillator.type = 'sine';
      
      gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.1);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.1);
    } catch (error) {
      console.error('Error playing sound:', error);
    }
  }
  
  playMatchSound() {
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
      
      gainNode.gain.setValueAtTime(0.15, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.2);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.2);
    } catch (error) {
      console.error('Error playing sound:', error);
    }
  }
  
  playMismatchSound() {
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
      oscillator.type = 'sawtooth';
      
      gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.15);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.15);
    } catch (error) {
      console.error('Error playing sound:', error);
    }
  }
  
  playWinSound() {
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
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.3);
      
      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.3);
      
      setTimeout(() => {
        const osc2 = this.audioContext.createOscillator();
        const gain2 = this.audioContext.createGain();
        
        osc2.connect(gain2);
        gain2.connect(this.audioContext.destination);
        
        osc2.frequency.value = 800;
        osc2.type = 'square';
        
        gain2.gain.setValueAtTime(0.15, this.audioContext.currentTime);
        gain2.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.3);
        
        osc2.start(this.audioContext.currentTime);
        osc2.stop(this.audioContext.currentTime + 0.3);
      }, 350);
    } catch (error) {
      console.error('Error playing sound:', error);
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  try {
    const game = new MemoryMatchGame();
    console.log('Memory Match game initialized successfully');
  } catch (error) {
    console.error('Error initializing Memory Match game:', error);
  }
});
</script>
