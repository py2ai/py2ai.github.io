---
layout: page
title: Games
permalink: /games/
---

<div class="games-container">
  <h1>Games</h1>
  <p>Welcome to our games collection! Have fun playing!</p>
  
  <div class="games-grid">
    <div class="game-card">
      <div class="game-icon">🐍</div>
      <h2>Snake Game</h2>
      <p>Classic snake game with high scores</p>
      <a href="/games/snake" class="btn">Play Now</a>
    </div>
    
    <div class="game-card">
      <div class="game-icon">🐦</div>
      <h2>Flappy Bird</h2>
      <p>Help the bird fly through pipes</p>
      <a href="/games/flappy-bird" class="btn">Play Now</a>
    </div>
    
    <div class="game-card">
      <div class="game-icon">🧱</div>
      <h2>Tetris</h2>
      <p>Classic block-stacking puzzle game</p>
      <a href="/games/tetris" class="btn">Play Now</a>
    </div>
    
    <div class="game-card">
      <div class="game-icon">🎴</div>
      <h2>Memory Match</h2>
      <p>Find matching pairs of cards</p>
      <a href="/games/memory-match" class="btn">Play Now</a>
    </div>
    
    <div class="game-card">
      <div class="game-icon">🏓</div>
      <h2>Pong</h2>
      <p>Classic arcade game with 10 levels</p>
      <a href="/games/pong" class="btn">Play Now</a>
    </div>
    
    <div class="game-card">
      <div class="game-icon">🔢</div>
      <h2>2048</h2>
      <p>Combine tiles to reach 2048!</p>
      <a href="/games/2048" class="btn">Play Now</a>
    </div>
    
    <div class="game-card">
      <div class="game-icon">⚓</div>
      <h2>Battleship</h2>
      <p>Sink all enemy ships to win!</p>
      <a href="/games/battleship" class="btn">Play Now</a>
    </div>
    
    <div class="game-card">
      <div class="game-icon">⚫⚪</div>
      <h2>GO Game</h2>
      <p>Classic strategy board game</p>
      <a href="/games/go" class="btn">Play Now</a>
    </div>
    
    <div class="game-card">
      <div class="game-icon">♟</div>
      <h2>Chess Game</h2>
      <p>Classic strategy board game</p>
      <a href="/games/chess" class="btn">Play Now</a>
    </div>
  </div>
</div>

<style>
.games-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
}

.games-container h1 {
  text-align: center;
  font-size: 2.5em;
  margin-bottom: 20px;
  color: #333;
}

.games-container p {
  text-align: center;
  font-size: 1.2em;
  color: #666;
  margin-bottom: 40px;
}

.games-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 30px;
  padding: 20px;
}

.game-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 20px;
  padding: 40px;
  text-align: center;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.game-card:hover {
  transform: translateY(-10px);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

.game-icon {
  font-size: 80px;
  margin-bottom: 20px;
}

.game-card h2 {
  font-size: 2em;
  color: white;
  margin-bottom: 15px;
}

.game-card p {
  color: rgba(255, 255, 255, 0.9);
  font-size: 1.1em;
  margin-bottom: 25px;
}

.btn {
  display: inline-block;
  padding: 12px 30px;
  background: white;
  color: #667eea;
  text-decoration: none;
  border-radius: 25px;
  font-weight: bold;
  font-size: 1.1em;
  transition: all 0.3s ease;
}

.btn:hover {
  background: #f0f0f0;
  transform: scale(1.05);
}
</style>
