"""
Test script for Trading Bot with RL
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

class TechnicalIndicators:
    """
    Technical Indicators for Trading
    
    Args:
        data: DataFrame with OHLCV data
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
    
    def sma(self, period: int = 20) -> pd.Series:
        """
        Simple Moving Average
        
        Args:
            period: Period for SMA
            
        Returns:
            SMA series
        """
        return self.data['close'].rolling(window=period).mean()
    
    def ema(self, period: int = 20) -> pd.Series:
        """
        Exponential Moving Average
        
        Args:
            period: Period for EMA
            
        Returns:
            EMA series
        """
        return self.data['close'].ewm(span=period, adjust=False).mean()
    
    def rsi(self, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            period: Period for RSI
            
        Returns:
            RSI series
        """
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """
        Moving Average Convergence Divergence
        
        Args:
            fast: Fast period
            slow: Slow period
            signal: Signal period
            
        Returns:
            (macd, signal, histogram)
        """
        ema_fast = self.data['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> tuple:
        """
        Bollinger Bands
        
        Args:
            period: Period for bands
            std_dev: Standard deviation multiplier
            
        Returns:
            (upper_band, middle_band, lower_band)
        """
        middle_band = self.sma(period)
        std = self.data['close'].rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band
    
    def add_all_indicators(self) -> pd.DataFrame:
        """
        Add all technical indicators to data
        
        Returns:
            DataFrame with indicators
        """
        self.data['sma_20'] = self.sma(20)
        self.data['ema_20'] = self.ema(20)
        self.data['rsi_14'] = self.rsi(14)
        macd, signal, _ = self.macd()
        self.data['macd'] = macd
        self.data['macd_signal'] = signal
        upper, middle, lower = self.bollinger_bands()
        self.data['bb_upper'] = upper
        self.data['bb_middle'] = middle
        self.data['bb_lower'] = lower
        
        # Fill NaN values
        self.data = self.data.fillna(method='bfill').fillna(method='ffill')
        
        return self.data

class TradingEnvironment:
    """
    Trading Environment for RL
    
    Args:
        data: DataFrame with price data
        initial_balance: Initial cash balance
        transaction_cost: Transaction cost per trade
        window_size: Size of observation window
    """
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001, window_size: int = 50):
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        
        # Calculate indicators
        indicators = TechnicalIndicators(self.data)
        self.data = indicators.add_all_indicators()
        
        # Normalize data
        self._normalize_data()
        
        self.reset()
    
    def _normalize_data(self):
        """Normalize price data"""
        for col in ['close', 'sma_20', 'ema_20', 'bb_upper', 'bb_middle', 'bb_lower']:
            if col in self.data.columns:
                self.data[col] = (self.data[col] - self.data[col].mean()) / (self.data[col].std() + 1e-8)
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.history = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state"""
        # Get window of data
        window = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Features: close, sma, ema, rsi, macd, bb
        features = ['close', 'sma_20', 'ema_20', 'rsi_14', 'macd', 'bb_upper', 'bb_lower']
        state = window[features].values.flatten()
        
        # Add position info
        position = np.array([self.shares / (self.balance + self.shares * self.data.iloc[self.current_step]['close'])])
        state = np.concatenate([state, position])
        
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action in environment
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            
        Returns:
            (next_state, reward, done, info)
        """
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                shares_to_buy = (self.balance * 0.5) / current_price
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.shares += shares_to_buy
                    self.balance -= cost
        
        elif action == 2:  # Sell
            if self.shares > 0:
                shares_to_sell = self.shares * 0.5
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.shares -= shares_to_sell
                self.balance += revenue
        
        # Update net worth
        self.net_worth = self.balance + self.shares * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Get next state
        next_state = self._get_state()
        
        # Info
        info = {
            'net_worth': self.net_worth,
            'shares': self.shares,
            'balance': self.balance
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward"""
        # Reward based on profit
        profit = self.net_worth - self.initial_balance
        reward = profit / self.initial_balance
        
        # Penalty for drawdown
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        reward -= drawdown * 0.5
        
        return reward

class TradingDQN(nn.Module):
    """
    DQN for Trading
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super(TradingDQN, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)

class TradingAgent:
    """
    Trading Agent with DQN
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate
        gamma: Discount factor
        buffer_size: Replay buffer size
        batch_size: Training batch size
        tau: Target network update rate
        exploration_rate: Initial exploration rate
        exploration_decay: Exploration decay rate
        min_exploration: Minimum exploration rate
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list = [256, 256],
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 tau: float = 0.001,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 0.995,
                 min_exploration: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Networks
        self.q_network = TradingDQN(state_dim, action_dim, hidden_dims)
        self.target_network = TradingDQN(state_dim, action_dim, hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.buffer = []
        self.buffer_size = buffer_size
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            eval_mode: Whether in evaluation mode
            
        Returns:
            Selected action
        """
        if not eval_mode and np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def train_step(self) -> float:
        """
        Perform one training step
        
        Returns:
            Loss value
        """
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        indices = np.random.choice(len(self.buffer), self.batch_size)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.FloatTensor([e[4] for e in batch])
        
        # Compute Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network using soft update"""
        for target_param, param in zip(self.target_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data +
                                   (1 - self.tau) * target_param.data)
    
    def decay_exploration(self):
        """Decay exploration rate"""
        self.exploration_rate = max(self.min_exploration,
                                   self.exploration_rate * self.exploration_decay)
    
    def train_episode(self, env: TradingEnvironment, max_steps: int = 1000) -> Tuple[float, float]:
        """
        Train for one episode
        
        Args:
            env: Environment
            max_steps: Maximum steps per episode
            
        Returns:
            (total_reward, final_net_worth)
        """
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done)
            
            # Train
            loss = self.train_step()
            
            # Update target network
            self.update_target_network()
            
            # Update state
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        self.decay_exploration()
        return total_reward, info['net_worth']
    
    def train(self, env: TradingEnvironment, n_episodes: int = 1000,
              max_steps: int = 1000, verbose: bool = True):
        """
        Train agent
        
        Args:
            env: Environment
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
        """
        rewards = []
        net_worths = []
        
        for episode in range(n_episodes):
            reward, net_worth = self.train_episode(env, max_steps)
            rewards.append(reward)
            net_worths.append(net_worth)
            
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                avg_net_worth = np.mean(net_worths[-10:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.4f}, "
                      f"Avg Net Worth: ${avg_net_worth:.2f}, Epsilon: {self.exploration_rate:.3f}")
        
        return rewards, net_worths

# Test the code
if __name__ == "__main__":
    print("Testing Trading Bot with Reinforcement Learning...")
    print("=" * 50)
    
    # Generate synthetic price data
    np.random.seed(42)
    n_days = 500
    prices = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    data = pd.DataFrame({
        'close': prices,
        'open': prices + np.random.randn(n_days) * 0.1,
        'high': prices + np.abs(np.random.randn(n_days)) * 0.5,
        'low': prices - np.abs(np.random.randn(n_days)) * 0.5,
        'volume': np.random.randint(1000, 10000, n_days)
    })
    
    # Create environment
    env = TradingEnvironment(data, initial_balance=10000.0, window_size=50)
    
    # Create agent
    state_dim = env._get_state().shape[0]
    agent = TradingAgent(state_dim=state_dim, action_dim=3)
    
    # Train agent
    print("\nTraining agent...")
    rewards, net_worths = agent.train(env, n_episodes=50, max_steps=400, verbose=True)
    
    # Test agent
    print("\nTesting trained agent...")
    state = env.reset()
    total_reward = 0
    
    for step in range(50):
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        
        if done:
            print(f"Episode finished after {step + 1} steps")
            break
    
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final net worth: ${info['net_worth']:.2f}")
    print("\nTrading Bot test completed successfully! ✓")
