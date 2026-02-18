"""
Test script for Game AI with Reinforcement Learning
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple, List

class TicTacToeEnvironment:
    """
    Tic-Tac-Toe Environment
    
    Args:
        board_size: Size of the board (default 3x3)
    """
    def __init__(self, board_size: int = 3):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
    
    def reset(self) -> np.ndarray:
        """Reset the game"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state"""
        return self.board.copy().astype(np.float32)
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get list of legal moves"""
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    moves.append((i, j))
        return moves
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action in the game
        
        Args:
            action: (row, col) tuple
            
        Returns:
            (next_state, reward, done, info)
        """
        # Make move
        self.board[action] = self.current_player
        
        # Check if game is over
        winner = self.check_winner()
        done = winner is not None or len(self.get_legal_moves()) == 0
        
        # Calculate reward
        if done:
            if winner == self.current_player:
                reward = 1.0
            elif winner == 0:
                reward = 0.0
            else:
                reward = -1.0
        else:
            reward = 0.0
        
        # Switch player
        self.current_player *= -1
        
        # Info
        info = {
            'winner': winner,
            'current_player': self.current_player
        }
        
        return self.get_state(), reward, done, info
    
    def check_winner(self) -> int:
        """Check if there's a winner (1, -1, or 0 for draw, None if not done)"""
        # Check rows
        for i in range(self.board_size):
            if abs(sum(self.board[i, :])) == self.board_size:
                return int(np.sign(sum(self.board[i, :])))
        
        # Check columns
        for j in range(self.board_size):
            if abs(sum(self.board[:, j])) == self.board_size:
                return int(np.sign(sum(self.board[:, j])))
        
        # Check diagonals
        diag1 = sum([self.board[i, i] for i in range(self.board_size)])
        if abs(diag1) == self.board_size:
            return int(np.sign(diag1))
        
        diag2 = sum([self.board[i, self.board_size - 1 - i] for i in range(self.board_size)])
        if abs(diag2) == self.board_size:
            return int(np.sign(diag2))
        
        # Check for draw
        if len(self.get_legal_moves()) == 0:
            return 0
        
        return None
    
    def render(self):
        """Render the board"""
        print()
        for i in range(self.board_size):
            row = []
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    row.append('X')
                elif self.board[i, j] == -1:
                    row.append('O')
                else:
                    row.append('.')
            print(' '.join(row))
        print()

class GameDQN(nn.Module):
    """
    DQN for Game AI
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super(GameDQN, self).__init__()
        
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

class GameAgent:
    """
    Game AI Agent with DQN
    
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
        self.q_network = GameDQN(state_dim, action_dim, hidden_dims)
        self.target_network = GameDQN(state_dim, action_dim, hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.buffer = []
        self.buffer_size = buffer_size
    
    def select_action(self, state: np.ndarray, legal_moves: List[int],
                     eval_mode: bool = False) -> int:
        """
        Select action using epsilon-greedy policy with legal move masking
        
        Args:
            state: Current state
            legal_moves: List of legal move indices
            eval_mode: Whether in evaluation mode
            
        Returns:
            Selected action
        """
        if not eval_mode and np.random.random() < self.exploration_rate:
            return np.random.choice(legal_moves)
        
        with torch.no_grad():
            # Flatten state
            state_flat = state.flatten()
            state_tensor = torch.FloatTensor(state_flat).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            
            # Mask illegal moves
            mask = torch.ones(self.action_dim) * float('-inf')
            mask[legal_moves] = 0
            q_values = q_values + mask
            
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        # Flatten states before storing
        state_flat = state.flatten()
        next_state_flat = next_state.flatten()
        self.buffer.append((state_flat, action, reward, next_state_flat, done))
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
    
    def train_episode(self, env: TicTacToeEnvironment, max_steps: int = 100) -> Tuple[float, bool]:
        """
        Train for one episode
        
        Args:
            env: Environment
            max_steps: Maximum steps per episode
            
        Returns:
            (total_reward, won)
        """
        state = env.reset()
        total_reward = 0
        won = False
        
        for step in range(max_steps):
            # Get legal moves
            legal_moves = [i * env.board_size + j for i, j in env.get_legal_moves()]
            
            # Select action
            action = self.select_action(state, legal_moves)
            
            # Convert action index to coordinates
            row, col = divmod(action, env.board_size)
            
            # Take action
            next_state, reward, done, info = env.step((row, col))
            
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
                won = (info['winner'] == 1)
                break
        
        self.decay_exploration()
        return total_reward, won
    
    def train(self, env: TicTacToeEnvironment, n_episodes: int = 1000,
              max_steps: int = 100, verbose: bool = True):
        """
        Train agent
        
        Args:
            env: Environment
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
        """
        rewards = []
        wins = []
        
        for episode in range(n_episodes):
            reward, won = self.train_episode(env, max_steps)
            rewards.append(reward)
            wins.append(1 if won else 0)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                win_rate = np.mean(wins[-100:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, "
                      f"Win Rate: {win_rate:.2%}, Epsilon: {self.exploration_rate:.3f}")
        
        return rewards, wins

# Test the code
if __name__ == "__main__":
    print("Testing Game AI with Reinforcement Learning...")
    print("=" * 50)
    
    # Create environment
    env = TicTacToeEnvironment(board_size=3)
    
    # Create agent
    state_dim = env.get_state().flatten().shape[0]
    action_dim = env.board_size * env.board_size
    agent = GameAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Train agent
    print("\nTraining agent...")
    rewards, wins = agent.train(env, n_episodes=500, max_steps=100, verbose=True)
    
    # Test agent
    print("\nTesting trained agent...")
    state = env.reset()
    env.render()
    
    for step in range(10):
        legal_moves = [i * env.board_size + j for i, j in env.get_legal_moves()]
        action = agent.select_action(state, legal_moves, eval_mode=True)
        row, col = divmod(action, env.board_size)
        
        next_state, reward, done, info = env.step((row, col))
        
        print(f"Step {step + 1}: Move to ({row}, {col}), Reward {reward:.2f}")
        env.render()
        
        state = next_state
        
        if done:
            if info['winner'] == 1:
                print("X wins!")
            elif info['winner'] == -1:
                print("O wins!")
            else:
                print("It's a draw!")
            break
    
    print("\nGame AI test completed successfully! ✓")
