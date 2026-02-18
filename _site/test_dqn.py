"""
Test script for Deep Q-Networks (DQN)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, List

class CartPoleEnvironment:
    """
    Simple CartPole-like Environment for DQN
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = None
        self.steps = 0
        self.max_steps = 200
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        self.steps = 0
        return self.state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take action in environment
        
        Args:
            action: Action to take
            
        Returns:
            (next_state, reward, done)
        """
        # Simple dynamics
        self.state = self.state + np.random.randn(self.state_dim).astype(np.float32) * 0.1
        
        # Reward based on state
        reward = 1.0 if abs(self.state[0]) < 2.0 else -1.0
        
        # Check if done
        self.steps += 1
        done = self.steps >= self.max_steps or abs(self.state[0]) > 4.0
        
        return self.state, reward, done

class DQN(nn.Module):
    """
    Deep Q-Network
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super(DQN, self).__init__()
        
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

class ReplayBuffer:
    """
    Experience Replay Buffer
    
    Args:
        capacity: Maximum buffer size
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> list:
        """Sample random batch from buffer"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return buffer size"""
        return len(self.buffer)

class DQNAgent:
    """
    DQN Agent
    
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
                 hidden_dims: list = [128, 128],
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
        self.q_network = DQN(state_dim, action_dim, hidden_dims)
        self.target_network = DQN(state_dim, action_dim, hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            eval_mode: Whether in evaluation mode
            
        Returns:
            Selected action
        """
        if not eval_mode and random.random() < self.exploration_rate:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Perform one training step
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
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
    
    def train_episode(self, env: CartPoleEnvironment, max_steps: int = 200) -> float:
        """
        Train for one episode
        
        Args:
            env: Environment
            max_steps: Maximum steps per episode
            
        Returns:
            Total reward for episode
        """
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            
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
        return total_reward
    
    def train(self, env: CartPoleEnvironment, n_episodes: int = 500,
              max_steps: int = 200, verbose: bool = True):
        """
        Train agent
        
        Args:
            env: Environment
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
        """
        rewards = []
        
        for episode in range(n_episodes):
            reward = self.train_episode(env, max_steps)
            rewards.append(reward)
            
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(rewards[-50:])
                print(f"Episode {episode + 1}, Avg Reward (last 50): {avg_reward:.2f}, "
                      f"Epsilon: {self.exploration_rate:.3f}")
        
        return rewards

# Test the code
if __name__ == "__main__":
    print("Testing Deep Q-Networks (DQN)...")
    print("=" * 50)
    
    # Create environment
    env = CartPoleEnvironment(state_dim=4, action_dim=2)
    
    # Create agent
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    # Train agent
    print("\nTraining agent...")
    rewards = agent.train(env, n_episodes=300, max_steps=200, verbose=True)
    
    # Test agent
    print("\nTesting trained agent...")
    state = env.reset()
    total_reward = 0
    
    for step in range(50):
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, done = env.step(action)
        
        total_reward += reward
        
        if done:
            print(f"Episode finished after {step + 1} steps")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print("\nDQN test completed successfully! ✓")
