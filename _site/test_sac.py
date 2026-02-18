"""
Test script for Soft Actor-Critic (SAC)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import List, Tuple

class PendulumEnvironment:
    """
    Simple Pendulum-like Environment for SAC (continuous action space)
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
    """
    def __init__(self, state_dim: int = 2, action_dim: int = 1):
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
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Take action in environment
        
        Args:
            action: Action to take (continuous)
            
        Returns:
            (next_state, reward, done)
        """
        # Simple dynamics
        self.state = self.state + np.random.randn(self.state_dim).astype(np.float32) * 0.1 + action * 0.1
        
        # Reward based on state (minimize angle)
        reward = -np.sum(self.state ** 2)
        
        # Check if done
        self.steps += 1
        done = self.steps >= self.max_steps
        
        return self.state, reward, done

class ActorNetwork(nn.Module):
    """
    Actor Network for SAC
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        action_scale: Scale for actions
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list = [256, 256],
                 action_scale: float = 1.0):
        super(ActorNetwork, self).__init__()
        
        self.action_dim = action_dim
        self.action_scale = action_scale
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim * 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            (mean, log_std)
        """
        output = self.network(x)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability
        
        Args:
            state: State tensor
            
        Returns:
            (action, log_prob)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Sample from normal distribution
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Squash action
        action = torch.tanh(action) * self.action_scale
        
        # Adjust log probability for squashing
        log_prob = log_prob - torch.log(1 - torch.tanh(action / self.action_scale) ** 2 + 1e-7).sum(dim=-1, keepdim=True)
        
        return action, log_prob

class CriticNetwork(nn.Module):
    """
    Critic Network for SAC
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super(CriticNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate
        gamma: Discount factor
        tau: Target network update rate
        alpha: Temperature parameter
        target_entropy: Target entropy for automatic tuning
        buffer_size: Replay buffer size
        batch_size: Training batch size
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list = [256, 256],
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 target_entropy: float = None,
                 buffer_size: int = 1000000,
                 batch_size: int = 256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.batch_size = batch_size
        
        # Actor and critic networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dims)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dims)
        self.target_critic1 = CriticNetwork(state_dim, action_dim, hidden_dims)
        self.target_critic2 = CriticNetwork(state_dim, action_dim, hidden_dims)
        
        # Initialize target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        # Automatic temperature tuning
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
        # Replay buffer
        self.buffer = []
        self.buffer_size = buffer_size
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Select action
        
        Args:
            state: Current state
            eval_mode: Whether in evaluation mode
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if eval_mode:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean).cpu().numpy()[0]
            else:
                action, _ = self.actor.sample(state_tensor)
                action = action.cpu().numpy()[0]
        
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def sample_batch(self) -> dict:
        """Sample random batch from buffer"""
        indices = np.random.choice(len(self.buffer), self.batch_size)
        batch = [self.buffer[i] for i in indices]
        
        return {
            'states': torch.FloatTensor(np.array([e[0] for e in batch])),
            'actions': torch.FloatTensor(np.array([e[1] for e in batch])),
            'rewards': torch.FloatTensor(np.array([e[2] for e in batch])).unsqueeze(1),
            'next_states': torch.FloatTensor(np.array([e[3] for e in batch])),
            'dones': torch.FloatTensor(np.array([e[4] for e in batch])).unsqueeze(1)
        }
    
    def update_target_networks(self):
        """Update target networks using soft update"""
        for target_param, param in zip(self.target_critic1.parameters(),
                                       self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data +
                                   (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(),
                                       self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data +
                                   (1 - self.tau) * target_param.data)
    
    def train_step(self) -> float:
        """
        Perform one training step
        
        Returns:
            Loss value
        """
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.sample_batch()
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        critic1_loss = nn.functional.mse_loss(q1, target_q)
        critic2_loss = nn.functional.mse_loss(q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        actions_pred, log_probs = self.actor.sample(states)
        q1_pred = self.critic1(states, actions_pred)
        q2_pred = self.critic2(states, actions_pred)
        q_pred = torch.min(q1_pred, q2_pred)
        
        actor_loss = (self.alpha * log_probs - q_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = (self.log_alpha * (-log_probs - self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # Update target networks
        self.update_target_networks()
        
        return actor_loss.item()
    
    def train_episode(self, env: PendulumEnvironment, max_steps: int = 200) -> float:
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
            
            # Update state
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        return total_reward
    
    def train(self, env: PendulumEnvironment, n_episodes: int = 500,
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
                      f"Alpha: {self.alpha.item():.3f}")
        
        return rewards

# Test the code
if __name__ == "__main__":
    print("Testing Soft Actor-Critic (SAC)...")
    print("=" * 50)
    
    # Create environment
    env = PendulumEnvironment(state_dim=2, action_dim=1)
    
    # Create agent
    agent = SACAgent(state_dim=2, action_dim=1)
    
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
    print("\nSAC test completed successfully! ✓")
