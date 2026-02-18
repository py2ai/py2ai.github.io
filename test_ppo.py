"""
Test script for Proximal Policy Optimization (PPO)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple

class CartPoleEnvironment:
    """
    Simple CartPole-like Environment for PPO
    
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

class ActorNetwork(nn.Module):
    """
    Actor Network for PPO
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super(ActorNetwork, self).__init__()
        
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
    
    def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Get action and log probability
        
        Args:
            state: Current state
            
        Returns:
            (action, log_prob)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self.forward(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get log probability of action
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Log probability
        """
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.log_prob(action)

class CriticNetwork(nn.Module):
    """
    Critic Network for PPO
    
    Args:
        state_dim: Dimension of state space
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(self, state_dim: int, hidden_dims: list = [128, 128]):
        super(CriticNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_epsilon: Clipping parameter
        n_epochs: Number of optimization epochs
        batch_size: Batch size
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list = [128, 128],
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 n_epochs: int = 10,
                 batch_size: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Actor and critic networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.critic = CriticNetwork(state_dim, hidden_dims)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
    
    def generate_episode(self, env: CartPoleEnvironment, max_steps: int = 200) -> List[Tuple]:
        """
        Generate one episode
        
        Args:
            env: Environment
            max_steps: Maximum steps per episode
            
        Returns:
            List of (state, action, log_prob, reward) tuples
        """
        episode = []
        state = env.reset()
        
        for step in range(max_steps):
            # Get action and log probability
            action, log_prob = self.actor.get_action(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Store experience
            episode.append((state, action, log_prob, reward))
            
            # Update state
            state = next_state
            
            if done:
                break
        
        return episode
    
    def compute_returns_and_advantages(self, episode: List[Tuple]) -> Tuple[List[float], List[float]]:
        """
        Compute returns and advantages using GAE
        
        Args:
            episode: List of experiences
            
        Returns:
            (returns, advantages)
        """
        states = [e[0] for e in episode]
        rewards = [e[3] for e in episode]
        
        # Compute values
        values = []
        for state in states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = self.critic(state_tensor).item()
            values.append(value)
        
        # Compute GAE advantages
        advantages = []
        gae = 0
        
        for t in reversed(range(len(episode))):
            if t == len(episode) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        # Compute returns
        returns = [a + v for a, v in zip(advantages, values)]
        
        return returns, advantages
    
    def update_policy(self, episode: List[Tuple], returns: List[float], advantages: List[float]):
        """
        Update actor and critic using PPO
        
        Args:
            episode: List of experiences
            returns: List of returns
            advantages: List of advantages
        """
        # Convert to tensors
        states = torch.FloatTensor(np.array([e[0] for e in episode]))
        actions = torch.LongTensor([e[1] for e in episode])
        old_log_probs = torch.stack([e[2] for e in episode]).detach()
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize for multiple epochs
        for epoch in range(self.n_epochs):
            # Get new log probabilities
            new_log_probs = self.actor.get_log_prob(states, actions)
            
            # Compute probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update critic
            values = self.critic(states).squeeze()
            critic_loss = nn.functional.mse_loss(values, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
    
    def train_episode(self, env: CartPoleEnvironment, max_steps: int = 200) -> float:
        """
        Train for one episode
        
        Args:
            env: Environment
            max_steps: Maximum steps per episode
            
        Returns:
            Total reward for episode
        """
        # Generate episode
        episode = self.generate_episode(env, max_steps)
        
        # Compute total reward
        total_reward = sum([e[3] for e in episode])
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(episode)
        
        # Update policy
        self.update_policy(episode, returns, advantages)
        
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
                print(f"Episode {episode + 1}, Avg Reward (last 50): {avg_reward:.2f}")
        
        return rewards

# Test the code
if __name__ == "__main__":
    print("Testing Proximal Policy Optimization (PPO)...")
    print("=" * 50)
    
    # Create environment
    env = CartPoleEnvironment(state_dim=4, action_dim=2)
    
    # Create agent
    agent = PPOAgent(state_dim=4, action_dim=2)
    
    # Train agent
    print("\nTraining agent...")
    rewards = agent.train(env, n_episodes=300, max_steps=200, verbose=True)
    
    # Test agent
    print("\nTesting trained agent...")
    state = env.reset()
    total_reward = 0
    
    for step in range(50):
        action, _ = agent.actor.get_action(state)
        next_state, reward, done = env.step(action)
        
        total_reward += reward
        
        if done:
            print(f"Episode finished after {step + 1} steps")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print("\nPPO test completed successfully! ✓")
