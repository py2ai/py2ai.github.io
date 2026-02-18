"""
Test script for Actor-Critic Methods (A2C)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple

class CartPoleEnvironment:
    """
    Simple CartPole-like Environment for Actor-Critic
    
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
    Actor Network
    
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

class CriticNetwork(nn.Module):
    """
    Critic Network
    
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

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) Agent
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate
        gamma: Discount factor
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list = [128, 128],
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
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
    
    def compute_returns(self, episode: List[Tuple]) -> List[float]:
        """
        Compute discounted returns for episode
        
        Args:
            episode: List of experiences
            
        Returns:
            List of returns
        """
        returns = []
        G = 0
        
        for _, _, _, reward in reversed(episode):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def compute_advantages(self, episode: List[Tuple], returns: List[float]) -> List[float]:
        """
        Compute advantages using critic
        
        Args:
            episode: List of experiences
            returns: List of returns
            
        Returns:
            List of advantages
        """
        advantages = []
        
        for (state, _, _, _), G in zip(episode, returns):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = self.critic(state_tensor).item()
            advantage = G - value
            advantages.append(advantage)
        
        return advantages
    
    def update_policy(self, episode: List[Tuple], returns: List[float], advantages: List[float]):
        """
        Update actor and critic
        
        Args:
            episode: List of experiences
            returns: List of returns
            advantages: List of advantages
        """
        # Convert to tensors
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update actor
        actor_loss = []
        for (_, _, log_prob, _), advantage in zip(episode, advantages):
            actor_loss.append(-log_prob * advantage)
        
        actor_loss = torch.stack(actor_loss).sum()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update critic
        critic_loss = []
        for (state, _, _, _), G in zip(episode, returns):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = self.critic(state_tensor)
            critic_loss.append(nn.functional.mse_loss(value, torch.FloatTensor([G])))
        
        critic_loss = torch.stack(critic_loss).sum()
        
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
        returns = self.compute_returns(episode)
        advantages = self.compute_advantages(episode, returns)
        
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
    print("Testing Actor-Critic Methods (A2C)...")
    print("=" * 50)
    
    # Create environment
    env = CartPoleEnvironment(state_dim=4, action_dim=2)
    
    # Create agent
    agent = A2CAgent(state_dim=4, action_dim=2)
    
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
    print("\nActor-Critic test completed successfully! ✓")
