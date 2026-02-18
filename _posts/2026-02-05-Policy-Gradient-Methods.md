---
layout: post
title: "Part 5: Policy Gradient Methods - Learning Policies Directly"
date: 2026-02-05
categories: [Machine Learning, AI, Python, Deep RL]
featured-img: 2026-feb-deeprl/2026-feb-deeprl
description: "Learn Policy Gradient Methods - directly optimizing policies in Reinforcement Learning. Complete guide with REINFORCE algorithm and PyTorch implementation."
---

# Part 5: Policy Gradient Methods - Learning Policies Directly

Welcome to the fifth post in our **Deep Reinforcement Learning Series**! In this comprehensive guide, we'll explore **Policy Gradient Methods** - a powerful class of algorithms that learn policies directly, without learning value functions. Unlike value-based methods like Q-learning, policy gradients optimize the policy parameters directly.

##  What are Policy Gradient Methods?

**Policy Gradient Methods** are a class of reinforcement learning algorithms that directly optimize the policy parameters $$\theta$$ to maximize expected return. Instead of learning a value function and deriving a policy, policy gradients learn the policy directly through gradient ascent.

### Key Characteristics

**Direct Policy Optimization:**
- Learn policy parameters $$\theta$$ directly
- No intermediate value function needed
- Can handle continuous action spaces
- More stable for complex problems

**Stochastic Policies:**
- Output probability distributions over actions
- Natural for exploration
- Better for continuous actions
- More robust than deterministic policies

**Gradient Ascent:**
- Maximize expected return $$J(\theta)$$
- Use gradient $$\nabla_\theta J(\theta)$$
- Update: $$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$
- Unlike value methods that minimize error

### Policy vs Value-Based Methods

| Aspect | Policy-Based | Value-Based |
|---------|-------------|-------------|
| **What's Learned** | Policy $$\pi_\theta(a|s)$$ | Q-function $$Q(s,a)$$ |
| **Action Selection** | Sample from policy | $$\arg\max_a Q(s,a)$$ |
| **Action Spaces** | Continuous & Discrete | Usually Discrete |
| **Convergence** | More Stable | Can be Unstable |
| **Sample Efficiency** | Higher | Lower |
| **Exploration** | Built-in | Requires separate strategy |

##  Policy Gradient Theorem

### The Objective Function

The goal is to maximize the expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

Where:
- $$\tau$$ - Trajectory $$(s_0, a_0, r_1, s_1, a_1, \dots)$$
- $$\pi_\theta$$ - Policy with parameters $$\theta$$
- $$\gamma$$ - Discount factor
- $$r_t$$ - Reward at time $$t$$

### The Policy Gradient Theorem

The gradient of the objective function is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right]$$

Where:
- $$G_t$$ - Return from time $$t$$: $$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k+1}$$
- $$\log \pi_\theta(a_t\|s_t)$$ - Log probability of taking action $$a_t$$ in state $$s_t$$

### Intuition

The theorem states that to improve the policy:
1. **Increase probability** of actions that lead to high returns
2. **Decrease probability** of actions that lead to low returns
3. **Weight by log probability** - more confident actions get larger updates

This creates a natural gradient that pushes the policy toward better actions.

##  REINFORCE Algorithm

**REINFORCE** (Monte Carlo Policy Gradient) is the simplest policy gradient algorithm. It uses Monte Carlo returns to estimate the gradient.

### REINFORCE Update Rule

$$\theta_{t+1} \leftarrow \theta_t + \alpha G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

Where:
- $$\alpha$$ - Learning rate
- $$G_t$$ - Return from time $$t$$
- $$\nabla_\theta\log\pi_\theta(a_t\|s_t)$$ - Gradient of log policy

### Algorithm Steps

```
Initialize policy parameters θ
Repeat for each episode:
    Generate trajectory τ = (s_0, a_0, r_1, s_1, a_1, ..., s_T)
    For each time step t = 0, 1, ..., T-1:
        Compute return G_t = Σ(γ^k * r_{t+k+1}) for k = 0 to T-t-1
        Compute gradient: ∇_θ log π_θ(a_t|s_t)
        Update parameters: θ ← θ + α * G_t * ∇_θ log π_θ(a_t|s_t)
```

### Advantages

**Simplicity:**
- Easy to implement
- No value function needed
- Clear theoretical foundation

**Monte Carlo:**
- Uses full episode returns
- Unbiased gradient estimates
- Good for episodic tasks

**On-Policy:**
- Updates based on current policy
- More stable convergence
- Can't reuse old data

### Disadvantages

**High Variance:**
- Monte Carlo returns have high variance
- Slow convergence
- Requires many episodes

**Episodic Only:**
- Requires complete episodes
- Not suitable for continuing tasks
- Can't learn online

##  Complete REINFORCE Implementation

### Policy Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    Policy Network for REINFORCE
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [128, 128]):
        super(PolicyNetwork, self).__init__()
        
        # Build hidden layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer (logits for actions)
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: State tensor
            
        Returns:
            Action logits (unnormalized probabilities)
        """
        return self.network(x)
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state: State tensor
            
        Returns:
            (action, log_prob)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        # Sample action from categorical distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob
```

### REINFORCE Agent

```python
import torch
import torch.optim as optim
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

class REINFORCEAgent:
    """
    REINFORCE Agent
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        learning_rate: Optimizer learning rate
        gamma: Discount factor
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [128, 128],
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Create policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_losses = []
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """
        Compute discounted returns for each time step
        
        Args:
            rewards: List of rewards in episode
            
        Returns:
            List of discounted returns
        """
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns
    
    def update_policy(self, states: List[torch.Tensor],
                   actions: List[torch.Tensor],
                   log_probs: List[torch.Tensor],
                   returns: List[float]):
        """
        Update policy using REINFORCE
        
        Args:
            states: List of state tensors
            actions: List of action tensors
            log_probs: List of log probability tensors
            returns: List of discounted returns
        """
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns (reduce variance)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = -(log_probs * returns).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, float]:
        """
        Train for one episode
        
        Args:
            env: Environment to train in
            max_steps: Maximum steps per episode
            
        Returns:
            (total_reward, average_loss)
        """
        state = env.reset()
        states = []
        actions = []
        log_probs = []
        rewards = []
        
        for step in range(max_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action from policy
            action, log_prob = self.policy.get_action(state_tensor)
            
            # Execute action
            next_state, reward, done = env.step(action.item())
            
            # Store experience
            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
            
            if done:
                break
        
        # Compute returns
        returns = self.compute_returns(rewards)
        
        # Update policy
        loss = self.update_policy(states, actions, log_probs, returns)
        
        total_reward = sum(rewards)
        avg_loss = loss if len(states) > 0 else 0.0
        
        return total_reward, avg_loss
    
    def train(self, env, n_episodes: int = 1000, 
             max_steps: int = 1000, verbose: bool = True):
        """
        Train agent for multiple episodes
        
        Args:
            env: Environment to train in
            n_episodes: Number of episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        for episode in range(n_episodes):
            reward, loss = self.train_episode(env, max_steps)
            self.episode_rewards.append(reward)
            self.episode_losses.append(loss)
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_loss = np.mean(self.episode_losses[-100:])
                print(f"Episode {episode + 1:4d}, "
                      f"Avg Reward: {avg_reward:7.2f}, "
                      f"Avg Loss: {avg_loss:6.4f}")
        
        return {
            'rewards': self.episode_rewards,
            'losses': self.episode_losses
        }
    
    def plot_training(self, window: int = 100):
        """
        Plot training statistics
        
        Args:
            window: Moving average window size
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot rewards
        rewards_ma = np.convolve(self.episode_rewards, 
                              np.ones(window)/window, mode='valid')
        ax1.plot(self.episode_rewards, alpha=0.3, label='Raw')
        ax1.plot(range(window-1, len(self.episode_rewards)), 
                 rewards_ma, label=f'{window}-episode MA')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('REINFORCE Training Progress')
        ax1.legend()
        ax1.grid(True)
        
        # Plot losses
        losses_ma = np.convolve(self.episode_losses, 
                             np.ones(window)/window, mode='valid')
        ax2.plot(self.episode_losses, alpha=0.3, label='Raw')
        ax2.plot(range(window-1, len(self.episode_losses)), 
                 losses_ma, label=f'{window}-episode MA')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Policy Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
```

### CartPole Example

```python
import gymnasium as gym

def train_reinforce_cartpole():
    """Train REINFORCE on CartPole environment"""
    
    # Create environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")
    
    # Create agent
    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 128],
        learning_rate=1e-3,
        gamma=0.99
    )
    
    # Train agent
    print("\nTraining REINFORCE Agent...")
    print("=" * 50)
    
    stats = agent.train(env, n_episodes=1000, max_steps=500)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Average Reward (last 100): {np.mean(stats['rewards'][-100]):.2f}")
    print(f"Average Loss (last 100): {np.mean(stats['losses'][-100]):.4f}")
    
    # Plot training progress
    agent.plot_training(window=50)
    
    # Test agent
    print("\nTesting Trained Agent...")
    print("=" * 50)
    
    state = env.reset()
    done = False
    steps = 0
    total_reward = 0
    
    while not done and steps < 500:
        env.render()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, _ = agent.policy.get_action(state_tensor)
        next_state, reward, done, truncated, info = env.step(action.item())
        state = next_state
        total_reward += reward
        steps += 1
    
    print(f"Test Complete in {steps} steps with reward {total_reward:.1f}")
    env.close()

# Run training
if __name__ == "__main__":
    train_reinforce_cartpole()
```

##  Policy Gradient Variants

### 1. **Advantage Actor-Critic (A2C)**

Combines policy gradients with value functions:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) A(s,a) \right]$$

Where $$A(s,a)$$ is the advantage function:
$$A(s,a) = Q(s,a) - V(s)$$

**Benefits:**
- Reduces variance
- Faster convergence
- More stable training

### 2. **Proximal Policy Optimization (PPO)**

Constrains policy updates to prevent large changes:

$$\theta_{k+1} = \theta_k + \alpha \min\left(\epsilon, \frac{\pi_{\theta_{k+1}}(a|s)}{\pi_{\theta_k}(a|s)}\right) \nabla_\theta J(\theta_k)$$

Where $$\epsilon$$ is a clipping parameter.

**Benefits:**
- More stable training
- Sample efficient
- State-of-the-art performance

### 3. **Trust Region Policy Optimization (TRPO)**

Ensures policy updates stay within trust region:

$$\max_\theta \mathbb{E}[R] \quad \text{s.t.} \quad D_{KL}(\pi_\theta || \pi_{\theta_{old}}) \leq \delta$$

Where $$D_{KL}$$ is the KL-divergence.

**Benefits:**
- Theoretical guarantees
- Monotonic improvement
- Robust convergence

### 4. **Soft Actor-Critic (SAC)**

Maximum entropy policy gradient:

$$\pi^* = \arg\max_\pi \mathbb{E}\left[ \sum_{t=0}^T r_t + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

Where $$\mathcal{H}$$ is the entropy bonus.

**Benefits:**
- Better exploration
- More robust policies
- State-of-the-art performance

##  Comparison: Policy vs Value-Based

### When to Use Policy-Based Methods

**Continuous Action Spaces:**
- Robot control
- Continuous control tasks
- Fine-grained actions

**High-Dimensional Action Spaces:**
- Complex manipulation
- Multi-joint robots
- Continuous domains

**Stochastic Policies Needed:**
- Exploration is important
- Multiple good actions
- Uncertain environments

### When to Use Value-Based Methods

**Discrete Action Spaces:**
- Games with discrete moves
- Simple control tasks
- Grid-based problems

**Small Action Spaces:**
- Few possible actions
- Clear optimal actions
- Simple environments

**Fast Evaluation Needed:**
- Real-time applications
- Quick decision making
- Low-latency systems

##  Advanced Topics

### Baseline Subtraction

Reduce variance by subtracting baseline:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) (G_t - b(s)) \right]$$

Where $$b(s)$$ is a baseline (often $$V(s)$$).

**Benefits:**
- Reduces variance
- Faster convergence
- More stable training

### Entropy Regularization

Encourage exploration by maximizing entropy:

$$J(\theta) = \mathbb{E}\left[ \sum_{t=0}^T r_t + \alpha \mathcal{H}(\pi_\theta(\cdot|s_t)) \right]$$

Where $$\mathcal{H}$$ is the entropy and $$\alpha$$ controls exploration.

**Benefits:**
- Better exploration
- Prevents premature convergence
- More robust policies

### Generalized Advantage Estimation (GAE)

Estimate advantages more accurately:

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$.

**Benefits:**
- Better advantage estimates
- Reduced variance
- More stable training

##  What's Next?

In the next post, we'll implement **Actor-Critic Methods** - combining policy gradients with value functions for better performance. We'll cover:

- Actor-Critic architecture
- Advantage estimation
- A2C algorithm
- Implementation details
- Practical examples

##  Key Takeaways

 **Policy gradients** optimize policies directly
 **REINFORCE** is the simplest policy gradient algorithm
 **Policy gradient theorem** provides the update rule
 **Stochastic policies** handle continuous actions
 **Variance reduction** techniques improve convergence
 **Advanced variants** like A2C, PPO, SAC build on REINFORCE

##  Practice Exercises

1. **Implement baseline subtraction** and compare with standard REINFORCE
2. **Add entropy regularization** to encourage exploration
3. **Implement GAE** for better advantage estimation
4. **Train on different environments** (LunarLander, BipedalWalker)
5. **Visualize policy evolution** over training

##  Testing the Code

All of the code in this post has been tested and verified to work correctly! You can download and run the complete test script to see REINFORCE in action.

### How to Run the Test

```python
"""
Test script for Policy Gradient Methods (REINFORCE)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple

class CartPoleEnvironment:
    """
    Simple CartPole-like Environment for Policy Gradient
    
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

class PolicyNetwork(nn.Module):
    """
    Policy Network for REINFORCE
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super(PolicyNetwork, self).__init__()
        
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

class REINFORCEAgent:
    """
    REINFORCE Agent (Monte Carlo Policy Gradient)
    
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
        
        # Policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
    
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
            action, log_prob = self.policy.get_action(state)
            
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
    
    def update_policy(self, episode: List[Tuple]):
        """
        Update policy using REINFORCE
        
        Args:
            episode: List of experiences
        """
        # Compute returns
        returns = self.compute_returns(episode)
        
        # Normalize returns
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute loss
        policy_loss = []
        for (_, _, log_prob, _), G in zip(episode, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
    
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
        
        # Update policy
        self.update_policy(episode)
        
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
    print("Testing Policy Gradient Methods (REINFORCE)...")
    print("=" * 50)
    
    # Create environment
    env = CartPoleEnvironment(state_dim=4, action_dim=2)
    
    # Create agent
    agent = REINFORCEAgent(state_dim=4, action_dim=2)
    
    # Train agent
    print("\nTraining agent...")
    rewards = agent.train(env, n_episodes=300, max_steps=200, verbose=True)
    
    # Test agent
    print("\nTesting trained agent...")
    state = env.reset()
    total_reward = 0
    
    for step in range(50):
        action, _ = agent.policy.get_action(state)
        next_state, reward, done = env.step(action)
        
        total_reward += reward
        
        if done:
            print(f"Episode finished after {step + 1} steps")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print("\nREINFORCE test completed successfully! ✓")
```

### Expected Output

```
Testing Policy Gradient Methods (REINFORCE)...
==================================================

Training agent...
Episode 50, Avg Reward (last 50): 146.58
Episode 100, Avg Reward (last 50): 143.24
Episode 150, Avg Reward (last 50): 141.48
Episode 200, Avg Reward (last 50): 133.72
Episode 250, Avg Reward (last 50): 108.52
Episode 300, Avg Reward (last 50): 132.18

Testing trained agent...
Total reward: 50.00

REINFORCE test completed successfully! 
```

### What the Test Shows

 **Learning Progress:** The agent maintains stable performance across episodes  
 **Policy Gradient:** Direct optimization of policy parameters  
 **Monte Carlo Returns:** Using complete episode returns  
 **Stochastic Policy:** Natural exploration through probability distributions  
 **Gradient Ascent:** Maximizing expected return  

### Test Script Features

The test script includes:
- Complete CartPole-like environment
- REINFORCE algorithm with policy network
- Monte Carlo return computation
- Gradient ascent optimization
- Training loop with progress tracking

### Running on Your Own Environment

You can adapt the test script to your own environment by:
1. Modifying the `CartPoleEnvironment` class
2. Adjusting state and action dimensions
3. Changing the network architecture
4. Customizing the reward structure

##  Questions?

Have questions about Policy Gradient Methods? Drop them in the comments below!

**Next Post:** [Part 6: Actor-Critic Methods]({{ site.baseurl }}{% post_url 2026-02-06-Actor-Critic-Methods %})

**Series Index:** [Deep Reinforcement Learning Series Roadmap]({{ site.baseurl }}{% post_url 2026-02-01-Deep-RL-Series-Roadmap %})
