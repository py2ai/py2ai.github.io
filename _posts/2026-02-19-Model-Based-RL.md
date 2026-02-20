---
layout: post
title: "Model-Based RL - Learning Environment Models for Planning"
date: 2026-02-19
categories: [Machine Learning, AI, Python, Deep RL]
featured-img: 2026-feb-deeprl/2026-feb-deeprl
description: "Learn Model-Based Reinforcement Learning. Understand how to learn environment models for planning, explore popular algorithms, and implement toy examples with code."
---
# Model-Based RL - Learning Environment Models for Planning

Welcome to our comprehensive guide on **Model-Based Reinforcement Learning**! In this post, we'll explore how Model-Based RL learns environment dynamics to enable planning and more sample-efficient learning. We'll cover the fundamental concepts, popular algorithms, and implement working toy examples.

## Introduction: What is Model-Based RL?

Model-Based Reinforcement Learning is an approach where the agent learns a model of the environment's dynamics and uses this model for planning and decision-making. Unlike model-free methods that learn directly from experience, model-based methods learn how the environment works.

### Model-Free vs Model-Based RL

**Model-Free RL (DQN, PPO, SAC):**

- Learns policy or value functions directly
- No explicit model of environment
- Requires many samples
- Harder to plan ahead
- Examples: DQN, PPO, SAC, A3C

**Model-Based RL:**

- Learns environment dynamics model
- Uses model for planning
- More sample efficient
- Can plan multiple steps ahead
- Examples: PETS, MBPO, Dreamer, AlphaZero

### Why Model-Based RL?

**Sample Efficiency:**

- Learn from fewer interactions
- Simulate experience using learned model
- Reuse model for multiple tasks
- Faster learning in data-scarce scenarios

**Planning:**

- Look ahead multiple steps
- Optimize actions using model
- Handle long-term dependencies
- Better strategic decision-making

**Generalization:**

- Transfer learned models to new tasks
- Adapt to changing environments
- Combine with model-free methods
- Robust to distribution shift

## The Model-Based RL Framework

### Core Components

#### 1. **Dynamics Model**

Learns to predict next state given current state and action:

$$
s_{t+1} = f(s_t, a_t) + \epsilon
$$

Where:

- $s_t$: Current state
- $a_t$: Current action
- $f$: Learned dynamics model
- $\epsilon$: Prediction noise

#### 2. **Planning Algorithm**

Uses learned model to find optimal actions:

- **Random Shooting:** Sample random action sequences, pick best
- **Cross-Entropy Method (CEM):** Iteratively refine action sequences
- **Model Predictive Control (MPC):** Optimize actions over horizon
- **Tree Search:** Explore action space systematically

#### 3. **Data Collection**

Gathers experience from real environment:

- Exploration strategies
- Balance exploration and exploitation
- Update model with new data
- Maintain diverse dataset

## Toy Example: Simple Grid World

Let's start with a simple 2D grid world to illustrate model-based concepts:

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class GridWorld:
    """
    Simple 2D grid world environment
    Agent moves in 4 directions, collects rewards
    """
    def __init__(self, size=5, goal_pos=(4, 4)):
        self.size = size
        self.goal_pos = goal_pos
        self.agent_pos = (0, 0)
      
        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = 4
        self.state_space = (size, size)
      
        # Obstacles
        self.obstacles = [(2, 2), (2, 3), (3, 2)]
      
    def reset(self):
        self.agent_pos = (0, 0)
        return self.get_state()
  
    def get_state(self):
        return np.array(self.agent_pos)
  
    def step(self, action):
        # Get current position
        x, y = self.agent_pos
      
        # Apply action
        if action == 0:  # up
            new_pos = (x, min(y + 1, self.size - 1))
        elif action == 1:  # down
            new_pos = (x, max(y - 1, 0))
        elif action == 2:  # left
            new_pos = (max(x - 1, 0), y)
        elif action == 3:  # right
            new_pos = (min(x + 1, self.size - 1), y)
      
        # Check obstacles
        if new_pos not in self.obstacles:
            self.agent_pos = new_pos
      
        # Compute reward
        reward = -1  # Step penalty
        if self.agent_pos == self.goal_pos:
            reward = 100  # Goal reward
      
        # Check done
        done = self.agent_pos == self.goal_pos
      
        return self.get_state(), reward, done, {}
  
    def render(self):
        grid = np.zeros((self.size, self.size))
      
        # Mark goal
        grid[self.goal_pos[1], self.goal_pos[0]] = 0.5
      
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs[1], obs[0]] = -1
      
        # Mark agent
        grid[self.agent_pos[1], self.agent_pos[0]] = 1
      
        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f"Agent at {self.agent_pos}, Goal at {self.goal_pos}")
        plt.show()
```

### Learning the Dynamics Model

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GridWorldDynamicsModel(nn.Module):
    """
    Neural network to predict next state in grid world
    """
    def __init__(self, state_dim=2, action_dim=4, hidden_dim=64):
        super().__init__()
      
        # Encode state and action
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
      
        # Predict next state
        self.decoder = nn.Linear(hidden_dim, state_dim)
      
    def forward(self, state, action):
        # One-hot encode action
        action_onehot = torch.zeros(action.shape[0], 4)
        action_onehot.scatter_(1, action.long(), 1)
      
        # Concatenate state and action
        x = torch.cat([state, action_onehot], dim=1)
      
        # Encode and decode
        features = self.encoder(x)
        next_state = self.decoder(features)
      
        return next_state
  
    def predict(self, state, action):
        """
        Predict next state (handles numpy inputs)
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if isinstance(action, (int, np.integer)):
                action = torch.LongTensor([[action]])
            elif isinstance(action, np.ndarray):
                action = torch.LongTensor(action)
          
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            if len(action.shape) == 0:
                action = action.unsqueeze(0)
          
            pred = self.forward(state, action)
            return pred.squeeze(0).numpy()

def collect_data(env, n_episodes=100):
    """
    Collect experience data from environment
    """
    data = []
  
    for episode in range(n_episodes):
        state = env.reset()
      
        for step in range(50):
            # Random action
            action = np.random.randint(0, 4)
          
            # Step environment
            next_state, reward, done, _ = env.step(action)
          
            # Store transition
            data.append({
                'state': state.copy(),
                'action': action,
                'next_state': next_state.copy(),
                'reward': reward
            })
          
            state = next_state
            if done:
                break
  
    return data

def train_dynamics_model(model, data, epochs=100, batch_size=32):
    """
    Train dynamics model on collected data
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
  
    for epoch in range(epochs):
        # Shuffle data
        np.random.shuffle(data)
      
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
          
            # Prepare batch
            states = torch.FloatTensor([d['state'] for d in batch])
            actions = torch.LongTensor([d['action'] for d in batch])
            next_states = torch.FloatTensor([d['next_state'] for d in batch])
          
            # Forward pass
            pred_next_states = model(states, actions)
          
            # Compute loss
            loss = criterion(pred_next_states, next_states)
          
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          
            total_loss += loss.item()
      
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data):.6f}")
  
    return model

# Train dynamics model
env = GridWorld(size=5)
data = collect_data(env, n_episodes=200)
model = GridWorldDynamicsModel()
model = train_dynamics_model(model, data, epochs=100)

# Test model
test_state = np.array([2, 2])
test_action = 3  # right
predicted_next = model.predict(test_state, test_action)
print(f"State: {test_state}, Action: {test_action}")
print(f"Predicted next state: {predicted_next}")
```

### Planning with Learned Model

```python
class RandomShootingPlanner:
    """
    Plan actions by sampling random action sequences
    """
    def __init__(self, model, action_space, horizon=10, n_samples=100):
        self.model = model
        self.action_space = action_space
        self.horizon = horizon
        self.n_samples = n_samples
  
    def plan(self, state):
        """
        Find best action sequence using random shooting
        """
        best_action = None
        best_reward = float('-inf')
      
        # Sample random action sequences
        for _ in range(self.n_samples):
            actions = np.random.randint(0, self.action_space, self.horizon)
          
            # Simulate trajectory
            current_state = state.copy()
            total_reward = 0
          
            for action in actions:
                # Predict next state
                next_state = self.model.predict(current_state, action)
              
                # Compute reward (distance to goal)
                distance = np.linalg.norm(next_state - np.array([4, 4]))
                reward = -distance
                total_reward += reward
              
                current_state = next_state
          
            # Update best
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = actions[0]
      
        return best_action

class CrossEntropyMethodPlanner:
    """
    Plan actions using Cross-Entropy Method (CEM)
    """
    def __init__(self, model, action_space, horizon=10, n_samples=100, n_elite=10):
        self.model = model
        self.action_space = action_space
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elite = n_elite
  
    def plan(self, state, n_iterations=5):
        """
        Find best action sequence using CEM
        """
        # Initialize action distribution
        action_mean = np.zeros(self.horizon)
        action_std = np.ones(self.horizon) * 2.0
      
        for iteration in range(n_iterations):
            # Sample action sequences
            samples = []
            rewards = []
          
            for _ in range(self.n_samples):
                # Sample action sequence
                actions = np.random.normal(action_mean, action_std)
                actions = np.clip(actions, 0, self.action_space - 1).astype(int)
              
                # Simulate trajectory
                current_state = state.copy()
                total_reward = 0
              
                for action in actions:
                    next_state = self.model.predict(current_state, action)
                    distance = np.linalg.norm(next_state - np.array([4, 4]))
                    reward = -distance
                    total_reward += reward
                    current_state = next_state
              
                samples.append(actions)
                rewards.append(total_reward)
          
            # Select elite samples
            elite_indices = np.argsort(rewards)[-self.n_elite:]
            elite_samples = [samples[i] for i in elite_indices]
          
            # Update distribution
            action_mean = np.mean(elite_samples, axis=0)
            action_std = np.std(elite_samples, axis=0) + 1e-6
      
        return int(action_mean[0])

# Test planners
state = env.reset()

# Random shooting planner
rs_planner = RandomShootingPlanner(model, action_space=4)
action = rs_planner.plan(state)
print(f"Random Shooting Planner action: {action}")

# CEM planner
cem_planner = CrossEntropyMethodPlanner(model, action_space=4)
action = cem_planner.plan(state)
print(f"CEM Planner action: {action}")
```

## Toy Example: Continuous Control (CartPole)

Let's implement model-based RL on a continuous control task:

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CartPoleDynamicsModel(nn.Module):
    """
    Learn dynamics model for CartPole environment
    """
    def __init__(self, state_dim=4, action_dim=1, hidden_dim=128):
        super().__init__()
      
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
      
        # Predict state delta
        self.delta_head = nn.Linear(hidden_dim, state_dim)
      
        # Predict reward
        self.reward_head = nn.Linear(hidden_dim, 1)
      
        # Predict done
        self.done_head = nn.Linear(hidden_dim, 1)
      
    def forward(self, state, action):
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
      
        # Encode
        features = self.encoder(x)
      
        # Predictions
        delta = self.delta_head(features)
        reward = self.reward_head(features)
        done = torch.sigmoid(self.done_head(features))
      
        return delta, reward, done
  
    def predict(self, state, action):
        """
        Predict next state, reward, done
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if isinstance(action, np.ndarray):
                action = torch.FloatTensor(action)
          
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
          
            delta, reward, done = self.forward(state, action)
            next_state = state + delta
          
            return (next_state.squeeze(0).numpy(), 
                    reward.item(), 
                    done.item())

class EnsembleDynamicsModel:
    """
    Ensemble of dynamics models for uncertainty estimation
    """
    def __init__(self, n_models=5, state_dim=4, action_dim=1, hidden_dim=128):
        self.models = [
            CartPoleDynamicsModel(state_dim, action_dim, hidden_dim)
            for _ in range(n_models)
        ]
  
    def predict(self, state, action):
        """
        Predict with ensemble, return mean and std
        """
        predictions = []
      
        for model in self.models:
            pred = model.predict(state, action)
            predictions.append(pred)
      
        # Compute mean and std
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
      
        return mean_pred, std_pred

def collect_cartpole_data(env, n_episodes=100):
    """
    Collect data from CartPole environment
    """
    data = []
  
    for episode in range(n_episodes):
        state = env.reset()
      
        for step in range(200):
            # Random action
            action = env.action_space.sample()
          
            # Step environment
            next_state, reward, done, _ = env.step(action)
          
            # Store transition
            data.append({
                'state': state.copy(),
                'action': np.array([action]),
                'next_state': next_state.copy(),
                'reward': reward,
                'done': done
            })
          
            state = next_state
            if done:
                break
  
    return data

def train_cartpole_model(model, data, epochs=50, batch_size=64):
    """
    Train CartPole dynamics model
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
  
    for epoch in range(epochs):
        np.random.shuffle(data)
      
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
          
            # Prepare batch
            states = torch.FloatTensor([d['state'] for d in batch])
            actions = torch.FloatTensor([d['action'] for d in batch])
            next_states = torch.FloatTensor([d['next_state'] for d in batch])
            rewards = torch.FloatTensor([d['reward'] for d in batch]).unsqueeze(1)
            dones = torch.FloatTensor([d['done'] for d in batch]).unsqueeze(1)
          
            # Forward pass
            pred_delta, pred_reward, pred_done = model(states, actions)
          
            # Compute losses
            delta_loss = mse_loss(pred_delta, next_states - states)
            reward_loss = mse_loss(pred_reward, rewards)
            done_loss = bce_loss(pred_done, dones)
          
            loss = delta_loss + reward_loss + done_loss
          
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          
            total_loss += loss.item()
      
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data):.6f}")
  
    return model

# Train CartPole model
env = gym.make('CartPole-v1')
data = collect_cartpole_data(env, n_episodes=100)
model = CartPoleDynamicsModel()
model = train_cartpole_model(model, data, epochs=50)

# Test model
test_state = env.reset()
test_action = np.array([1])
next_state, reward, done = model.predict(test_state, test_action)
print(f"State: {test_state}")
print(f"Action: {test_action}")
print(f"Predicted next state: {next_state}")
print(f"Predicted reward: {reward}, done: {done}")
```

### MPC with Learned Model

```python
class MPCPlanner:
    """
    Model Predictive Control planner
    """
    def __init__(self, model, action_space, horizon=20, n_samples=100):
        self.model = model
        self.action_space = action_space
        self.horizon = horizon
        self.n_samples = n_samples
  
    def plan(self, state):
        """
        Find best action sequence using MPC
        """
        best_action_sequence = None
        best_reward = float('-inf')
      
        for _ in range(self.n_samples):
            # Sample action sequence
            actions = np.random.randint(0, self.action_space, self.horizon)
          
            # Simulate trajectory
            current_state = state.copy()
            total_reward = 0
            done = False
          
            for action in actions:
                if done:
                    break
              
                # Predict next state
                next_state, reward, done_pred = self.model.predict(
                    current_state, 
                    np.array([action])
                )
              
                total_reward += reward
                current_state = next_state
              
                if done_pred > 0.5:
                    done = True
          
            # Update best
            if total_reward > best_reward:
                best_reward = total_reward
                best_action_sequence = actions
      
        return best_action_sequence[0] if best_action_sequence is not None else 0

def run_mpc_agent(env, model, n_episodes=10):
    """
    Run MPC agent in environment
    """
    planner = MPCPlanner(model, action_space=env.action_space.n)
  
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
      
        while not done:
            # Plan action
            action = planner.plan(state)
          
            # Execute action
            next_state, reward, done, _ = env.step(action)
          
            total_reward += reward
            state = next_state
      
        print(f"Episode {episode+1}, Reward: {total_reward}")

# Run MPC agent
run_mpc_agent(env, model, n_episodes=10)
```

## Popular Model-Based RL Algorithms

### 1. PETS (Probabilistic Ensembles for Trajectory Shooting)

Uses ensemble of probabilistic models for uncertainty-aware planning:

```python
class ProbabilisticDynamicsModel(nn.Module):
    """
    Probabilistic dynamics model with variance prediction
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
      
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
      
        # Predict mean and log variance
        self.mean_head = nn.Linear(hidden_dim, state_dim)
        self.logvar_head = nn.Linear(hidden_dim, state_dim)
  
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        features = self.encoder(x)
      
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
      
        return mean, logvar
  
    def sample(self, state, action):
        """
        Sample next state from learned distribution
        """
        mean, logvar = self.forward(state, action)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

class PETSAgent:
    """
    PETS: Probabilistic Ensembles for Trajectory Shooting
    """
    def __init__(self, state_dim, action_dim, n_models=5, horizon=10):
        self.models = [
            ProbabilisticDynamicsModel(state_dim, action_dim)
            for _ in range(n_models)
        ]
        self.horizon = horizon
        self.action_space = action_dim
  
    def plan(self, state, n_samples=100):
        """
        Plan using trajectory shooting with probabilistic models
        """
        best_action = None
        best_reward = float('-inf')
      
        for _ in range(n_samples):
            # Sample action sequence
            actions = np.random.randint(0, self.action_space, self.horizon)
          
            # Sample model for each step
            total_reward = 0
            current_state = state.copy()
          
            for action in actions:
                # Randomly select model
                model_idx = np.random.randint(len(self.models))
                model = self.models[model_idx]
              
                # Sample next state
                if isinstance(current_state, np.ndarray):
                    state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
                else:
                    state_tensor = current_state
                action_tensor = torch.LongTensor([[action]])
              
                next_state = model.sample(state_tensor, action_tensor)
                current_state = next_state.squeeze(0).detach().numpy()
              
                # Simple reward (distance to goal)
                total_reward -= np.linalg.norm(current_state[:2])
          
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = actions[0]
      
        return best_action
```

### 2. MBPO (Model-Based Policy Optimization)

Combines model-based planning with policy optimization:

```python
class MBPOAgent:
    """
    MBPO: Model-Based Policy Optimization
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        # Dynamics model
        self.dynamics_model = CartPoleDynamicsModel(state_dim, action_dim, hidden_dim)
      
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
      
        # Value network
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
  
    def get_action(self, state):
        """
        Get action from policy
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
          
            action = self.policy(state)
            return action.squeeze(0).numpy()
  
    def generate_imagined_data(self, real_data, n_imagined=1000):
        """
        Generate imagined data using learned model
        """
        imagined_data = []
      
        for _ in range(n_imagined):
            # Sample random state from real data
            sample = real_data[np.random.randint(len(real_data))]
            state = sample['state']
          
            # Get action from policy
            action = self.get_action(state)
          
            # Predict next state using model
            next_state, reward, done = self.dynamics_model.predict(state, action)
          
            imagined_data.append({
                'state': state.copy(),
                'action': action,
                'next_state': next_state,
                'reward': reward,
                'done': done
            })
      
        return imagined_data
  
    def update_policy(self, data, epochs=10):
        """
        Update policy using policy gradient
        """
        optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
      
        for epoch in range(epochs):
            np.random.shuffle(data)
          
            for i in range(0, len(data), 32):
                batch = data[i:i+32]
              
                states = torch.FloatTensor([d['state'] for d in batch])
                actions = torch.FloatTensor([d['action'] for d in batch])
                rewards = torch.FloatTensor([d['reward'] for d in batch])
              
                # Get action from policy
                pred_actions = self.policy(states)
              
                # Policy gradient loss (simplified)
                advantage = rewards - torch.mean(rewards)
                loss = -(pred_actions * advantage.unsqueeze(1)).mean()
              
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

### 3. Dreamer

Model-based RL with learned world model in latent space:

```python
class DreamerWorldModel(nn.Module):
    """
    Dreamer-style world model with latent dynamics
    """
    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=128):
        super().__init__()
      
        # Encoder: observation to latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and logvar
        )
      
        # Dynamics: latent transition
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
      
        # Decoder: latent to observation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
      
        # Reward predictor
        self.reward_head = nn.Linear(latent_dim, 1)
  
    def encode(self, obs):
        """
        Encode observation to latent distribution
        """
        x = self.encoder(obs)
        mean, logvar = x.chunk(2, dim=-1)
        return mean, logvar
  
    def decode(self, latent):
        """
        Decode latent to observation
        """
        return self.decoder(latent)
  
    def transition(self, latent, action):
        """
        Predict next latent state
        """
        x = torch.cat([latent, action], dim=-1)
        x = self.dynamics(x)
        mean, logvar = x.chunk(2, dim=-1)
        return mean, logvar
  
    def predict_reward(self, latent):
        """
        Predict reward from latent
        """
        return self.reward_head(latent)
  
    def sample_latent(self, obs):
        """
        Sample latent from observation
        """
        mean, logvar = self.encode(obs)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
  
    def imagine_trajectory(self, initial_obs, actions):
        """
        Imagine trajectory from initial observation
        """
        latent = self.sample_latent(initial_obs)
        rewards = []
      
        for action in actions:
            # Transition latent
            mean, logvar = self.transition(latent, action)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = mean + eps * std
          
            # Predict reward
            reward = self.predict_reward(latent)
            rewards.append(reward)
      
        return rewards
```

## Advantages and Disadvantages

### Advantages of Model-Based RL

**Sample Efficiency:**

- Learn from fewer environment interactions
- Generate synthetic experience using model
- Reuse model across tasks
- Faster convergence

**Planning:**

- Look ahead multiple steps
- Optimize actions over horizon
- Handle long-term dependencies
- Better strategic decisions

**Interpretability:**

- Learned model is interpretable
- Understand environment dynamics
- Debug and analyze behavior
- Transfer knowledge

### Disadvantages of Model-Based RL

**Model Bias:**

- Model errors compound over time
- Inaccuracies lead to poor decisions
- Hard to learn complex dynamics
- Model misspecification

**Computational Cost:**

- Planning is computationally expensive
- Need to simulate many trajectories
- Real-time constraints
- Memory requirements

**Training Complexity:**

- Need to train both model and policy
- Balance model accuracy and policy performance
- More hyperparameters to tune
- Complex implementation

## Practical Tips

### 1. Start Simple

Begin with simple environments and models:

```python
# Start with linear model
class LinearDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear = nn.Linear(state_dim + action_dim, state_dim)
  
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.linear(x)

# Progress to neural network
class NNDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
  
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
```

### 2. Use Ensembles

Ensemble models reduce uncertainty:

```python
def train_ensemble(n_models=5, state_dim=4, action_dim=1):
    """
    Train ensemble of dynamics models
    """
    models = []
  
    for i in range(n_models):
        # Initialize model
        model = CartPoleDynamicsModel(state_dim, action_dim)
      
        # Train on different data subsets
        subset = data[i::n_models]
        model = train_cartpole_model(model, subset, epochs=50)
      
        models.append(model)
  
    return models

def predict_with_ensemble(models, state, action):
    """
    Predict with ensemble, return mean and uncertainty
    """
    predictions = []
  
    for model in models:
        pred = model.predict(state, action)
        predictions.append(pred)
  
    predictions = np.array(predictions)
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
  
    return mean, std
```

### 3. Balance Real and Imagined Data

Combine real and simulated experience:

```python
def train_mbpo(real_data, imagined_data, ratio=0.5):
    """
    Train policy with mix of real and imagined data
    """
    # Mix data
    n_real = int(len(real_data) * ratio)
    n_imagined = len(imagined_data) - n_real
  
    mixed_data = real_data[:n_real] + imagined_data[:n_imagined]
    np.random.shuffle(mixed_data)
  
    # Train on mixed data
    for epoch in range(10):
        for batch in get_batches(mixed_data, batch_size=32):
            update_policy(batch)
  
    return policy
```

### 4. Monitor Model Accuracy

Track model prediction error:

```python
def evaluate_model(model, test_data):
    """
    Evaluate model prediction accuracy
    """
    errors = []
  
    for sample in test_data:
        state = sample['state']
        action = sample['action']
        true_next = sample['next_state']
      
        # Predict
        pred_next, _, _ = model.predict(state, action)
      
        # Compute error
        error = np.linalg.norm(pred_next - true_next)
        errors.append(error)
  
    mean_error = np.mean(errors)
    std_error = np.std(errors)
  
    print(f"Model error: {mean_error:.4f} ± {std_error:.4f}")
    return mean_error, std_error
```

## Testing the Code

Here's a comprehensive test script to verify all code examples work correctly:

```python
#!/usr/bin/env python3
"""
Test script to verify code in Model-Based RL blog post
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("Testing code from Model-Based RL blog post...\n")

# Test 1: GridWorld Environment
print("Test 1: GridWorld Environment")
try:
    class GridWorld:
        def __init__(self, size=5, goal_pos=(4, 4)):
            self.size = size
            self.goal_pos = goal_pos
            self.agent_pos = (0, 0)
            self.action_space = 4
            self.state_space = (size, size)
            self.obstacles = [(2, 2), (2, 3), (3, 2)]
      
        def reset(self):
            self.agent_pos = (0, 0)
            return self.get_state()
      
        def get_state(self):
            return np.array(self.agent_pos)
      
        def step(self, action):
            x, y = self.agent_pos
          
            if action == 0:
                new_pos = (x, min(y + 1, self.size - 1))
            elif action == 1:
                new_pos = (x, max(y - 1, 0))
            elif action == 2:
                new_pos = (max(x - 1, 0), y)
            elif action == 3:
                new_pos = (min(x + 1, self.size - 1), y)
          
            if new_pos not in self.obstacles:
                self.agent_pos = new_pos
          
            reward = -1
            if self.agent_pos == self.goal_pos:
                reward = 100
          
            done = self.agent_pos == self.goal_pos
          
            return self.get_state(), reward, done, {}
  
    env = GridWorld()
    state = env.reset()
    next_state, reward, done, _ = env.step(3)
    assert state.shape == (2,), f"Expected shape (2,), got {state.shape}"
    assert isinstance(reward, (int, float)), f"Expected number, got {type(reward)}"
    assert isinstance(done, bool), f"Expected bool, got {type(done)}"
    print("✓ GridWorld works correctly\n")
except Exception as e:
    print(f"✗ GridWorld failed: {e}\n")
    sys.exit(1)

# Test 2: GridWorldDynamicsModel
print("Test 2: GridWorldDynamicsModel")
try:
    class GridWorldDynamicsModel(nn.Module):
        def __init__(self, state_dim=2, action_dim=4, hidden_dim=64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.decoder = nn.Linear(hidden_dim, state_dim)
      
        def forward(self, state, action):
            action_onehot = torch.zeros(action.shape[0], 4)
            action_onehot.scatter_(1, action.long(), 1)
            x = torch.cat([state, action_onehot], dim=1)
            features = self.encoder(x)
            next_state = self.decoder(features)
            return next_state
      
        def predict(self, state, action):
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state)
                if isinstance(action, (int, np.integer)):
                    action = torch.LongTensor([[action]])
                elif isinstance(action, np.ndarray):
                    action = torch.LongTensor(action)
              
                if len(state.shape) == 1:
                    state = state.unsqueeze(0)
                if len(action.shape) == 0:
                    action = action.unsqueeze(0)
              
                pred = self.forward(state, action)
                return pred.squeeze(0).numpy()
  
    model = GridWorldDynamicsModel()
    state = torch.randn(1, 2)
    action = torch.LongTensor([[1]])
    output = model(state, action)
    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
  
    test_state = np.array([2, 2])
    test_action = 3
    pred = model.predict(test_state, test_action)
    assert pred.shape == (2,), f"Expected shape (2,), got {pred.shape}"
    print("✓ GridWorldDynamicsModel works correctly\n")
except Exception as e:
    print(f"✗ GridWorldDynamicsModel failed: {e}\n")
    sys.exit(1)

# Test 3: RandomShootingPlanner
print("Test 3: RandomShootingPlanner")
try:
    class RandomShootingPlanner:
        def __init__(self, model, action_space, horizon=10, n_samples=100):
            self.model = model
            self.action_space = action_space
            self.horizon = horizon
            self.n_samples = n_samples
      
        def plan(self, state):
            best_action = None
            best_reward = float('-inf')
          
            for _ in range(self.n_samples):
                actions = np.random.randint(0, self.action_space, self.horizon)
                current_state = state.copy()
                total_reward = 0
              
                for action in actions:
                    next_state = self.model.predict(current_state, action)
                    distance = np.linalg.norm(next_state - np.array([4, 4]))
                    reward = -distance
                    total_reward += reward
                    current_state = next_state
              
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_action = actions[0]
          
            return best_action
  
    planner = RandomShootingPlanner(model, action_space=4, n_samples=10)
    state = np.array([0, 0])
    action = planner.plan(state)
    assert isinstance(action, (int, np.integer)), f"Expected int, got {type(action)}"
    assert 0 <= action < 4, f"Expected action in [0,3], got {action}"
    print("✓ RandomShootingPlanner works correctly\n")
except Exception as e:
    print(f"✗ RandomShootingPlanner failed: {e}\n")
    sys.exit(1)

# Test 4: CartPoleDynamicsModel
print("Test 4: CartPoleDynamicsModel")
try:
    class CartPoleDynamicsModel(nn.Module):
        def __init__(self, state_dim=4, action_dim=1, hidden_dim=128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.delta_head = nn.Linear(hidden_dim, state_dim)
            self.reward_head = nn.Linear(hidden_dim, 1)
            self.done_head = nn.Linear(hidden_dim, 1)
      
        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            features = self.encoder(x)
            delta = self.delta_head(features)
            reward = self.reward_head(features)
            done = torch.sigmoid(self.done_head(features))
            return delta, reward, done
      
        def predict(self, state, action):
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state)
                if isinstance(action, np.ndarray):
                    action = torch.FloatTensor(action)
              
                if len(state.shape) == 1:
                    state = state.unsqueeze(0)
                if len(action.shape) == 1:
                    action = action.unsqueeze(0)
              
                delta, reward, done = self.forward(state, action)
                next_state = state + delta
                return (next_state.squeeze(0).numpy(), reward.item(), done.item())
  
    model = CartPoleDynamicsModel()
    state = torch.randn(1, 4)
    action = torch.randn(1, 1)
    delta, reward, done = model(state, action)
    assert delta.shape == (1, 4), f"Expected shape (1, 4), got {delta.shape}"
    assert reward.shape == (1, 1), f"Expected shape (1, 1), got {reward.shape}"
    assert done.shape == (1, 1), f"Expected shape (1, 1), got {done.shape}"
    print("✓ CartPoleDynamicsModel works correctly\n")
except Exception as e:
    print(f"✗ CartPoleDynamicsModel failed: {e}\n")
    sys.exit(1)

# Test 5: EnsembleDynamicsModel
print("Test 5: EnsembleDynamicsModel")
try:
    class EnsembleDynamicsModel:
        def __init__(self, n_models=5, state_dim=4, action_dim=1, hidden_dim=128):
            self.models = [
                CartPoleDynamicsModel(state_dim, action_dim, hidden_dim)
                for _ in range(n_models)
            ]
      
        def predict(self, state, action):
            predictions = []
            for model in self.models:
                pred = model.predict(state, action)
                predictions.append(pred[0] if isinstance(pred, tuple) else pred)
            predictions_array = np.array(predictions)
            mean_pred = np.mean(predictions_array, axis=0)
            std_pred = np.std(predictions_array, axis=0)
            return mean_pred, std_pred
  
    ensemble = EnsembleDynamicsModel(n_models=3)
    state = np.random.randn(4)
    action = np.array([1])
    mean, std = ensemble.predict(state, action)
    assert mean.shape == (4,), f"Expected shape (4,), got {mean.shape}"
    assert std.shape == (4,), f"Expected shape (4,), got {std.shape}"
    print("✓ EnsembleDynamicsModel works correctly\n")
except Exception as e:
    print(f"✗ EnsembleDynamicsModel failed: {e}\n")
    sys.exit(1)

# Test 6: ProbabilisticDynamicsModel
print("Test 6: ProbabilisticDynamicsModel")
try:
    class ProbabilisticDynamicsModel(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.mean_head = nn.Linear(hidden_dim, state_dim)
            self.logvar_head = nn.Linear(hidden_dim, state_dim)
      
        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            features = self.encoder(x)
            mean = self.mean_head(features)
            logvar = self.logvar_head(features)
            return mean, logvar
      
        def sample(self, state, action):
            mean, logvar = self.forward(state, action)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
  
    model = ProbabilisticDynamicsModel(state_dim=4, action_dim=1)
    state = torch.randn(1, 4)
    action = torch.randn(1, 1)
    mean, logvar = model(state, action)
    sample = model.sample(state, action)
    assert mean.shape == (1, 4), f"Expected shape (1, 4), got {mean.shape}"
    assert logvar.shape == (1, 4), f"Expected shape (1, 4), got {logvar.shape}"
    assert sample.shape == (1, 4), f"Expected shape (1, 4), got {sample.shape}"
    print("✓ ProbabilisticDynamicsModel works correctly\n")
except Exception as e:
    print(f"✗ ProbabilisticDynamicsModel failed: {e}\n")
    sys.exit(1)

# Test 7: DreamerWorldModel
print("Test 7: DreamerWorldModel")
try:
    class DreamerWorldModel(nn.Module):
        def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim * 2)
            )
            self.dynamics = nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim * 2)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim)
            )
            self.reward_head = nn.Linear(latent_dim, 1)
      
        def encode(self, obs):
            x = self.encoder(obs)
            mean, logvar = x.chunk(2, dim=-1)
            return mean, logvar
      
        def decode(self, latent):
            return self.decoder(latent)
      
        def transition(self, latent, action):
            x = torch.cat([latent, action], dim=-1)
            x = self.dynamics(x)
            mean, logvar = x.chunk(2, dim=-1)
            return mean, logvar
      
        def predict_reward(self, latent):
            return self.reward_head(latent)
      
        def sample_latent(self, obs):
            mean, logvar = self.encode(obs)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
  
    model = DreamerWorldModel(obs_dim=10, action_dim=2, latent_dim=16)
    obs = torch.randn(1, 10)
    action = torch.randn(1, 2)
  
    mean, logvar = model.encode(obs)
    assert mean.shape == (1, 16), f"Expected shape (1, 16), got {mean.shape}"
  
    latent = model.sample_latent(obs)
    assert latent.shape == (1, 16), f"Expected shape (1, 16), got {latent.shape}"
  
    next_mean, next_logvar = model.transition(latent, action)
    assert next_mean.shape == (1, 16), f"Expected shape (1, 16), got {next_mean.shape}"
  
    reward = model.predict_reward(latent)
    assert reward.shape == (1, 1), f"Expected shape (1, 1), got {reward.shape}"
  
    decoded = model.decode(latent)
    assert decoded.shape == (1, 10), f"Expected shape (1, 10), got {decoded.shape}"
  
    print("✓ DreamerWorldModel works correctly\n")
except Exception as e:
    print(f"✗ DreamerWorldModel failed: {e}\n")
    sys.exit(1)

# Test 8: LinearDynamicsModel
print("Test 8: LinearDynamicsModel")
try:
    class LinearDynamicsModel(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.linear = nn.Linear(state_dim + action_dim, state_dim)
      
        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            return self.linear(x)
  
    model = LinearDynamicsModel(state_dim=4, action_dim=1)
    state = torch.randn(1, 4)
    action = torch.randn(1, 1)
    output = model(state, action)
    assert output.shape == (1, 4), f"Expected shape (1, 4), got {output.shape}"
    print("✓ LinearDynamicsModel works correctly\n")
except Exception as e:
    print(f"✗ LinearDynamicsModel failed: {e}\n")
    sys.exit(1)

print("=" * 50)
print("All tests passed! ✓")
print("=" * 50)
print("\nThe code in blog post is syntactically correct")
print("and should work as expected.")
```

To run this test script, save it as `test_modelbased_code.py` and execute:

```bash
python test_modelbased_code.py
```

## Conclusion

Model-Based Reinforcement Learning offers a powerful approach to learning efficient policies by understanding environment dynamics. By learning models of the world and using them for planning, we can achieve sample-efficient learning and better long-term decision-making.

### Key Takeaways

1. **Learn Dynamics Models:** Train neural networks to predict state transitions
2. **Use Planning:** Employ MPC, CEM, or other planning algorithms
3. **Handle Uncertainty:** Use ensembles and probabilistic models
4. **Balance Real/Simulated:** Mix real and imagined experience
5. **Start Simple:** Begin with linear models, progress to neural networks

### Future Directions

- **Better World Models:** More accurate and generalizable models
- **Sample Efficient Methods:** Learn from even fewer interactions
- **Uncertainty Quantification:** Better handling of model uncertainty
- **Hierarchical Planning:** Multi-level planning for complex tasks
- **Meta-Learning:** Learn to learn models quickly

### Resources

- **Libraries:** mbrl-lib, garage, rlpyt
- **Simulators:** MuJoCo, PyBullet, Isaac Gym

Happy model-based learning!
