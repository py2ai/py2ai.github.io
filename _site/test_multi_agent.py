"""
Test script for Multi-Agent Reinforcement Learning (MADDPG)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

class MultiAgentEnvironment:
    """
    Simple Multi-Agent Environment for MADDPG
    
    Args:
        n_agents: Number of agents
        state_dim: Dimension of state space
        action_dim: Dimension of action space
    """
    def __init__(self, n_agents: int = 2, state_dim: int = 4, action_dim: int = 2):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = None
        self.steps = 0
        self.max_steps = 200
    
    def reset(self) -> List[np.ndarray]:
        """Reset environment"""
        self.states = [np.random.randn(self.state_dim).astype(np.float32) 
                      for _ in range(self.n_agents)]
        self.steps = 0
        return self.states
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool]:
        """
        Take actions in environment
        
        Args:
            actions: List of actions for each agent (continuous)
            
        Returns:
            (next_states, rewards, done)
        """
        # Simple dynamics
        for i in range(self.n_agents):
            # Use only first action dimension to update state
            action_effect = actions[i][0] if len(actions[i]) > 0 else 0
            self.states[i] = self.states[i] + np.random.randn(self.state_dim).astype(np.float32) * 0.1 + action_effect * 0.1
        
        # Rewards based on states
        rewards = [1.0 if abs(state[0]) < 2.0 else -1.0 for state in self.states]
        
        # Check if done
        self.steps += 1
        done = self.steps >= self.max_steps
        
        return self.states, rewards, done

class MADDPGNetwork(nn.Module):
    """
    MADDPG Network for Multi-Agent RL
    
    Args:
        n_agents: Number of agents
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(self, n_agents: int, state_dim: int, action_dim: int,
                 hidden_dims: list = [64, 64]):
        super(MADDPGNetwork, self).__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor networks (one per agent)
        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], action_dim),
                nn.Tanh()
            )
            for _ in range(n_agents)
        ])
        
        # Critic network (centralized)
        self.critic = nn.Sequential(
            nn.Linear(n_agents * (state_dim + action_dim), hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
    
    def get_actions(self, observations: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Get actions for all agents
        
        Args:
            observations: List of observations for each agent
            
        Returns:
            List of actions
        """
        actions = []
        for i, obs in enumerate(observations):
            action = self.actors[i](obs)
            actions.append(action)
        return actions
    
    def get_q_values(self, all_observations: List[torch.Tensor],
                     all_actions: List[torch.Tensor],
                     agent_idx: int) -> torch.Tensor:
        """
        Get Q-values for specific agent
        
        Args:
            all_observations: All agents' observations
            all_actions: All agents' actions
            agent_idx: Index of agent to get Q-value for
            
        Returns:
            Q-value for agent
        """
        # Concatenate all observations and actions
        obs_actions = []
        for obs, action in zip(all_observations, all_actions):
            obs_actions.append(torch.cat([obs, action], dim=-1))
        
        x = torch.cat(obs_actions, dim=-1)
        q_values = self.critic(x)
        return q_values

class MADDPGAgent:
    """
    MADDPG Agent
    
    Args:
        n_agents: Number of agents
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        lr: Learning rate
        gamma: Discount factor
        tau: Target network update rate
        buffer_size: Replay buffer size
        batch_size: Training batch size
    """
    def __init__(self, n_agents: int, state_dim: int, action_dim: int,
                 hidden_dims: list = [64, 64],
                 lr: float = 1e-3,
                 gamma: float = 0.95,
                 tau: float = 0.01,
                 buffer_size: int = 100000,
                 batch_size: int = 64):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Networks
        self.network = MADDPGNetwork(n_agents, state_dim, action_dim, hidden_dims)
        self.target_network = MADDPGNetwork(n_agents, state_dim, action_dim, hidden_dims)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Optimizers
        self.actor_optimizers = [optim.Adam(self.network.actors[i].parameters(), lr=lr)
                                 for i in range(n_agents)]
        self.critic_optimizer = optim.Adam(self.network.critic.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = []
        self.buffer_size = buffer_size
    
    def select_actions(self, states: List[np.ndarray]) -> List[np.ndarray]:
        """
        Select actions for all agents
        
        Args:
            states: List of states for each agent
            
        Returns:
            List of actions
        """
        actions = []
        for state in states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.network.actors[0](state_tensor)
                actions.append(action.squeeze(0).numpy())
        return actions
    
    def store_experience(self, observations, actions, rewards, next_observations, done):
        """Store experience in buffer"""
        self.buffer.append((observations, actions, rewards, next_observations, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def sample_batch(self) -> dict:
        """Sample random batch from buffer"""
        indices = np.random.choice(len(self.buffer), self.batch_size)
        batch = [self.buffer[i] for i in indices]
        
        return {
            'observations': [torch.FloatTensor(np.array([e[0][i] for e in batch]))
                           for i in range(self.n_agents)],
            'actions': [torch.FloatTensor(np.array([e[1][i] for e in batch]))
                       for i in range(self.n_agents)],
            'rewards': [torch.FloatTensor(np.array([e[2][i] for e in batch]))
                       for i in range(self.n_agents)],
            'next_observations': [torch.FloatTensor(np.array([e[3][i] for e in batch]))
                                for i in range(self.n_agents)],
            'dones': torch.FloatTensor(np.array([e[4] for e in batch]))
        }
    
    def update_target_network(self):
        """Update target network using soft update"""
        for target_param, param in zip(self.target_network.parameters(),
                                       self.network.parameters()):
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
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']
        
        # Update critic
        with torch.no_grad():
            next_actions = self.target_network.get_actions(next_observations)
            next_q = self.target_network.get_q_values(next_observations, next_actions, 0)
            target_q = rewards[0].unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        q_values = self.network.get_q_values(observations, actions, 0)
        critic_loss = nn.functional.mse_loss(q_values, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actors
        for i in range(self.n_agents):
            obs = observations[i]
            action = self.network.actors[i](obs)
            q_value = self.network.get_q_values(observations, actions, i)
            actor_loss = -q_value.mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
        
        # Update target network
        self.update_target_network()
        
        return critic_loss.item()
    
    def train_episode(self, env: MultiAgentEnvironment, max_steps: int = 200) -> float:
        """
        Train for one episode
        
        Args:
            env: Environment
            max_steps: Maximum steps per episode
            
        Returns:
            Total reward for episode
        """
        states = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Select actions
            actions = self.select_actions(states)
            
            # Take actions
            next_states, rewards, done = env.step(actions)
            
            # Store experience
            self.store_experience(states, actions, rewards, next_states, done)
            
            # Train
            loss = self.train_step()
            
            # Update states
            states = next_states
            total_reward += sum(rewards)
            
            if done:
                break
        
        return total_reward
    
    def train(self, env: MultiAgentEnvironment, n_episodes: int = 500,
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
    print("Testing Multi-Agent Reinforcement Learning (MADDPG)...")
    print("=" * 50)
    
    # Create environment
    env = MultiAgentEnvironment(n_agents=2, state_dim=4, action_dim=2)
    
    # Create agent
    agent = MADDPGAgent(n_agents=2, state_dim=4, action_dim=2)
    
    # Train agent
    print("\nTraining agent...")
    rewards = agent.train(env, n_episodes=300, max_steps=200, verbose=True)
    
    # Test agent
    print("\nTesting trained agent...")
    states = env.reset()
    total_reward = 0
    
    for step in range(50):
        actions = agent.select_actions(states)
        next_states, rewards, done = env.step(actions)
        
        total_reward += sum(rewards)
        
        if done:
            print(f"Episode finished after {step + 1} steps")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print("\nMulti-Agent RL test completed successfully! ✓")
