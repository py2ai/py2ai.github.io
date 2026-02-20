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
    
    # Test predict method
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
                # Extract next_state from tuple (next_state, reward, done)
                predictions.append(pred[0] if isinstance(pred, tuple) else pred)
            # Stack predictions
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
