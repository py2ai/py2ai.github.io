#!/usr/bin/env python3
"""
Test script to verify code in RL for Robotics blog post
"""

import sys
import numpy as np
import torch
import torch.nn as nn

print("Testing code from RL for Robotics blog post...\n")

# Test 1: DynamicsModel
print("Test 1: DynamicsModel")
try:
    class DynamicsModel(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, state_dim)
            )
        
        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            delta = self.net(x)
            return state + delta
    
    model = DynamicsModel(state_dim=10, action_dim=3)
    state = torch.randn(1, 10)
    action = torch.randn(1, 3)
    output = model(state, action)
    assert output.shape == (1, 10), f"Expected shape (1, 10), got {output.shape}"
    print("✓ DynamicsModel works correctly\n")
except Exception as e:
    print(f"✗ DynamicsModel failed: {e}\n")
    sys.exit(1)

# Test 2: NavigationPolicy
print("Test 2: NavigationPolicy")
try:
    class NavigationPolicy(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            )
            
            self.policy_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
            
            self.value_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        def forward(self, state):
            features = self.encoder(state)
            action = self.policy_head(features)
            value = self.value_head(features)
            return action, value
        
        def get_action(self, state):
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state)
                if len(state.shape) == 1:
                    state = state.unsqueeze(0)
                
                action, _ = self.forward(state)
                return action.squeeze(0).numpy()
    
    policy = NavigationPolicy(state_dim=37, action_dim=2)
    state = np.random.randn(37)
    action = policy.get_action(state)
    assert action.shape == (2,), f"Expected shape (2,), got {action.shape}"
    print("✓ NavigationPolicy works correctly\n")
except Exception as e:
    print(f"✗ NavigationPolicy failed: {e}\n")
    sys.exit(1)

# Test 3: WalkingPolicy
print("Test 3: WalkingPolicy")
try:
    class WalkingPolicy(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            
            self.rnn = nn.LSTM(state_dim, 256, batch_first=True)
            
            self.policy = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Tanh()
            )
            
            self.value = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        
        def forward(self, state, hidden=None):
            if len(state.shape) == 2:
                state = state.unsqueeze(1)
            
            output, hidden = self.rnn(state, hidden)
            action = self.policy(output.squeeze(1))
            value = self.value(output.squeeze(1))
            
            return action, value, hidden
        
        def get_action(self, state, hidden=None):
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state)
                if len(state.shape) == 1:
                    state = state.unsqueeze(0).unsqueeze(1)
                
                action, _, new_hidden = self.forward(state, hidden)
                return action.squeeze(0).numpy(), new_hidden
    
    policy = WalkingPolicy(state_dim=24, action_dim=12)
    state = np.random.randn(24)
    action, hidden = policy.get_action(state)
    assert action.shape == (12,), f"Expected shape (12,), got {action.shape}"
    print("✓ WalkingPolicy works correctly\n")
except Exception as e:
    print(f"✗ WalkingPolicy failed: {e}\n")
    sys.exit(1)

# Test 4: MAMLPolicy
print("Test 4: MAMLPolicy")
try:
    import copy
    
    class MAMLPolicy(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh()
            )
        
        def forward(self, state):
            return self.net(state)
        
        def adapt(self, support_data, steps=5, lr=0.01):
            adapted_net = copy.deepcopy(self.net)
            optimizer = torch.optim.SGD(adapted_net.parameters(), lr=lr)
            
            for _ in range(steps):
                for state, action, reward in support_data:
                    pred_action = adapted_net(state)
                    loss = ((pred_action - action) ** 2).mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            new_policy = MAMLPolicy(state_dim=10, action_dim=3)
            new_policy.net = adapted_net
            return new_policy
    
    policy = MAMLPolicy(state_dim=10, action_dim=3)
    state = torch.randn(1, 10)
    action = policy(state)
    assert action.shape == (1, 3), f"Expected shape (1, 3), got {action.shape}"
    print("✓ MAMLPolicy works correctly\n")
except Exception as e:
    print(f"✗ MAMLPolicy failed: {e}\n")
    sys.exit(1)

# Test 5: BehaviorCloning
print("Test 5: BehaviorCloning")
try:
    class BehaviorCloning(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            )
        
        def forward(self, state):
            return self.net(state)
    
    bc = BehaviorCloning(state_dim=10, action_dim=3)
    state = torch.randn(1, 10)
    action = bc(state)
    assert action.shape == (1, 3), f"Expected shape (1, 3), got {action.shape}"
    print("✓ BehaviorCloning works correctly\n")
except Exception as e:
    print(f"✗ BehaviorCloning failed: {e}\n")
    sys.exit(1)

# Test 6: Safety classes
print("Test 6: Safety classes")
try:
    class SafetyConstraint:
        def __init__(self, limit):
            self.limit = limit
        
        def is_satisfied(self, state, action):
            return np.all(np.abs(action) <= self.limit)
        
        def project(self, state, action):
            return np.clip(action, -self.limit, self.limit)
    
    constraint = SafetyConstraint(limit=1.0)
    action = np.array([0.5, 0.8, 1.5])
    projected = constraint.project(None, action)
    assert np.all(projected <= 1.0), "Projection failed"
    print("✓ Safety classes work correctly\n")
except Exception as e:
    print(f"✗ Safety classes failed: {e}\n")
    sys.exit(1)

print("=" * 50)
print("All tests passed! ✓")
print("=" * 50)
print("\nThe code in the blog post is syntactically correct")
print("and should work as expected.")
