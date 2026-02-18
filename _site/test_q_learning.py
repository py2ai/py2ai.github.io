"""
Test script for Q-Learning from Scratch
"""
import numpy as np
import random
from typing import Tuple, List

class GridWorldEnvironment:
    """
    Simple Grid World Environment for Q-Learning
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        start_pos: Starting position (row, col)
        goal_pos: Goal position (row, col)
        obstacles: List of obstacle positions
    """
    def __init__(self, grid_size: int = 4, start_pos: Tuple[int, int] = (0, 0),
                 goal_pos: Tuple[int, int] = (3, 3), obstacles: List[Tuple[int, int]] = None):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles if obstacles else []
        self.current_pos = start_pos
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    
    def reset(self) -> Tuple[int, int]:
        """Reset environment to starting position"""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take action in environment
        
        Args:
            action: Action index (0: Right, 1: Left, 2: Down, 3: Up)
            
        Returns:
            (next_state, reward, done)
        """
        # Get action
        row_action, col_action = self.actions[action]
        
        # Calculate new position
        new_row = self.current_pos[0] + row_action
        new_col = self.current_pos[1] + col_action
        
        # Check bounds
        if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
            # Check obstacles
            if (new_row, new_col) not in self.obstacles:
                self.current_pos = (new_row, new_col)
        
        # Calculate reward
        if self.current_pos == self.goal_pos:
            reward = 10.0
            done = True
        else:
            reward = -0.1
            done = False
        
        return self.current_pos, reward, done
    
    def get_state_index(self, state: Tuple[int, int]) -> int:
        """Convert state to index"""
        return state[0] * self.grid_size + state[1]
    
    def render(self):
        """Render the grid"""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) == self.current_pos:
                    print("A", end=" ")  # Agent
                elif (row, col) == self.goal_pos:
                    print("G", end=" ")  # Goal
                elif (row, col) in self.obstacles:
                    print("X", end=" ")  # Obstacle
                else:
                    print(".", end=" ")  # Empty
            print()
        print()

class QLearningAgent:
    """
    Q-Learning Agent
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        learning_rate: Learning rate (alpha)
        discount_factor: Discount factor (gamma)
        exploration_rate: Initial exploration rate (epsilon)
        exploration_decay: Exploration decay rate
        min_exploration: Minimum exploration rate
    """
    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 0.995,
                 min_exploration: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Initialize Q-table
        self.q_table = np.zeros((state_dim, action_dim))
    
    def select_action(self, state: int) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: int, action: int, reward: float,
                      next_state: int, done: bool):
        """
        Update Q-value using Q-learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Q-learning update
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
    
    def decay_exploration(self):
        """Decay exploration rate"""
        self.exploration_rate = max(self.min_exploration,
                                   self.exploration_rate * self.exploration_decay)
    
    def train_episode(self, env: GridWorldEnvironment, max_steps: int = 100) -> float:
        """
        Train for one episode
        
        Args:
            env: Environment
            max_steps: Maximum steps per episode
            
        Returns:
            Total reward for episode
        """
        state = env.get_state_index(env.reset())
        total_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state)
            
            # Take action
            next_state_pos, reward, done = env.step(action)
            next_state = env.get_state_index(next_state_pos)
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        self.decay_exploration()
        return total_reward
    
    def train(self, env: GridWorldEnvironment, n_episodes: int = 1000,
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
        
        for episode in range(n_episodes):
            reward = self.train_episode(env, max_steps)
            rewards.append(reward)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.2f}, "
                      f"Epsilon: {self.exploration_rate:.3f}")
        
        return rewards

# Test the code
if __name__ == "__main__":
    print("Testing Q-Learning from Scratch...")
    print("=" * 50)
    
    # Create environment
    env = GridWorldEnvironment(grid_size=4, start_pos=(0, 0), goal_pos=(3, 3))
    
    # Create agent
    agent = QLearningAgent(state_dim=16, action_dim=4)
    
    # Train agent
    print("\nTraining agent...")
    rewards = agent.train(env, n_episodes=500, max_steps=100, verbose=True)
    
    # Test agent
    print("\nTesting trained agent...")
    state = env.get_state_index(env.reset())
    env.render()
    
    for step in range(20):
        action = agent.select_action(state)
        next_state_pos, reward, done = env.step(action)
        next_state = env.get_state_index(next_state_pos)
        
        print(f"Step {step + 1}: Action {action}, Reward {reward:.2f}")
        env.render()
        
        state = next_state
        
        if done:
            print("Goal reached!")
            break
    
    print("\nQ-Learning test completed successfully! ✓")
