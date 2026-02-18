---
layout: post
title: "Part 1: Introduction to Reinforcement Learning - Core Concepts and Fundamentals"
date: 2026-02-01
categories: [Machine Learning, AI, Python, Deep RL]
featured-img: 2026-feb-deeprl/2026-feb-deeprl
description: "Learn fundamentals of Reinforcement Learning from scratch. Understand agents, environments, rewards, and RL loop with practical examples and mathematical foundations."
---
# Part 1: Introduction to Reinforcement Learning - Core Concepts and Fundamentals

Welcome to first post in our **Deep Reinforcement Learning Series**! In this comprehensive guide, we'll explore fundamental concepts of Reinforcement Learning (RL), understand how it differs from other machine learning approaches, and build a solid foundation for advanced topics.

## What is Reinforcement Learning?

Reinforcement Learning is a subfield of machine learning where an **agent** learns to make decisions by interacting with an **environment**. Unlike supervised learning, where model learns from labeled data, RL agents learn through **trial and error**, receiving **rewards** or **penalties** based on their actions.

### Key Characteristics

**Learning by Interaction:**

- Agent takes actions in an environment
- Environment responds with new states and rewards
- Agent learns from experience
- No labeled training data required

**Goal-Oriented:**

- Agent aims to maximize cumulative reward
- Long-term planning is essential
- Trade-off between immediate and future rewards

**Sequential Decision Making:**

- Actions affect future states
- Current decisions impact future possibilities
- Requires planning and strategy

## The RL Framework

### Core Components

#### 1. **Agent**

The learner and decision-maker that:

- Observes environment
- Takes actions based on policy
- Learns from experience
- Aims to maximize rewards

#### 2. **Environment**

The world agent interacts with:

- Provides observations/states
- Gives rewards/penalties
- Transitions to new states
- Follows (often unknown) dynamics

#### 3. **State (S)**

Complete information about situation:

- Current configuration of environment
- Everything agent needs to know
- Can be fully or partially observable

**Mathematical Notation:**

$$S_t$$ represents state at time step $$t$$

#### 4. **Action (A)**
What the agent can do:
- Set of all possible moves
- Can be discrete or continuous
- Determined by agent's policy

**Mathematical Notation:**
$$A_t$$ represents state at time step $$t$$

#### 5. **Reward (R)**
Feedback signal from environment:
- Positive for good actions
- Negative for bad actions
- Guides learning process

**Mathematical Notation:**
$$R_t$$ represents state at time step $$t$$

#### 6. **Policy (π)**
The agent's strategy:
- Maps states to actions
- Determines agent's behavior
- Can be deterministic or stochastic

**Mathematical Notation:**
$$\pi(a|s) = P(A_t = a | S_t = s)
$$

This represents the probability of taking action $$a$$ given sate $$s$$.

## The Reinforcement Learning Loop

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Learning Cycle                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Agent observes state: S_t                               │
│     ↓                                                       │
│  2. Agent selects action: A_t based on policy π             │
│     ↓                                                       │
│  3. Environment executes action A_t                         │
│     ↓                                                       │
│  4. Environment transitions to new state: S_{t+1}           │
│     ↓                                                       │
│  5. Environment gives reward: R_{t+1}                       │
│     ↓                                                       │
│  6. Agent updates policy as   (S_t, A_t, R_{t+1}, S_{t+1})  │
│     ↓                                                       │
│  7. Repeat from step 1                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Mathematical Formulation

At each time step $$t$$:

1. **Observe State:**

$$
S_t \sim P(S_0)
$$

2. **Select Action:**

$$
A_t \sim \pi(\cdot|S_t)
$$

3. **Receive Reward and Next State:**

$$
S_{t+1} \sim P(\cdot|S_t, A_t)
$$

$$
R_{t+1} = R(S_t, A_t, S_{t+1})
$$

4. **Update Policy:**

$$
\pi \leftarrow \pi + \alpha \nabla_\pi J(\pi)
$$

## Types of Reinforcement Learning

### 1. **Model-Based vs Model-Free**

**Model-Based RL:**

- Learns environment dynamics
- Can simulate future states
- More sample efficient
- Harder to implement

**Model-Free RL:**

- Learns directly from experience
- No environment model needed
- Simpler to implement
- Requires more samples

### 2. **Value-Based vs Policy-Based**

**Value-Based:**

- Learns value of states/actions
- Policy derived from values
- Examples: Q-Learning, DQN
- Good for discrete actions

**Policy-Based:**

- Learns policy directly
- Can handle continuous actions
- Examples: REINFORCE, PPO
- More stable convergence

### 3. **On-Policy vs Off-Policy**

**On-Policy:**

- Learns from current policy's actions
- Updates policy with current behavior
- Examples: SARSA, PPO
- More stable but slower

**Off-Policy:**

- Learns from any experience
- Can reuse old data
- Examples: Q-Learning, DQN
- More sample efficient

### 4. **Episodic vs Continuing**

**Episodic Tasks:**

- Have natural ending points
- Episodes are independent
- Examples: Games, puzzles
- Clear success/failure

**Continuing Tasks:**

- No natural termination
- Go on indefinitely
- Examples: Robot control, trading
- Focus on long-term performance

## Simple Example: Grid World

Let's understand RL with a classic Grid World problem.

### Problem Setup

```
┌─────┬─────┬─────┬─────┐
│  S  │     │     │  G  │
│(0,0)│     │     │(0,3)│
├─────┼─────┼─────┼─────┤
│     │  X  │     │     │
│     │(1,1)│     │     │
├─────┼─────┼─────┼─────┤
│     │     │     │     │
│     │     │     │     │
├─────┼─────┼─────┼─────┤
│     │     │     │     │
│     │     │     │     │
└─────┴─────┴─────┴─────┘

S = Start (0,0)
G = Goal (0,3)
X = Obstacle (1,1)
```

### RL Elements

**State Space:**

$$
\mathcal{S} = \{(x,y) | x \in \{0,1,2,3\}, y \in \{0,1,2,3\}\}
$$

**Action Space:**

$$
\mathcal{A} = \{\text{up}, \text{down}, \text{left}, \text{right}\}
$$

**Reward Function:**

$$
R(s,a,s') = \begin{cases}
+10 & \text{if } s' = \text{goal} \\
-1 & \text{if } s' = \text{obstacle} \\
0 & \text{otherwise}
\end{cases}
$$

**Goal:** Find optimal policy to reach goal while avoiding obstacles.

## Python Implementation

### Basic RL Agent Structure

```python
import numpy as np
from typing import Tuple, List

class RLAgent:
    def __init__(self, state_space: int, action_space: int):
        """
        Initialize RL agent

        Args:
            state_space: Number of possible states
            action_space: Number of possible actions
        """
        self.state_space = state_space
        self.action_space = action_space

        # Q-table: stores value of state-action pairs
        self.q_table = np.zeros((state_space, action_space))

        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration = 0.01

    def select_action(self, state: int) -> int:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state

        Returns:
            Selected action
        """
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_space)

        # Exploitation: best action from Q-table
        return np.argmax(self.q_table[state])

    def learn(self, state: int, action: int, reward: float,
             next_state: int, done: bool):
        """
        Update Q-table using Q-learning update rule

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])

        # Bellman equation
        target = reward + self.discount_factor * \
                 self.q_table[next_state][best_next_action] * (1 - done)

        # Update Q-value
        self.q_table[state][action] += self.learning_rate * \
                                      (target - self.q_table[state][action])

        # Decay exploration
        self.exploration_rate = max(self.min_exploration,
                                   self.exploration_rate * self.exploration_decay)
```

### Grid World Environment

```python
class GridWorld:
    def __init__(self, width: int = 4, height: int = 4):
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.goal = (0, 3)
        self.obstacles = [(1, 1)]

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        self.reset()

    def reset(self) -> Tuple[int, int]:
        """Reset environment to start state"""
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute action in environment

        Args:
            action: Action to take

        Returns:
            next_state, reward, done
        """
        x, y = self.state
        dx, dy = self.actions[action]

        # Calculate new position
        new_x = max(0, min(self.width - 1, x + dx))
        new_y = max(0, min(self.height - 1, y + dy))
        new_state = (new_x, new_y)

        # Check if hit obstacle
        if new_state in self.obstacles:
            reward = -1
            done = False
            new_state = self.state  # Stay in place

        # Check if reached goal
        elif new_state == self.goal:
            reward = 10
            done = True

        # Normal move
        else:
            reward = 0
            done = False

        self.state = new_state
        return new_state, reward, done

    def render(self):
        """Display current state"""
        grid = [['.' for _ in range(self.width)]
                for _ in range(self.height)]

        # Mark positions
        grid[self.start[1]][self.start[0]] = 'S'
        grid[self.goal[1]][self.goal[0]] = 'G'

        for obs in self.obstacles:
            grid[obs[1]][obs[0]] = 'X'

        grid[self.state[1]][self.state[0]] = 'A'

        for row in reversed(grid):
            print(' '.join(row))
        print()
```

### Training Loop

```python
def train_agent(episodes: int = 1000):
    """Train RL agent on Grid World"""

    # Initialize environment and agent
    env = GridWorld()
    agent = RLAgent(state_space=16, action_space=4)

    rewards = []

    for episode in range(episodes):
        state = env.reset()
        state_idx = state[0] * 4 + state[1]  # Convert to index
        total_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state_idx)

            # Execute action
            next_state, reward, done = env.step(action)
            next_state_idx = next_state[0] * 4 + next_state[1]

            # Learn from experience
            agent.learn(state_idx, action, reward,
                     next_state_idx, done)

            state_idx = next_state_idx
            total_reward += reward

        rewards.append(total_reward)

        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Exploration: {agent.exploration_rate:.2f}")

    return agent, rewards

# Train agent
agent, rewards = train_agent(episodes=1000)

# Test trained agent
print("\nTesting trained agent:")
env = GridWorld()
state = env.reset()
state_idx = state[0] * 4 + state[1]
done = False

while not done:
    env.render()
    action = agent.select_action(state_idx)
    next_state, reward, done = env.step(action)
    next_state_idx = next_state[0] * 4 + next_state[1]
    state_idx = next_state_idx
```

## Key RL Concepts

### 1. **Exploration vs Exploitation**

**Exploration:** Trying new actions to discover better strategies

- Essential for learning
- Reduces over time
- Balances with exploitation

**Exploitation:** Using known best actions

- Maximizes immediate reward
- Can get stuck in local optima
- Increases over time

**Epsilon-Greedy Strategy:**

$$
\pi(a|s) = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s,a) & \text{with probability } 1-\epsilon
\end{cases}
$$

### 2. **Discount Factor (γ)**

Determines importance of future rewards:

- $$\gamma \approx 0$$: Myopic (focus on immediate rewards)
- $$\gamma \approx 1$$: Farsighted (consider long-term rewards)
- Typical values: 0.9 to 0.99

**Discounted Return:**

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

### 3. **Credit Assignment Problem**

Determining which actions deserve credit for rewards:

- Delayed rewards make learning difficult
- Requires temporal credit assignment
- Solved by value functions and TD learning

### 4. **State Representation**

**Fully Observable:**

- Agent sees complete state
- Markov property holds
- Easier to solve

**Partially Observable:**

- Agent sees limited information
- Requires belief states
- More challenging

## RL vs Other ML Approaches

### Supervised Learning

- **Learning:** From labeled examples
- **Feedback:** Immediate, correct answer
- **Goal:** Predict labels
- **Examples:** Classification, regression

### Unsupervised Learning

- **Learning:** From unlabeled data
- **Feedback:** None (internal structure)
- **Goal:** Discover patterns
- **Examples:** Clustering, dimensionality reduction

### Reinforcement Learning

- **Learning:** From interaction
- **Feedback:** Delayed rewards
- **Goal:** Maximize cumulative reward
- **Examples:** Games, robotics, control

## Real-World Applications

### 1. **Game Playing**

- Chess, Go, Atari games
- AlphaGo, AlphaZero
- Superhuman performance

### 2. **Robotics**

- Walking robots
- Manipulation tasks
- Autonomous navigation

### 3. **Finance**

- Algorithmic trading
- Portfolio optimization
- Risk management

### 4. **Healthcare**

- Treatment optimization
- Drug discovery
- Personalized medicine

### 5. **Transportation**

- Self-driving cars
- Traffic light control
- Route optimization

### 6. **Recommendation Systems**

- Content recommendations
- Ad placement
- User engagement

## Common Challenges

### 1. **Sparse Rewards**

- Rewards are rare
- Learning is slow
- Solution: Reward shaping

### 2. **Credit Assignment**

- Hard to link actions to rewards
- Delayed consequences
- Solution: Temporal difference learning

### 3. **Exploration**

- Finding optimal policy takes time
- Can get stuck locally
- Solution: Curiosity, intrinsic motivation

### 4. **Sample Efficiency**

- Requires many interactions
- Expensive in real world
- Solution: Model-based RL, transfer learning

### 5. **Stability**

- Training can be unstable
- Hyperparameter sensitivity
- Solution: Target networks, experience replay

## What's Next?

In next post, we'll dive deep into **Markov Decision Processes (MDPs)** - the mathematical foundation of reinforcement learning. We'll cover:

- Formal definition of MDPs
- Transition probabilities
- Reward functions
- Value functions
- Bellman equations
- Solving MDPs

## Key Takeaways

- **RL learns through interaction** with an environment
- **Agent maximizes cumulative reward** over time
- **Policy maps states to actions**
- **Exploration vs exploitation** is crucial
- **Discount factor** balances immediate and future rewards
- **RL differs** from supervised and unsupervised learning
- **Real-world applications** span many domains

## Practice Exercises

1. **Modify the Grid World** to add more obstacles
2. **Experiment with different learning rates** and discount factors
3. **Try different exploration strategies** (softmax, UCB)
4. **Implement a different reward function** and observe behavior
5. **Visualize learning curve** with matplotlib

## Questions?

Have questions about RL fundamentals? Drop them in the comments below!

**Next Post:** [Part 2: Markov Decision Processes Explained]({{ site.baseurl }}{% post_url 2026-02-02-Markov-Decision-Processes-Explained %})

**Series Index:** [Deep Reinforcement Learning Series Roadmap]({{ site.baseurl }}{% post_url 2026-02-01-Deep-RL-Series-Roadmap %})
