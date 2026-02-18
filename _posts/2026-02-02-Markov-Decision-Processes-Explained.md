---
layout: post
title: "Part 2: Markov Decision Processes Explained - Mathematical Foundation of RL"
date: 2026-02-02
categories: [Machine Learning, AI, Python, Deep RL]
featured-img: 2026-feb-deeprl/2026-feb-deeprl
description: "Deep dive into Markov Decision Processes (MDPs) - the mathematical foundation of Reinforcement Learning. Learn states, actions, transitions, rewards, and Bellman equations."
---

# Part 2: Markov Decision Processes Explained - Mathematical Foundation of RL

Welcome to the second post in our **Deep Reinforcement Learning Series**! In this comprehensive guide, we'll explore **Markov Decision Processes (MDPs)** - the mathematical framework that formalizes reinforcement learning problems. Understanding MDPs is crucial for grasping the theoretical foundations of RL.

##  What is a Markov Decision Process?

A **Markov Decision Process (MDP)** is a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. MDPs provide a formal way to describe the environment in which reinforcement learning agents operate.

### Key Properties

**Markov Property:**
The future is independent of the past given the present:
$$P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, \dots) = P(S_{t+1} | S_t, A_t)$$

This means the current state contains all information needed to make optimal decisions.

**Stationary Transitions:**
Transition dynamics don't change over time:
$$P(S_{t+1} | S_t, A_t) = P(S_{t+1} | S_t, A_t) \quad \forall t$$

##  Formal Definition of MDP

An MDP is defined by the tuple:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

### 1. **State Space (S)**

The set of all possible states the environment can be in:

$$\mathcal{S} = \{s_1, s_2, \dots, s_n\}$$

**Types of State Spaces:**

**Discrete State Space:**
- Finite number of states
- Examples: Chess board positions, Grid World
- Easier to solve analytically

**Continuous State Space:**
- Infinite number of states
- Examples: Robot joint angles, Stock prices
- Requires function approximation

**Example - Grid World:**
$$\mathcal{S} = \{(x,y) | x \in \{0,1,2,3\}, y \in \{0,1,2,3\}\}$$

### 2. **Action Space (A)**

The set of all possible actions the agent can take:

$$\mathcal{A} = \{a_1, a_2, \dots, a_m\}$$

**Types of Action Spaces:**

**Discrete Action Space:**
- Finite number of actions
- Examples: Move up/down/left/right, Press button
- Suitable for value-based methods

**Continuous Action Space:**
- Infinite action possibilities
- Examples: Robot motor torques, Steering angle
- Requires policy-based methods

**Example - Grid World:**
$$\mathcal{A} = \{\text{up}, \text{down}, \text{left}, \text{right}\}$$

### 3. **Transition Function (P)**

The probability of transitioning to sate $$s'$$ given current sate $$s$$ and action $$a$$:

$$\mathcal{P}(s' | s, a) = P(S_{t+1} = s' | S_t = s, A_t = a)$$

**Properties:**
- **Stochastic:** $$0 \leq P(s'|s,a) \leq 1$$
- **Normalized:** $$\sum_{s' \in \mathcal{S}} P(s'|s,a) = 1$$
**Example - Deterministic Transitions:**

$$P(s'|s,a) = \begin{cases}
1 & \text{if } s' = f(s,a) \\
0 & \text{otherwise}
\end{cases}$$

Where $$f(s,a)$$ is the deterministic transition function.

**Example - Stochastic Transitions:**
```python
# 80% chance to move in intended direction
# 20% chance to move in random direction
P(s'|s,a) = {
    intended_direction: 0.8,
    other_directions: 0.2/3
}
```

### 4. **Reward Function (R)**

The expected immediate reward after taking action $$a$$ in sate $$s$$ and transitioning to $$s'$$:

$$\mathcal{R}(s, a, s') = \mathbb{E}[R_{t+1} | S_t = s, A_t = a, S_{t+1} = s']$$

**Types of Rewards:**

**Deterministic Rewards:**
$$R(s,a,s') = r \quad \text{(fixed value)}$$

**Stochastic Rewards:**
$$R(s,a,s') = \mathbb{E}[R | s,a,s']$$

**Example - Grid World:**
$$R(s,a,s') = \begin{cases}
+10 & \text{if } s' = \text{goal} \\
-1 & \text{if } s' = \text{obstacle} \\
-0.1 & \text{otherwise (time penalty)}
\end{cases}$$

### 5. **Discount Factor (γ)**

A factor $$\gamma \in [0,1]$$ that determines the importance of future rewards:

**Interpretation:**
- $$\gamma \approx 0$$: Agent is myopic (focus on immediate rewards)
- $$\gamma \approx 1$$: Agent is farsighted (considers long-term rewards)
- $$\gamma = 1$$: Values all future rewards equally

**Typical Values:**
- $$\gamma = 0.9$$: Moderate discounting
- $$\gamma = 0.95$$: Strong long-term consideration
- $$\gamma = 0.99$$: Very long-term planning

##  The MDP Dynamics

### Episode Trajectory

A sequence of states, actions, and rewards:

$$\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_T)$$

### Return (Cumulative Reward)

The sum of discounted rewards from time step $$t$$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**Finite Horizon (Episodic):**
$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

**Infinite Horizon (Continuing):**
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

##  Value Functions

### 1. **State-Value Function (V)**

The expected return starting from sate $$s$$ and following policy $$\pi$$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t | S_t = s \right]$$

**Bellman Expectation Equation:**

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

**Optimal State-Value Function:**

$$V^*(s) = \max_\pi V^\pi(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

### 2. **Action-Value Function (Q)**

The expected return starting from sate $$s$$, taking action $$a$$, and then following policy $$\pi$$:

$$Q^\pi(s,a) = \mathbb{E}_\pi \left[ G_t | S_t = s, A_t = a \right]$$

**Bellman Expectation Equation:**

$$Q^\pi(s,a) = \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^\pi(s',a') \right]$$

**Optimal Action-Value Function:**

$$Q^*(s,a) = \max_\pi Q^\pi(s,a) = \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s',a') \right]$$

### 3. **Relationship Between V and Q**

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q^\pi(s,a)$$

$$Q^\pi(s,a) = \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

##  Policies

### Definition

A policy $$\pi$$ is a mapping from states to actions:

$$\pi: \mathcal{S} \rightarrow \mathcal{A}$$

### Types of Policies

**Deterministic Policy:**
$$\pi(s) = a \quad \text{(single action)}$$

**Stochastic Policy:**
$$\pi(a|s) = P(A_t = a | S_t = s) \quad \text{(probability distribution)}$$

**Optimal Policy:**

$$\pi^*(s) = \arg\max_{a \in \mathcal{A}} Q^*(s,a)$$

##  Bellman Equations

### Bellman Expectation Equation for V

$$V^\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s \right]$$

**Expanded Form:**

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

### Bellman Optimality Equation for V

$$V^*(s) = \max_{a \in \mathcal{A}} \mathbb{E} \left[ R_{t+1} + \gamma V^*(S_{t+1}) | S_t = s, A_t = a \right]$$

**Expanded Form:**

$$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

### Bellman Expectation Equation for Q

$$Q^\pi(s,a) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma Q^\pi(S_{t+1}, A_{t+1}) | S_t = s, A_t = a \right]$$

**Expanded Form:**

$$Q^\pi(s,a) = \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^\pi(s',a') \right]$$

### Bellman Optimality Equation for Q

$$Q^*(s,a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a' \in \mathcal{A}} Q^*(S_{t+1}, a') | S_t = s, A_t = a \right]$$

**Expanded Form:**

$$Q^*(s,a) = \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s',a') \right]$$

##  Python Implementation

### MDP Class

```python
import numpy as np
from typing import Tuple, List, Dict

class MDP:
    """
    Markov Decision Process implementation
    
    Args:
        states: List of all possible states
        actions: List of all possible actions
        transitions: Transition function P(s'|s,a)
        rewards: Reward function R(s,a,s')
        gamma: Discount factor
    """
    def __init__(self, 
                 states: List, 
                 actions: List,
                 transitions: Dict,
                 rewards: Dict,
                 gamma: float = 0.95):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
        self.n_states = len(states)
        self.n_actions = len(actions)
    
    def get_transition_prob(self, state: int, action: int, 
                          next_state: int) -> float:
        """
        Get transition probability P(s'|s,a)
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            Transition probability
        """
        return self.transitions.get((state, action, next_state), 0.0)
    
    def get_reward(self, state: int, action: int, 
                 next_state: int) -> float:
        """
        Get expected reward R(s,a,s')
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            Expected reward
        """
        return self.rewards.get((state, action, next_state), 0.0)
    
    def get_expected_reward(self, state: int, action: int) -> float:
        """
        Get expected reward for state-action pair
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Expected reward
        """
        expected_reward = 0.0
        for next_state in self.states:
            prob = self.get_transition_prob(state, action, next_state)
            reward = self.get_reward(state, action, next_state)
            expected_reward += prob * reward
        return expected_reward
```

### Value Iteration Algorithm

```python
def value_iteration(mdp: MDP, theta: float = 1e-6, 
                 max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve MDP using Value Iteration
    
    Args:
        mdp: Markov Decision Process
        theta: Convergence threshold
        max_iterations: Maximum iterations
        
    Returns:
        (optimal_values, optimal_policy)
    """
    # Initialize value function
    V = np.zeros(mdp.n_states)
    
    for iteration in range(max_iterations):
        delta = 0.0
        V_new = np.zeros(mdp.n_states)
        
        # Update each state
        for state in mdp.states:
            # Bellman optimality equation
            max_value = -np.inf
            for action in mdp.actions:
                # Expected value for this action
                action_value = mdp.get_expected_reward(state, action)
                
                # Add discounted future value
                for next_state in mdp.states:
                    prob = mdp.get_transition_prob(state, action, next_state)
                    action_value += prob * mdp.gamma * V[next_state]
                
                max_value = max(max_value, action_value)
            
            V_new[state] = max_value
            delta = max(delta, abs(V_new[state] - V[state]))
        
        V = V_new
        
        # Check convergence
        if delta < theta:
            print(f"Converged in {iteration + 1} iterations")
            break
    
    # Extract optimal policy
    policy = np.zeros(mdp.n_states, dtype=int)
    for state in mdp.states:
        max_value = -np.inf
        best_action = 0
        
        for action in mdp.actions:
            action_value = mdp.get_expected_reward(state, action)
            
            for next_state in mdp.states:
                prob = mdp.get_transition_prob(state, action, next_state)
                action_value += prob * mdp.gamma * V[next_state]
            
            if action_value > max_value:
                max_value = action_value
                best_action = action
        
        policy[state] = best_action
    
    return V, policy
```

### Policy Iteration Algorithm

```python
def policy_iteration(mdp: MDP, max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve MDP using Policy Iteration
    
    Args:
        mdp: Markov Decision Process
        max_iterations: Maximum iterations
        
    Returns:
        (optimal_values, optimal_policy)
    """
    # Initialize random policy
    policy = np.random.randint(0, mdp.n_actions, size=mdp.n_states)
    V = np.zeros(mdp.n_states)
    
    for iteration in range(max_iterations):
        # Policy Evaluation
        policy_stable = True
        
        for _ in range(100):  # Fixed iterations for evaluation
            V_new = np.zeros(mdp.n_states)
            
            for state in mdp.states:
                action = policy[state]
                
                # Bellman expectation equation
                value = mdp.get_expected_reward(state, action)
                
                for next_state in mdp.states:
                    prob = mdp.get_transition_prob(state, action, next_state)
                    value += prob * mdp.gamma * V[next_state]
                
                V_new[state] = value
            
            V = V_new
        
        # Policy Improvement
        for state in mdp.states:
            old_action = policy[state]
            max_value = -np.inf
            best_action = 0
            
            for action in mdp.actions:
                action_value = mdp.get_expected_reward(state, action)
                
                for next_state in mdp.states:
                    prob = mdp.get_transition_prob(state, action, next_state)
                    action_value += prob * mdp.gamma * V[next_state]
                
                if action_value > max_value:
                    max_value = action_value
                    best_action = action
            
            policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        if policy_stable:
            print(f"Converged in {iteration + 1} iterations")
            break
    
    return V, policy
```

### Example: Grid World MDP

```python
def create_grid_world_mdp() -> MDP:
    """
    Create a 4x4 Grid World MDP
    
    Returns:
        MDP instance
    """
    # States: 16 positions (4x4 grid)
    states = list(range(16))
    
    # Actions: 0=up, 1=down, 2=left, 3=right
    actions = list(range(4))
    
    # Define transitions and rewards
    transitions = {}
    rewards = {}
    
    for state in states:
        x, y = state // 4, state % 4
        
        for action in actions:
            # Calculate next position
            if action == 0:  # up
                next_x, next_y = x, min(3, y + 1)
            elif action == 1:  # down
                next_x, next_y = x, max(0, y - 1)
            elif action == 2:  # left
                next_x, next_y = max(0, x - 1), y
            else:  # right
                next_x, next_y = min(3, x + 1), y
            
            next_state = next_x * 4 + next_y
            
            # Deterministic transition
            transitions[(state, action, next_state)] = 1.0
            
            # Define rewards
            if next_state == 3:  # Goal at (0,3)
                rewards[(state, action, next_state)] = 10.0
            elif next_state == 5:  # Obstacle at (1,1)
                rewards[(state, action, next_state)] = -1.0
            else:
                rewards[(state, action, next_state)] = -0.1  # Time penalty
    
    return MDP(states, actions, transitions, rewards, gamma=0.95)

# Create and solve MDP
mdp = create_grid_world_mdp()

# Solve using Value Iteration
V, policy = value_iteration(mdp)

print("Optimal Values:")
print(V.reshape(4, 4))

print("\nOptimal Policy:")
print(policy.reshape(4, 4))

# Visualize policy
action_symbols = ['↑', '↓', '←', '→']
policy_grid = np.array([[action_symbols[a] for a in row] for row in policy.reshape(4, 4)])

print("\nPolicy Visualization:")
for row in reversed(policy_grid):
    print(' '.join(row))
```

##  Solving MDPs

### Dynamic Programming Methods

**Value Iteration:**
- Iteratively improves value function
- Converges to optimal values
- Guarantees optimality

**Policy Iteration:**
- Alternates between evaluation and improvement
- Often faster convergence
- Guarantees optimality

### Linear Programming

Formulate as linear program:
- Variables: State values
- Constraints: Bellman equations
- Objective: Maximize expected return

### Monte Carlo Methods

Estimate values through sampling:
- No model required
- Model-free approach
- Converges with enough samples

### Temporal Difference Learning

Combine ideas from DP and MC:
- Learn from experience
- Online learning
- Basis for Q-learning

##  Example: Small MDP

### Problem Setup

Consider a simple 3-state MDP:

```
    ┌─────────┐
    │   S0    │
    │ (Start) │
    └────┬────┘
         │
    ┌────┴────┐
    │         │
  S1        S2
 (Good)    (Bad)
```

**States:** $$\mathcal{S} = \{S_0, S_1, S_2\}$$

**Actions:** $$\mathcal{A} = \{\text{left}, \text{right}\}$$

**Transitions:**
- From $$S_0$$: 50% to $$S_1$$, 50% to $$S_2$$
- From $$S_1$$: Stay in $$S_1$$ (terminal)
- From $$S_2$$: Stay in $$S_2$$ (terminal)

**Rewards:**
- $$R(S_0, \cdot, S_1) = +1$$
- $$R(S_0, \cdot, S_2) = -1$$
- $$R(S_1, \cdot, S_1) = 0$$
- $$R(S_2, \cdot, S_2) = 0$$

**Discount:** $$\gamma = 0.9$$

### Solving with Bellman Equations

**Value Function:**

$$V(S_0) = 0.5 \times [1 + 0.9 \times V(S_1)] + 0.5 \times [-1 + 0.9 \times V(S_2)]$$
$$V(S_1) = 0$$
$$V(S_2) = 0$$

**Substituting:**

$$V(S_0) = 0.5 \times [1 + 0.9 \times 0] + 0.5 \times [-1 + 0.9 \times 0]$$
$$V(S_0) = 0.5 \times 1 + 0.5 \times (-1)$$
$$V(S_0) = 0$**Optimal Policy:**$\pi^*(S_0) = \text{left} \quad \text{(both actions give same value)}$$

**Optimal Policy:**
$$\pi^*(S_0) = \text{left} \quad \text{(both actions give same value)}$$

## Partially Observable MDPs (POMDPs)

### Definition

When the agent cannot fully observe the state:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{P}, \mathcal{R}, \mathcal{Z}, \gamma)$$

Where:
- $\mathcal{O}$ - Observation space
- $\mathcal{Z}$ - Observation function $P(o|s,a,s')$

### Belief State

Agent maintains probability distribution over states:

$$b_t(s) = P(S_t = s | o_0, a_0, o_1, a_1, \dots, o_t)$$

### Solving POMDPs

- **Belief state MDP**: Convert to MDP over beliefs
- **Point-based value iteration**: Approximate solution
- **Monte Carlo methods**: Sample-based approaches

##  MDP Properties

### Finite Horizon

- Fixed number of time steps
- Episode terminates after $$T$$ steps
- Value function depends on time to go

### Infinite Horizon

- No fixed termination
- Discounting ensures convergence
- Stationary optimal policy

### Episodic vs Continuing

**Episodic:**
- Natural termination points
- Episodes are independent
- Example: Games

**Continuing:**
- No natural termination
- Goes on indefinitely
- Example: Robot control

##  What's Next?

In the next post, we'll implement **Q-Learning from Scratch** - a model-free algorithm that learns optimal policies through experience. We'll cover:

- Q-learning algorithm
- Exploration strategies
- Implementation details
- Practical examples
- Hyperparameter tuning

##  Key Takeaways

 **MDPs formalize** RL problems mathematically
 **Bellman equations** provide recursive structure
 **Value functions** measure expected returns
 **Dynamic programming** can solve MDPs exactly
 **Model-free methods** learn from experience
 **POMDPs** handle partial observability

##  Practice Exercises

1. **Implement Value Iteration** for a different MDP
2. **Compare Value vs Policy Iteration** convergence speed
3. **Create a POMDP** and implement belief updates
4. **Experiment with different discount factors**
5. **Visualize value functions** as heatmaps

##  Questions?

Have questions about MDPs or Bellman equations? Drop them in the comments below!

**Next Post:** [Part 3: Q-Learning from Scratch]({{ site.baseurl }}{% post_url 2026-02-03-Q-Learning-from-Scratch %})

**Series Index:** [Deep Reinforcement Learning Series Roadmap]({{ site.baseurl }}{% post_url 2026-02-01-Deep-RL-Series-Roadmap %})
