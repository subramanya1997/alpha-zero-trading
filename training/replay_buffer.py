"""
Experience replay buffer for training
"""
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
from collections import deque
import random


class ReplayBuffer:
    """Simple replay buffer for storing transitions"""
    
    def __init__(
        self,
        capacity: int = 100000,
        device: str = "cpu",
    ):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
            device: Device for tensor operations
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.FloatTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(np.array(rewards)).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(np.array(dones)).to(self.device),
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples"""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer
    
    Samples transitions with probability proportional to TD error
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        device: str = "cpu",
    ):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent
            beta_increment: How much to increase beta per sample
            device: Device for tensors
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add transition with max priority"""
        transition = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(
        self, 
        batch_size: int,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample with prioritization
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        n = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get transitions
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.FloatTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(np.array(rewards)).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(np.array(dones)).to(self.device),
            indices,
            torch.FloatTensor(weights).to(self.device),
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities after learning"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon for stability
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return len(self.buffer)


class RolloutStorage:
    """
    Storage for on-policy rollouts (for PPO)
    """
    
    def __init__(
        self,
        n_steps: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: str = "cpu",
    ):
        """
        Initialize rollout storage
        
        Args:
            n_steps: Number of steps to store
            obs_shape: Shape of observations
            action_shape: Shape of actions
            device: Device for tensors
        """
        self.n_steps = n_steps
        self.device = device
        
        self.observations = torch.zeros(n_steps + 1, *obs_shape)
        self.actions = torch.zeros(n_steps, *action_shape)
        self.rewards = torch.zeros(n_steps)
        self.values = torch.zeros(n_steps + 1)
        self.log_probs = torch.zeros(n_steps)
        self.dones = torch.zeros(n_steps + 1)
        self.returns = torch.zeros(n_steps + 1)
        self.advantages = torch.zeros(n_steps)
        
        self.step = 0
    
    def insert(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool,
    ):
        """Insert a new transition"""
        self.observations[self.step + 1] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.log_probs[self.step] = log_prob
        self.dones[self.step + 1] = done
        
        self.step = (self.step + 1) % self.n_steps
    
    def compute_returns(
        self,
        next_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Compute returns and advantages using GAE"""
        self.values[-1] = next_value
        gae = 0
        
        for step in reversed(range(self.n_steps)):
            delta = (
                self.rewards[step]
                + gamma * self.values[step + 1] * (1 - self.dones[step + 1])
                - self.values[step]
            )
            gae = delta + gamma * gae_lambda * (1 - self.dones[step + 1]) * gae
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]
    
    def get_batches(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Generate mini-batches for training
        
        Yields:
            Tuple of (obs, actions, returns, old_log_probs, advantages)
        """
        indices = torch.randperm(self.n_steps)
        
        for start in range(0, self.n_steps, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                self.observations[batch_indices].to(self.device),
                self.actions[batch_indices].to(self.device),
                self.returns[batch_indices].to(self.device),
                self.log_probs[batch_indices].to(self.device),
                self.advantages[batch_indices].to(self.device),
            )
    
    def reset(self):
        """Reset storage"""
        self.step = 0
        self.observations[0] = self.observations[-1]
        self.dones[0] = self.dones[-1]


if __name__ == "__main__":
    # Test replay buffers
    print("=== Replay Buffer Test ===")
    
    buffer = ReplayBuffer(capacity=1000)
    
    # Fill with random data
    for i in range(100):
        state = np.random.randn(60, 258)
        action = np.random.rand(5)
        reward = np.random.randn()
        next_state = np.random.randn(60, 258)
        done = np.random.rand() > 0.95
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Sample
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"Sampled batch shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    
    # Test prioritized buffer
    print("\n=== Prioritized Buffer Test ===")
    
    pbuffer = PrioritizedReplayBuffer(capacity=1000)
    
    for i in range(100):
        state = np.random.randn(60, 258)
        action = np.random.rand(5)
        reward = np.random.randn()
        next_state = np.random.randn(60, 258)
        done = np.random.rand() > 0.95
        
        pbuffer.push(state, action, reward, next_state, done)
    
    states, actions, rewards, next_states, dones, indices, weights = pbuffer.sample(32)
    print(f"Prioritized sample shapes:")
    print(f"  States: {states.shape}")
    print(f"  Weights: {weights.shape}")
    print(f"  Indices: {indices[:5]}...")
    
    # Update priorities
    new_priorities = np.random.rand(32)
    pbuffer.update_priorities(indices, new_priorities)
    print(f"Updated priorities")

