"""
Self-play style training for AlphaZero Trading
Implements data augmentation, curriculum learning, and parallel environments
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
import random


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training"""
    # Parallel environments
    n_envs: int = 8
    
    # Curriculum learning
    initial_leverage_mult: float = 0.3  # Start with 30% of max leverage
    leverage_increase_rate: float = 0.0001  # Increase per step
    max_leverage_mult: float = 1.0
    
    # Data augmentation
    use_augmentation: bool = True
    noise_std: float = 0.01  # Add noise to observations
    time_shift_prob: float = 0.3  # Probability of time-shifted start
    
    # Experience replay
    buffer_size: int = 100000
    min_buffer_size: int = 10000
    
    # Episode settings
    min_episode_length: int = 20  # Minimum steps before early termination
    max_episode_length: int = 252  # ~1 trading year


class DataAugmenter:
    """Augment market data for more diverse training"""
    
    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std
    
    def add_gaussian_noise(self, obs: np.ndarray, std: Optional[float] = None) -> np.ndarray:
        """Add Gaussian noise to observations"""
        std = std or self.noise_std
        noise = np.random.normal(0, std, obs.shape)
        return obs + noise
    
    def scale_features(self, obs: np.ndarray, scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """Randomly scale feature values"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return obs * scale
    
    def time_warp(self, obs: np.ndarray, warp_factor: float = 0.1) -> np.ndarray:
        """Slightly warp the time dimension (stretch/compress)"""
        seq_len = obs.shape[0]
        indices = np.linspace(0, seq_len - 1, seq_len)
        
        # Random warp
        warp = 1.0 + np.random.uniform(-warp_factor, warp_factor)
        warped_indices = np.clip(indices * warp, 0, seq_len - 1).astype(int)
        
        return obs[warped_indices]
    
    def dropout_features(self, obs: np.ndarray, dropout_prob: float = 0.1) -> np.ndarray:
        """Randomly zero out some features"""
        mask = np.random.random(obs.shape) > dropout_prob
        return obs * mask
    
    def augment(self, obs: np.ndarray, augment_prob: float = 0.5) -> np.ndarray:
        """Apply random augmentation"""
        if np.random.random() > augment_prob:
            return obs
        
        obs = obs.copy()
        
        # Randomly apply augmentations
        if np.random.random() < 0.5:
            obs = self.add_gaussian_noise(obs)
        
        if np.random.random() < 0.3:
            obs = self.scale_features(obs)
        
        if np.random.random() < 0.2:
            obs = self.time_warp(obs)
        
        return obs


class SyntheticScenarioGenerator:
    """Generate synthetic market scenarios for training diversity"""
    
    def __init__(self, base_features: np.ndarray, base_returns: np.ndarray):
        """
        Initialize with historical data statistics
        
        Args:
            base_features: Historical feature data (n_samples, seq_len, n_features)
            base_returns: Historical returns (n_samples, n_assets)
        """
        self.feature_mean = np.mean(base_features, axis=(0, 1))
        self.feature_std = np.std(base_features, axis=(0, 1))
        self.return_mean = np.mean(base_returns, axis=0)
        self.return_std = np.std(base_returns, axis=0)
        self.return_corr = np.corrcoef(base_returns.T) if base_returns.shape[1] > 1 else np.array([[1.0]])
        
    def generate_trending_scenario(self, seq_len: int, n_features: int, trend: str = "bull") -> np.ndarray:
        """Generate a trending market scenario"""
        # Start from mean
        features = np.random.normal(0, 1, (seq_len, n_features))
        
        # Add trend component
        trend_strength = 0.02 if trend == "bull" else -0.02
        trend_component = np.linspace(0, trend_strength * seq_len, seq_len)
        
        # Apply to price-related features (assuming first few are returns)
        features[:, 0] += trend_component
        
        return features * self.feature_std + self.feature_mean
    
    def generate_volatile_scenario(self, seq_len: int, n_features: int, vol_mult: float = 2.0) -> np.ndarray:
        """Generate a high volatility scenario"""
        features = np.random.normal(0, vol_mult, (seq_len, n_features))
        return features * self.feature_std + self.feature_mean
    
    def generate_regime_change(self, seq_len: int, n_features: int) -> np.ndarray:
        """Generate a scenario with regime change"""
        mid = seq_len // 2
        
        # First half: low volatility
        first_half = np.random.normal(0, 0.5, (mid, n_features))
        
        # Second half: high volatility with opposite trend
        second_half = np.random.normal(0, 2.0, (seq_len - mid, n_features))
        second_half[:, 0] -= 0.01  # Drawdown
        
        features = np.vstack([first_half, second_half])
        return features * self.feature_std + self.feature_mean
    
    def generate_crash_scenario(self, seq_len: int, n_features: int) -> np.ndarray:
        """Generate a market crash scenario"""
        features = np.random.normal(0, 1, (seq_len, n_features))
        
        # Add crash in the middle
        crash_start = seq_len // 3
        crash_end = 2 * seq_len // 3
        
        features[crash_start:crash_end, 0] -= np.linspace(0, 0.3, crash_end - crash_start)
        features[crash_start:crash_end, :] *= 3  # High volatility during crash
        
        return features * self.feature_std + self.feature_mean


class ParallelEnvManager:
    """Manage multiple parallel environments for self-play"""
    
    def __init__(
        self,
        env_class,
        env_kwargs: Dict,
        n_envs: int = 8,
        augmenter: Optional[DataAugmenter] = None,
    ):
        """
        Initialize parallel environments
        
        Args:
            env_class: Environment class to instantiate
            env_kwargs: Keyword arguments for environment
            n_envs: Number of parallel environments
            augmenter: Optional data augmenter
        """
        self.n_envs = n_envs
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.augmenter = augmenter or DataAugmenter()
        
        # Create environments
        self.envs = [env_class(**env_kwargs) for _ in range(n_envs)]
        
        # Track states
        self.observations = [None] * n_envs
        self.dones = [True] * n_envs
        self.episode_rewards = [0.0] * n_envs
        self.episode_lengths = [0] * n_envs
    
    def reset_all(self) -> np.ndarray:
        """Reset all environments"""
        for i in range(self.n_envs):
            self.observations[i], _ = self.envs[i].reset()
            self.dones[i] = False
            self.episode_rewards[i] = 0.0
            self.episode_lengths[i] = 0
        
        return np.stack(self.observations)
    
    def reset_single(self, idx: int, random_start: bool = True) -> np.ndarray:
        """Reset a single environment with optional random start"""
        env = self.envs[idx]
        obs, info = env.reset()
        
        # Random starting point for diversity
        if random_start:
            max_start = len(env.features) - 300
            if max_start > 0:
                env.current_step = np.random.randint(0, max_start)
        
        self.observations[idx] = obs
        self.dones[idx] = False
        self.episode_rewards[idx] = 0.0
        self.episode_lengths[idx] = 0
        
        return obs
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments
        
        Args:
            actions: Actions for all environments (n_envs, action_dim)
            
        Returns:
            observations, rewards, dones, infos
        """
        rewards = np.zeros(self.n_envs)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = []
        
        for i in range(self.n_envs):
            if self.dones[i]:
                # Reset if done
                self.observations[i] = self.reset_single(i)
                infos.append({"reset": True})
                continue
            
            # Step environment
            obs, reward, done, truncated, info = self.envs[i].step(actions[i])
            
            # Apply augmentation to observation
            if self.augmenter and np.random.random() < 0.3:
                obs = self.augmenter.augment(obs)
            
            self.observations[i] = obs
            rewards[i] = reward
            dones[i] = done or truncated
            self.dones[i] = dones[i]
            self.episode_rewards[i] += reward
            self.episode_lengths[i] += 1
            
            info["episode_reward"] = self.episode_rewards[i]
            info["episode_length"] = self.episode_lengths[i]
            infos.append(info)
        
        return np.stack(self.observations), rewards, dones, infos
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics from completed episodes"""
        return {
            "mean_episode_reward": np.mean(self.episode_rewards),
            "mean_episode_length": np.mean(self.episode_lengths),
            "max_episode_reward": np.max(self.episode_rewards),
            "min_episode_reward": np.min(self.episode_rewards),
        }


class CurriculumScheduler:
    """Curriculum learning scheduler for gradual difficulty increase"""
    
    def __init__(
        self,
        initial_leverage_mult: float = 0.3,
        leverage_increase_rate: float = 0.0001,
        max_leverage_mult: float = 1.0,
        initial_drawdown_mult: float = 2.0,  # Start with 2x allowed drawdown
        drawdown_decrease_rate: float = 0.0001,
        min_drawdown_mult: float = 1.0,
    ):
        self.leverage_mult = initial_leverage_mult
        self.leverage_increase_rate = leverage_increase_rate
        self.max_leverage_mult = max_leverage_mult
        
        self.drawdown_mult = initial_drawdown_mult
        self.drawdown_decrease_rate = drawdown_decrease_rate
        self.min_drawdown_mult = min_drawdown_mult
        
        self.step_count = 0
    
    def step(self):
        """Update curriculum"""
        self.step_count += 1
        
        # Gradually increase leverage
        self.leverage_mult = min(
            self.max_leverage_mult,
            self.leverage_mult + self.leverage_increase_rate
        )
        
        # Gradually decrease allowed drawdown
        self.drawdown_mult = max(
            self.min_drawdown_mult,
            self.drawdown_mult - self.drawdown_decrease_rate
        )
    
    def get_effective_leverage(self, base_leverage: float) -> float:
        """Get current effective leverage limit"""
        return base_leverage * self.leverage_mult
    
    def get_effective_drawdown(self, base_drawdown: float) -> float:
        """Get current effective drawdown limit"""
        return base_drawdown * self.drawdown_mult
    
    def get_state(self) -> Dict[str, float]:
        """Get current curriculum state"""
        return {
            "leverage_mult": self.leverage_mult,
            "drawdown_mult": self.drawdown_mult,
            "step_count": self.step_count,
        }


class PrioritizedExperienceReplay:
    """Experience replay buffer with prioritization based on TD error"""
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling
        beta_increment: float = 0.001,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Dict,
    ):
        """Add experience with max priority"""
        experience = (obs, action, reward, next_obs, done, info)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with prioritization"""
        n = len(self.buffer)
        if n < batch_size:
            batch_size = n
        
        # Calculate sampling probabilities
        priorities = self.priorities[:n] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get experiences
        batch = [self.buffer[i] for i in indices]
        obs, actions, rewards, next_obs, dones, infos = zip(*batch)
        
        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards),
            np.array(next_obs),
            np.array(dones),
            list(infos),
            indices,
            weights,
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6
            self.max_priority = max(self.max_priority, abs(error))
    
    def __len__(self):
        return len(self.buffer)


class ImprovedRewardShaper:
    """Shape rewards to encourage risk-adjusted returns"""
    
    def __init__(
        self,
        sharpe_window: int = 20,
        drawdown_penalty_scale: float = 10.0,
        turnover_penalty: float = 0.1,
        leverage_penalty_threshold: float = 8.0,
    ):
        self.sharpe_window = sharpe_window
        self.drawdown_penalty_scale = drawdown_penalty_scale
        self.turnover_penalty = turnover_penalty
        self.leverage_penalty_threshold = leverage_penalty_threshold
        
        self.returns_buffer = []
    
    def shape_reward(
        self,
        raw_reward: float,
        daily_return: float,
        drawdown: float,
        leverage: float,
        turnover: float,
        max_drawdown_limit: float,
    ) -> float:
        """
        Shape the reward to encourage risk-adjusted behavior
        
        Returns:
            Shaped reward
        """
        shaped = 0.0
        
        # 1. Base return component (scaled)
        shaped += daily_return * 100
        
        # 2. Rolling Sharpe bonus
        self.returns_buffer.append(daily_return)
        if len(self.returns_buffer) > self.sharpe_window:
            self.returns_buffer.pop(0)
        
        if len(self.returns_buffer) >= self.sharpe_window:
            returns = np.array(self.returns_buffer)
            if returns.std() > 0:
                rolling_sharpe = returns.mean() / returns.std() * np.sqrt(252)
                shaped += rolling_sharpe * 0.5  # Bonus for positive Sharpe
        
        # 3. Drawdown penalty (exponential)
        if drawdown < 0:
            dd_ratio = abs(drawdown) / max_drawdown_limit
            dd_penalty = (dd_ratio ** 2) * self.drawdown_penalty_scale
            shaped -= dd_penalty
        
        # 4. Turnover penalty (encourage holding)
        shaped -= turnover * self.turnover_penalty
        
        # 5. Excessive leverage penalty
        if leverage > self.leverage_penalty_threshold:
            leverage_penalty = (leverage - self.leverage_penalty_threshold) * 0.5
            shaped -= leverage_penalty
        
        # 6. Survival bonus (reward for not blowing up)
        shaped += 0.1
        
        return shaped
    
    def reset(self):
        """Reset for new episode"""
        self.returns_buffer = []


if __name__ == "__main__":
    # Test augmenter
    print("=== Data Augmenter Test ===")
    aug = DataAugmenter(noise_std=0.01)
    
    obs = np.random.randn(60, 100)
    aug_obs = aug.augment(obs)
    print(f"Original shape: {obs.shape}, Augmented shape: {aug_obs.shape}")
    print(f"Mean diff: {np.abs(obs - aug_obs).mean():.4f}")
    
    # Test curriculum
    print("\n=== Curriculum Scheduler Test ===")
    curriculum = CurriculumScheduler(
        initial_leverage_mult=0.3,
        leverage_increase_rate=0.01,
    )
    
    for i in range(10):
        curriculum.step()
        state = curriculum.get_state()
        print(f"Step {i+1}: leverage_mult={state['leverage_mult']:.2f}, drawdown_mult={state['drawdown_mult']:.2f}")
    
    # Test reward shaper
    print("\n=== Reward Shaper Test ===")
    shaper = ImprovedRewardShaper()
    
    for i in range(25):
        daily_return = np.random.normal(0.001, 0.01)
        drawdown = -abs(np.random.normal(0.02, 0.01))
        leverage = 5 + np.random.random() * 3
        turnover = np.random.random() * 0.1
        
        shaped = shaper.shape_reward(
            raw_reward=0,
            daily_return=daily_return,
            drawdown=drawdown,
            leverage=leverage,
            turnover=turnover,
            max_drawdown_limit=0.05,
        )
        
        if i % 5 == 0:
            print(f"Step {i}: return={daily_return:.4f}, dd={drawdown:.4f}, shaped_reward={shaped:.4f}")
    
    print("\n=== Tests Complete ===")

