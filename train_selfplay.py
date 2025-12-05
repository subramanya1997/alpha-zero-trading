#!/usr/bin/env python
"""
Self-Play Training for AlphaZero Trading Agent
With data augmentation, curriculum learning, and improved reward shaping
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

from config import Config, DEFAULT_CONFIG
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.preprocessing import DataPreprocessor
from environment.trading_env import TradingEnv, EnvConfig
from models.alphazero_agent import AlphaZeroAgent, AgentConfig, PPOTrainer, RolloutBuffer
from training.metrics import MetricsTracker
from training.checkpointing import CheckpointManager
from training.self_play import (
    DataAugmenter, CurriculumScheduler, ImprovedRewardShaper, 
    PrioritizedExperienceReplay, SelfPlayConfig
)
from utils.helpers import set_seed, get_device, count_parameters
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Play Training for AlphaZero Trading")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    
    # Model
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--encoder-type", type=str, default="gated_transformer",
                       choices=["transformer", "gated_transformer", "lstm", "tcn", "hybrid"])
    
    # Self-play settings
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--use-curriculum", action="store_true", default=True,
                       help="Use curriculum learning")
    parser.add_argument("--use-augmentation", action="store_true", default=True,
                       help="Use data augmentation")
    parser.add_argument("--initial-leverage-mult", type=float, default=0.3,
                       help="Initial leverage multiplier for curriculum")
    
    # Trading
    parser.add_argument("--initial-capital", type=float, default=10000)
    parser.add_argument("--max-leverage", type=float, default=10)
    parser.add_argument("--min-leverage", type=float, default=1)  # Start lower
    parser.add_argument("--max-drawdown", type=float, default=0.10)  # Start with 10%, reduce to 5%
    
    # Logging
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="alphazero-trading")
    parser.add_argument("--run-name", type=str, default=None)
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-freq", type=int, default=50000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--log-freq", type=int, default=1000)
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    
    return parser.parse_args()


class SelfPlayTrainer:
    """Self-play training with curriculum and augmentation"""
    
    def __init__(
        self,
        agent: AlphaZeroAgent,
        train_env: TradingEnv,
        val_env: TradingEnv,
        config: SelfPlayConfig,
        device: str = "cpu",
        use_wandb: bool = False,
    ):
        self.agent = agent
        self.train_env = train_env
        self.val_env = val_env
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Data augmenter
        self.augmenter = DataAugmenter(noise_std=0.01) if config.use_augmentation else None
        
        # Curriculum scheduler
        self.curriculum = CurriculumScheduler(
            initial_leverage_mult=config.initial_leverage_mult,
            leverage_increase_rate=0.00005,  # Slow increase
            initial_drawdown_mult=2.0,  # Start with 2x allowed drawdown
            drawdown_decrease_rate=0.00002,
        )
        
        # Reward shaper for each environment
        self.reward_shapers = [ImprovedRewardShaper() for _ in range(config.n_envs)]
        
        # Experience buffer
        self.experience_buffer = PrioritizedExperienceReplay(
            capacity=config.buffer_size,
            alpha=0.6,
            beta=0.4,
        )
        
        # PPO trainer
        self.ppo = PPOTrainer(
            agent,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.02,  # Higher entropy for exploration
            max_grad_norm=0.5,
            n_epochs=10,
            batch_size=64,
            device=device,
        )
        
        # Metrics tracking
        self.episode_returns = deque(maxlen=100)
        self.episode_sharpes = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Multiple parallel environments (simulated via different starting points)
        self.env_states = []
        self.env_returns_buffer = [[] for _ in range(config.n_envs)]
    
    def _get_random_start_env(self, env_idx: int) -> tuple:
        """Get environment with random starting point"""
        obs, info = self.train_env.reset()
        
        # Random starting point
        max_start = len(self.train_env.features) - 300
        if max_start > 0:
            start_idx = np.random.randint(0, max_start)
            self.train_env.current_step = start_idx
        
        # Apply curriculum to environment config
        effective_leverage = self.curriculum.get_effective_leverage(
            self.train_env.config.max_leverage
        )
        effective_drawdown = self.curriculum.get_effective_drawdown(
            self.train_env.config.max_drawdown
        )
        
        # Update environment's effective limits
        self.train_env.config.max_leverage = min(effective_leverage, 10.0)
        self.train_env.config.min_leverage = max(1.0, effective_leverage * 0.3)
        
        # Get fresh observation
        obs = self.train_env._get_observation()
        
        # Reset reward shaper
        self.reward_shapers[env_idx].reset()
        self.env_returns_buffer[env_idx] = []
        
        return obs, info
    
    def _shape_reward(
        self, 
        env_idx: int,
        raw_reward: float, 
        info: dict,
        action: np.ndarray,
        prev_action: np.ndarray,
    ) -> float:
        """Shape reward using improved reward shaper"""
        daily_return = info.get("daily_return", 0)
        drawdown = info.get("drawdown", 0)
        leverage = info.get("leverage", 1)
        cumulative_return = info.get("cumulative_return", 0)
        
        # Calculate turnover (position change)
        turnover = np.abs(action - prev_action).sum() if prev_action is not None else 0
        
        shaped = self.reward_shapers[env_idx].shape_reward(
            raw_reward=raw_reward,
            daily_return=daily_return,
            drawdown=drawdown,
            leverage=leverage,
            turnover=turnover,
            max_drawdown_limit=self.train_env.config.max_drawdown,
        )
        
        # Track returns for Sharpe calculation (use actual daily return)
        if daily_return != 0:  # Only track non-zero returns
            self.env_returns_buffer[env_idx].append(daily_return)
        else:
            # Even zero returns should be tracked for proper Sharpe
            self.env_returns_buffer[env_idx].append(daily_return)
        
        return shaped
    
    def _calculate_episode_sharpe(self, env_idx: int) -> float:
        """Calculate Sharpe ratio for completed episode"""
        returns = np.array(self.env_returns_buffer[env_idx])
        if len(returns) < 10:
            return 0.0
        std = returns.std()
        if std < 1e-8:  # Near-zero volatility
            return 0.0
        sharpe = returns.mean() / std * np.sqrt(252)
        return np.clip(sharpe, -10, 10)  # Clip extreme values
    
    def _augment_observation(self, obs: np.ndarray) -> np.ndarray:
        """Augment observation with probability"""
        if self.augmenter and np.random.random() < 0.3:
            return self.augmenter.augment(obs)
        return obs
    
    def collect_rollout(self, n_steps: int) -> dict:
        """Collect rollout with self-play features"""
        self.agent.eval()
        
        # Storage
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        # Episode tracking
        episode_rewards = []
        episode_lengths = []
        episode_sharpes = []
        
        # Initialize environments with random starts
        obs, info = self._get_random_start_env(0)
        prev_action = None
        episode_reward = 0
        episode_length = 0
        
        for step in range(n_steps):
            # Augment observation
            obs_aug = self._augment_observation(obs)
            obs_tensor = torch.FloatTensor(obs_aug).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, _, value = self.agent(obs_tensor)
            
            action_np = action.cpu().numpy()[0]
            
            # Step environment
            next_obs, raw_reward, done, truncated, info = self.train_env.step(action_np)
            
            # Shape reward
            shaped_reward = self._shape_reward(0, raw_reward, info, action_np, prev_action)
            
            # Store transition
            observations.append(torch.FloatTensor(obs_aug))
            actions.append(action.squeeze(0).cpu())
            rewards.append(shaped_reward)
            values.append(value.cpu())
            log_probs.append(log_prob.cpu())
            dones.append(done or truncated)
            
            # Add to experience buffer
            self.experience_buffer.add(
                obs_aug, action_np, shaped_reward, next_obs, done or truncated, info
            )
            
            episode_reward += shaped_reward
            episode_length += 1
            prev_action = action_np
            
            # Episode end
            if done or truncated:
                sharpe = self._calculate_episode_sharpe(0)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_sharpes.append(sharpe)
                
                self.episode_returns.append(episode_reward)
                self.episode_sharpes.append(sharpe)
                self.episode_lengths.append(episode_length)
                
                # Reset with new random start
                obs, info = self._get_random_start_env(0)
                prev_action = None
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
            
            # Update curriculum
            self.curriculum.step()
        
        # Get final value
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            next_value = self.agent.get_value(obs_tensor).cpu()
        
        # Stack tensors
        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs).squeeze()
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Compute GAE (ensure all on CPU for computation)
        advantages, returns = self.ppo.compute_gae(
            rewards.cpu(), values.cpu(), dones.cpu(), next_value.squeeze().cpu()
        )
        
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "values": values,
            "log_probs": log_probs,
            "dones": dones,
            "advantages": advantages,
            "returns": returns,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "episode_sharpes": episode_sharpes,
        }
    
    def train_step(self, rollout: dict) -> dict:
        """Perform PPO update"""
        self.agent.train()
        
        train_metrics = self.ppo.update(
            rollout["observations"].to(self.device),
            rollout["actions"].to(self.device),
            rollout["log_probs"].to(self.device),
            rollout["advantages"].to(self.device),
            rollout["returns"].to(self.device),
        )
        
        return train_metrics
    
    def evaluate(self) -> dict:
        """Evaluate on validation set"""
        self.agent.eval()
        
        obs, info = self.val_env.reset()
        self.val_env.current_step = 0
        self.val_env.portfolio.reset()
        
        returns = []
        values = []
        done = False
        truncated = False
        
        while not done and not truncated:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _, _, _ = self.agent(obs_tensor, deterministic=True)
            
            obs, reward, done, truncated, info = self.val_env.step(action.cpu().numpy()[0])
            returns.append(info.get("daily_return", 0))
            values.append(info.get("total_value", 0))
        
        returns_arr = np.array(returns)
        
        sharpe = returns_arr.mean() / returns_arr.std() * np.sqrt(252) if returns_arr.std() > 0 else 0
        total_return = (values[-1] / values[0]) - 1 if values else 0
        max_dd = min(self.val_env.episode_drawdowns) if self.val_env.episode_drawdowns else 0
        
        return {
            "sharpe_ratio": sharpe,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "n_steps": len(returns),
            "final_value": values[-1] if values else 0,
        }
    
    def get_training_stats(self) -> dict:
        """Get current training statistics"""
        curriculum_state = self.curriculum.get_state()
        
        return {
            "mean_episode_reward": np.mean(self.episode_returns) if self.episode_returns else 0,
            "mean_episode_sharpe": np.mean(self.episode_sharpes) if self.episode_sharpes else 0,
            "mean_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
            "curriculum_leverage_mult": curriculum_state["leverage_mult"],
            "curriculum_drawdown_mult": curriculum_state["drawdown_mult"],
            "experience_buffer_size": len(self.experience_buffer),
        }


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    logger = setup_logger(log_dir="logs")
    
    logger.info("=" * 60)
    logger.info("AlphaZero Self-Play Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Encoder: {args.encoder_type}")
    logger.info(f"Curriculum: {args.use_curriculum}")
    logger.info(f"Augmentation: {args.use_augmentation}")
    logger.info(f"Parallel Envs: {args.n_envs}")
    
    # Load data
    logger.info("\nLoading data...")
    data_loader = DataLoader()
    data = data_loader.download_all()
    
    engineer = FeatureEngineer()
    processed = engineer.process_all_symbols(data)
    processed = engineer.add_cross_asset_features(processed)
    
    preprocessor = DataPreprocessor()
    split, sequences = preprocessor.prepare_training_data(processed)
    
    # Create environments
    env_config = EnvConfig(
        initial_capital=args.initial_capital,
        max_leverage=args.max_leverage,
        min_leverage=args.min_leverage,
        max_drawdown=args.max_drawdown,
    )
    
    train_features, train_prices, train_returns, train_dates = sequences["train"]
    val_features, val_prices, val_returns, val_dates = sequences["val"]
    
    train_env = TradingEnv(train_features, train_prices, train_returns, train_dates, env_config)
    val_env = TradingEnv(val_features, val_prices, val_returns, val_dates, env_config)
    
    logger.info(f"Train: {len(train_features)} samples, {split.train_dates}")
    logger.info(f"Val: {len(val_features)} samples, {split.val_dates}")
    
    # Create agent
    input_dim = train_features.shape[2] + 10
    
    agent_config = AgentConfig(
        input_dim=input_dim,
        seq_len=60,
        n_assets=4,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        encoder_type=args.encoder_type,
    )
    
    agent = AlphaZeroAgent(agent_config).to(device)
    logger.info(f"Agent parameters: {count_parameters(agent):,}")
    
    # Self-play config
    selfplay_config = SelfPlayConfig(
        n_envs=args.n_envs,
        initial_leverage_mult=args.initial_leverage_mult,
        use_augmentation=args.use_augmentation,
    )
    
    # Create trainer
    trainer = SelfPlayTrainer(
        agent=agent,
        train_env=train_env,
        val_env=val_env,
        config=selfplay_config,
        device=device,
        use_wandb=args.use_wandb,
    )
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_best=3,
        keep_latest=5,
    )
    
    # Wandb
    if args.use_wandb:
        run_name = args.run_name or f"selfplay-{args.encoder_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
    
    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("Starting Self-Play Training...")
    logger.info("=" * 60)
    
    global_step = 0
    best_val_sharpe = -float("inf")
    
    while global_step < args.total_timesteps:
        # Collect rollout
        rollout = trainer.collect_rollout(args.n_steps)
        global_step += args.n_steps
        
        # Train
        train_metrics = trainer.train_step(rollout)
        
        # Get stats
        stats = trainer.get_training_stats()
        
        # Logging
        if global_step % args.log_freq == 0:
            logger.info(f"\n[Step {global_step:,}]")
            logger.info(f"  Episode Reward: {stats['mean_episode_reward']:.2f}")
            logger.info(f"  Episode Sharpe: {stats['mean_episode_sharpe']:.2f}")
            logger.info(f"  Episode Length: {stats['mean_episode_length']:.0f}")
            logger.info(f"  Curriculum Leverage: {stats['curriculum_leverage_mult']:.2f}x")
            logger.info(f"  Policy Loss: {train_metrics['policy_loss']:.4f}")
            logger.info(f"  Value Loss: {train_metrics['value_loss']:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "train/step": global_step,
                    "train/episode_reward": stats['mean_episode_reward'],
                    "train/episode_sharpe": stats['mean_episode_sharpe'],
                    "train/episode_length": stats['mean_episode_length'],
                    "train/curriculum_leverage": stats['curriculum_leverage_mult'],
                    "train/curriculum_drawdown": stats['curriculum_drawdown_mult'],
                    "train/policy_loss": train_metrics['policy_loss'],
                    "train/value_loss": train_metrics['value_loss'],
                    "train/entropy": train_metrics['entropy'],
                })
        
        # Evaluation
        if global_step % args.eval_freq == 0:
            logger.info("\n--- Validation ---")
            val_metrics = trainer.evaluate()
            
            logger.info(f"  Sharpe: {val_metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Return: {val_metrics['total_return']:.2%}")
            logger.info(f"  Max DD: {val_metrics['max_drawdown']:.2%}")
            logger.info(f"  Steps: {val_metrics['n_steps']}")
            
            if args.use_wandb:
                wandb.log({
                    "val/sharpe_ratio": val_metrics['sharpe_ratio'],
                    "val/total_return": val_metrics['total_return'],
                    "val/max_drawdown": val_metrics['max_drawdown'],
                })
            
            # Save best
            if val_metrics['sharpe_ratio'] > best_val_sharpe:
                best_val_sharpe = val_metrics['sharpe_ratio']
                checkpoint_manager.save(
                    agent, trainer.ppo.optimizer, global_step,
                    val_metrics, is_best=True,
                )
                logger.info(f"  New best model! Sharpe: {best_val_sharpe:.2f}")
        
        # Checkpoint
        if global_step % args.save_freq == 0:
            checkpoint_manager.save(
                agent, trainer.ppo.optimizer, global_step,
                {"step": global_step}, is_best=False,
            )
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best Validation Sharpe: {best_val_sharpe:.2f}")
    logger.info("=" * 60)
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

