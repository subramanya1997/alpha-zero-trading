"""
Main training loop for AlphaZero trading agent
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import wandb

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config, DEFAULT_CONFIG
from models.alphazero_agent import AlphaZeroAgent, AgentConfig, PPOTrainer, RolloutBuffer
from environment.trading_env import TradingEnv, EnvConfig
from risk.risk_manager import RiskManager, RiskLimits
from training.metrics import MetricsTracker
from training.checkpointing import CheckpointManager


class Trainer:
    """
    Main trainer for AlphaZero trading agent
    """
    
    def __init__(
        self,
        env: TradingEnv,
        val_env: Optional[TradingEnv] = None,
        config: Optional[Config] = None,
        use_wandb: bool = True,
    ):
        """
        Initialize trainer
        
        Args:
            env: Training environment
            val_env: Optional validation environment
            config: Configuration
            use_wandb: Whether to log to wandb
        """
        self.config = config or DEFAULT_CONFIG
        self.env = env
        self.val_env = val_env
        self.use_wandb = use_wandb
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")
        
        # Create agent
        agent_config = AgentConfig(
            input_dim=env.n_features + 10,  # features + portfolio state
            seq_len=config.trading.lookback_window,
            n_assets=config.model.n_assets,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            d_ff=config.model.d_ff,
            hidden_dims=config.model.hidden_dims,
            dropout=config.model.dropout,
        )
        
        self.agent = AlphaZeroAgent(agent_config).to(self.device)
        print(f"Agent parameters: {sum(p.numel() for p in self.agent.parameters()):,}")
        
        # PPO trainer
        self.ppo = PPOTrainer(
            self.agent,
            lr=config.training.learning_rate,
            gamma=config.training.gamma,
            gae_lambda=config.training.gae_lambda,
            clip_epsilon=config.training.clip_epsilon,
            value_coef=config.training.value_coef,
            entropy_coef=config.training.entropy_coef,
            max_grad_norm=config.training.max_grad_norm,
            n_epochs=config.training.n_epochs,
            batch_size=config.training.batch_size,
            device=self.device,
        )
        
        # Risk manager
        self.risk_manager = RiskManager(RiskLimits(
            max_drawdown=config.trading.max_drawdown,
            max_position_size=config.trading.max_position_size,
            max_leverage=config.trading.max_leverage,
            min_leverage=config.trading.min_leverage,
        ))
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(device=self.device)
        
        # Metrics tracker
        self.metrics = MetricsTracker()
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            keep_best=3,
            keep_latest=5,
        )
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_val_sharpe = -float("inf")
        
    def setup_wandb(self, run_name: Optional[str] = None):
        """Initialize wandb logging"""
        if not self.use_wandb:
            return
        
        run_name = run_name or f"alphazero-trading-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        wandb.init(
            project=self.config.training.project_name,
            name=run_name,
            config={
                "trading": vars(self.config.trading),
                "model": vars(self.config.model),
                "training": vars(self.config.training),
            },
        )
        
        # Watch model
        wandb.watch(self.agent, log="all", log_freq=100)
    
    def collect_rollout(self, n_steps: int) -> Dict[str, float]:
        """
        Collect rollout from environment
        
        Args:
            n_steps: Number of steps to collect
            
        Returns:
            Episode metrics
        """
        self.rollout_buffer.reset()
        self.agent.eval()
        
        obs, info = self.env.reset()
        episode_reward = 0
        episode_returns = []
        episode_drawdowns = []
        
        for step in range(n_steps):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, _, value = self.agent(obs_tensor)
            
            # Convert to numpy
            action_np = action.cpu().numpy()[0]
            
            # Step environment
            next_obs, reward, done, truncated, info = self.env.step(action_np)
            
            # Store in buffer
            self.rollout_buffer.add(
                torch.FloatTensor(obs),
                action.squeeze(0).cpu(),
                reward,
                value.cpu(),
                log_prob.cpu(),
                done or truncated,
            )
            
            episode_reward += reward
            episode_returns.append(info.get("daily_return", 0))
            episode_drawdowns.append(info.get("drawdown", 0))
            
            obs = next_obs
            self.global_step += 1
            
            if done or truncated:
                obs, info = self.env.reset()
                self.episode_count += 1
        
        # Get final value for GAE
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            next_value = self.agent.get_value(obs_tensor).cpu()
        
        # Compute GAE
        obs_batch, actions, rewards, values, log_probs, dones = self.rollout_buffer.get()
        advantages, returns = self.ppo.compute_gae(rewards, values.squeeze(), dones, next_value.squeeze())
        
        # Update agent
        self.agent.train()
        train_metrics = self.ppo.update(obs_batch, actions, log_probs.squeeze(), advantages, returns)
        
        # Episode metrics
        episode_metrics = {
            "episode_reward": episode_reward,
            "episode_return": np.sum(episode_returns),
            "episode_sharpe": self._calculate_sharpe(episode_returns),
            "episode_max_dd": min(episode_drawdowns) if episode_drawdowns else 0,
            "final_value": info.get("total_value", 0),
            "final_leverage": info.get("leverage", 0),
            **train_metrics,
        }
        
        return episode_metrics
    
    def _calculate_sharpe(self, returns: list, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        returns = np.array(returns)
        excess = returns - risk_free / 252
        if np.std(returns) == 0:
            return 0.0
        return np.mean(excess) / np.std(returns) * np.sqrt(252)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate agent on validation environment"""
        if self.val_env is None:
            return {}
        
        self.agent.eval()
        
        obs, info = self.val_env.reset()
        total_reward = 0
        returns = []
        drawdowns = []
        
        done = False
        truncated = False
        
        while not done and not truncated:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, _, _, _ = self.agent(obs_tensor, deterministic=True)
            
            action_np = action.cpu().numpy()[0]
            obs, reward, done, truncated, info = self.val_env.step(action_np)
            
            total_reward += reward
            returns.append(info.get("daily_return", 0))
            drawdowns.append(info.get("drawdown", 0))
        
        val_metrics = {
            "val/total_reward": total_reward,
            "val/total_return": info.get("cumulative_return", 0),
            "val/sharpe_ratio": self._calculate_sharpe(returns),
            "val/max_drawdown": min(drawdowns) if drawdowns else 0,
            "val/final_value": info.get("total_value", 0),
        }
        
        return val_metrics
    
    def train(
        self,
        total_timesteps: int,
        n_steps: int = 2048,
        eval_freq: int = 10000,
        save_freq: int = 50000,
        log_freq: int = 100,
    ):
        """
        Main training loop
        
        Args:
            total_timesteps: Total timesteps to train
            n_steps: Steps per rollout
            eval_freq: Evaluation frequency
            save_freq: Checkpoint save frequency
            log_freq: Logging frequency
        """
        print(f"\nStarting training for {total_timesteps:,} timesteps")
        print(f"  n_steps per rollout: {n_steps}")
        print(f"  eval_freq: {eval_freq}")
        print(f"  save_freq: {save_freq}")
        
        start_time = time.time()
        
        while self.global_step < total_timesteps:
            # Collect rollout and update
            episode_metrics = self.collect_rollout(n_steps)
            
            # Track metrics
            self.metrics.update(episode_metrics)
            
            # Logging
            if self.global_step % log_freq == 0:
                avg_metrics = self.metrics.get_averages()
                
                print(f"\n[Step {self.global_step:,}] Episode {self.episode_count}")
                print(f"  Reward: {avg_metrics.get('episode_reward', 0):.2f}")
                print(f"  Return: {avg_metrics.get('episode_return', 0):.2%}")
                print(f"  Sharpe: {avg_metrics.get('episode_sharpe', 0):.2f}")
                print(f"  Max DD: {avg_metrics.get('episode_max_dd', 0):.2%}")
                print(f"  Policy Loss: {avg_metrics.get('policy_loss', 0):.4f}")
                print(f"  Value Loss: {avg_metrics.get('value_loss', 0):.4f}")
                
                if self.use_wandb:
                    wandb.log({
                        "step": self.global_step,
                        "episode": self.episode_count,
                        **avg_metrics,
                    })
                
                self.metrics.reset()
            
            # Evaluation
            if self.global_step % eval_freq == 0:
                val_metrics = self.evaluate()
                
                if val_metrics:
                    print(f"\n[Validation]")
                    print(f"  Return: {val_metrics.get('val/total_return', 0):.2%}")
                    print(f"  Sharpe: {val_metrics.get('val/sharpe_ratio', 0):.2f}")
                    print(f"  Max DD: {val_metrics.get('val/max_drawdown', 0):.2%}")
                    
                    if self.use_wandb:
                        wandb.log({
                            "step": self.global_step,
                            **val_metrics,
                        })
                    
                    # Save best model
                    val_sharpe = val_metrics.get("val/sharpe_ratio", 0)
                    if val_sharpe > self.best_val_sharpe:
                        self.best_val_sharpe = val_sharpe
                        self.checkpoint_manager.save(
                            self.agent,
                            self.ppo.optimizer,
                            self.global_step,
                            val_metrics,
                            is_best=True,
                        )
                        print(f"  New best model! Sharpe: {val_sharpe:.2f}")
            
            # Checkpointing
            if self.global_step % save_freq == 0:
                self.checkpoint_manager.save(
                    self.agent,
                    self.ppo.optimizer,
                    self.global_step,
                    episode_metrics,
                    is_best=False,
                )
        
        # Final save
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/3600:.2f} hours")
        print(f"Total episodes: {self.episode_count}")
        print(f"Best validation Sharpe: {self.best_val_sharpe:.2f}")
        
        self.checkpoint_manager.save(
            self.agent,
            self.ppo.optimizer,
            self.global_step,
            {"final": True},
            is_best=False,
        )
        
        if self.use_wandb:
            wandb.finish()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = self.checkpoint_manager.load(
            checkpoint_path,
            self.agent,
            self.ppo.optimizer,
        )
        if checkpoint:
            self.global_step = checkpoint.get("step", 0)
            print(f"Loaded checkpoint from step {self.global_step}")


if __name__ == "__main__":
    from data.data_loader import DataLoader
    from data.feature_engineering import FeatureEngineer
    from data.preprocessing import DataPreprocessor
    
    print("Loading data...")
    loader = DataLoader()
    data = loader.download_all()
    
    print("Processing features...")
    engineer = FeatureEngineer()
    processed = engineer.process_all_symbols(data)
    processed = engineer.add_cross_asset_features(processed)
    
    print("Preprocessing...")
    preprocessor = DataPreprocessor()
    split, sequences = preprocessor.prepare_training_data(processed)
    
    # Create environments
    print("\nCreating environments...")
    env_config = EnvConfig(
        initial_capital=10000,
        max_leverage=10,
        min_leverage=5,
        max_drawdown=0.05,
    )
    
    train_env = TradingEnv(*sequences["train"], config=env_config)
    val_env = TradingEnv(*sequences["val"], config=env_config)
    
    # Create trainer
    config = DEFAULT_CONFIG
    trainer = Trainer(train_env, val_env, config, use_wandb=False)
    
    # Short test training
    print("\n--- Starting test training ---")
    trainer.train(
        total_timesteps=5000,
        n_steps=256,
        eval_freq=1000,
        save_freq=5000,
        log_freq=256,
    )

