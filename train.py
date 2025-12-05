#!/usr/bin/env python
"""
Main training script for AlphaZero Trading Agent
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
import wandb

from config import Config, DEFAULT_CONFIG
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.preprocessing import DataPreprocessor
from environment.trading_env import TradingEnv, EnvConfig
from models.alphazero_agent import AlphaZeroAgent, AgentConfig, PPOTrainer, RolloutBuffer
from training.metrics import MetricsTracker
from training.checkpointing import CheckpointManager
from utils.helpers import set_seed, get_device, count_parameters
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train AlphaZero Trading Agent")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                       help="Total timesteps to train")
    parser.add_argument("--n-steps", type=int, default=2048,
                       help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for PPO updates")
    parser.add_argument("--n-epochs", type=int, default=10,
                       help="PPO epochs per update")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    
    # Model
    parser.add_argument("--d-model", type=int, default=128,
                       help="Transformer model dimension")
    parser.add_argument("--n-heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4,
                       help="Number of transformer layers")
    
    # Trading
    parser.add_argument("--initial-capital", type=float, default=10000,
                       help="Initial capital")
    parser.add_argument("--max-leverage", type=float, default=10,
                       help="Maximum leverage")
    parser.add_argument("--min-leverage", type=float, default=5,
                       help="Minimum leverage")
    parser.add_argument("--max-drawdown", type=float, default=0.05,
                       help="Maximum allowed drawdown")
    
    # Logging
    parser.add_argument("--use-wandb", action="store_true",
                       help="Log to wandb")
    parser.add_argument("--wandb-project", type=str, default="alphazero-trading",
                       help="Wandb project name")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Run name for wandb")
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--save-freq", type=int, default=50000,
                       help="Checkpoint save frequency")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency")
    parser.add_argument("--log-freq", type=int, default=1000,
                       help="Logging frequency")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda, mps, cpu)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    logger = setup_logger(log_dir="logs")
    
    logger.info("=" * 60)
    logger.info("AlphaZero Trading Agent - Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")
    
    # Load and prepare data
    logger.info("\nLoading data...")
    data_loader = DataLoader()
    data = data_loader.download_all()
    
    logger.info("Processing features...")
    engineer = FeatureEngineer()
    processed = engineer.process_all_symbols(data)
    processed = engineer.add_cross_asset_features(processed)
    
    logger.info("Preprocessing...")
    preprocessor = DataPreprocessor()
    split, sequences = preprocessor.prepare_training_data(processed)
    
    # Create environments
    logger.info("\nCreating environments...")
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
    
    logger.info(f"Train env: {len(train_features)} samples")
    logger.info(f"Val env: {len(val_features)} samples")
    
    # Create agent
    logger.info("\nCreating agent...")
    input_dim = train_features.shape[2] + 10  # Features + portfolio state
    
    agent_config = AgentConfig(
        input_dim=input_dim,
        seq_len=60,
        n_assets=4,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
    )
    
    agent = AlphaZeroAgent(agent_config).to(device)
    logger.info(f"Agent parameters: {count_parameters(agent):,}")
    
    # PPO trainer
    ppo = PPOTrainer(
        agent,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
    )
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_best=3,
        keep_latest=5,
    )
    
    # Resume from checkpoint
    global_step = 0
    if args.resume:
        logger.info(f"\nResuming from {args.resume}")
        checkpoint = checkpoint_manager.load(args.resume, agent, ppo.optimizer, device)
        if checkpoint:
            global_step = checkpoint.get("step", 0)
    
    # Setup wandb
    if args.use_wandb:
        run_name = args.run_name or f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
        wandb.watch(agent, log="all", log_freq=100)
    
    # Training
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    rollout_buffer = RolloutBuffer(device=device)
    metrics_tracker = MetricsTracker()
    
    episode_count = 0
    best_val_sharpe = -float("inf")
    
    obs, info = train_env.reset()
    
    while global_step < args.total_timesteps:
        # Collect rollout
        rollout_buffer.reset()
        agent.eval()
        
        episode_reward = 0
        episode_returns = []
        
        for step in range(args.n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, log_prob, _, value = agent(obs_tensor)
            
            action_np = action.cpu().numpy()[0]
            next_obs, reward, done, truncated, info = train_env.step(action_np)
            
            rollout_buffer.add(
                torch.FloatTensor(obs),
                action.squeeze(0).cpu(),
                reward,
                value.cpu(),
                log_prob.cpu(),
                done or truncated,
            )
            
            episode_reward += reward
            episode_returns.append(info.get("daily_return", 0))
            
            obs = next_obs
            global_step += 1
            
            if done or truncated:
                obs, info = train_env.reset()
                episode_count += 1
        
        # Compute GAE and update
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            next_value = agent.get_value(obs_tensor).cpu()
        
        obs_batch, actions, rewards, values, log_probs, dones = rollout_buffer.get()
        advantages, returns = ppo.compute_gae(rewards, values.squeeze(), dones, next_value.squeeze())
        
        agent.train()
        train_metrics = ppo.update(obs_batch, actions, log_probs.squeeze(), advantages, returns)
        
        # Calculate episode sharpe
        returns_arr = torch.tensor(episode_returns).numpy()
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            episode_sharpe = returns_arr.mean() / returns_arr.std() * (252 ** 0.5)
        else:
            episode_sharpe = 0
        
        metrics_tracker.update({
            "episode_reward": episode_reward,
            "episode_sharpe": episode_sharpe,
            **train_metrics,
        })
        
        # Logging
        if global_step % args.log_freq == 0:
            avg_metrics = metrics_tracker.get_averages()
            
            logger.info(f"\n[Step {global_step:,}] Episode {episode_count}")
            logger.info(f"  Reward: {avg_metrics.get('episode_reward', 0):.2f}")
            logger.info(f"  Sharpe: {avg_metrics.get('episode_sharpe', 0):.2f}")
            logger.info(f"  Policy Loss: {avg_metrics.get('policy_loss', 0):.4f}")
            logger.info(f"  Value Loss: {avg_metrics.get('value_loss', 0):.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "step": global_step,
                    "episode": episode_count,
                    **avg_metrics,
                })
            
            metrics_tracker.reset()
        
        # Evaluation
        if global_step % args.eval_freq == 0:
            logger.info("\n--- Validation ---")
            agent.eval()
            
            val_obs, val_info = val_env.reset()
            val_obs, val_info = val_env.reset()  # Reset to start
            val_env.current_step = 0
            val_env.portfolio.reset()
            
            val_returns = []
            val_done = False
            val_truncated = False
            
            while not val_done and not val_truncated:
                val_obs_tensor = torch.FloatTensor(val_obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    val_action, _, _, _ = agent(val_obs_tensor, deterministic=True)
                val_obs, _, val_done, val_truncated, val_info = val_env.step(val_action.cpu().numpy()[0])
                val_returns.append(val_info.get("daily_return", 0))
            
            val_returns_arr = torch.tensor(val_returns).numpy()
            if len(val_returns_arr) > 1 and val_returns_arr.std() > 0:
                val_sharpe = val_returns_arr.mean() / val_returns_arr.std() * (252 ** 0.5)
            else:
                val_sharpe = 0
            
            val_total_return = val_info.get("cumulative_return", 0)
            val_max_dd = min(val_env.episode_drawdowns) if val_env.episode_drawdowns else 0
            
            logger.info(f"  Return: {val_total_return:.2%}")
            logger.info(f"  Sharpe: {val_sharpe:.2f}")
            logger.info(f"  Max DD: {val_max_dd:.2%}")
            
            if args.use_wandb:
                wandb.log({
                    "step": global_step,
                    "val/total_return": val_total_return,
                    "val/sharpe_ratio": val_sharpe,
                    "val/max_drawdown": val_max_dd,
                })
            
            # Save best
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                checkpoint_manager.save(
                    agent, ppo.optimizer, global_step,
                    {"val_sharpe": val_sharpe, "val_return": val_total_return},
                    is_best=True,
                )
                logger.info(f"  New best model! Sharpe: {val_sharpe:.2f}")
        
        # Checkpointing
        if global_step % args.save_freq == 0:
            checkpoint_manager.save(
                agent, ppo.optimizer, global_step,
                {"episode": episode_count},
                is_best=False,
            )
    
    # Final save
    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info(f"Total episodes: {episode_count}")
    logger.info(f"Best validation Sharpe: {best_val_sharpe:.2f}")
    logger.info("=" * 60)
    
    checkpoint_manager.save(
        agent, ppo.optimizer, global_step,
        {"final": True, "best_val_sharpe": best_val_sharpe},
        is_best=False,
    )
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

