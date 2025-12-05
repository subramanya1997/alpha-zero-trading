#!/usr/bin/env python
"""
Main training script for AlphaZero Trading Agent
Enhanced with comprehensive wandb logging
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
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
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    
    # Model (defaults updated for Gated Transformer)
    parser.add_argument("--d-model", type=int, default=256,
                       help="Transformer model dimension")
    parser.add_argument("--n-heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=6,
                       help="Number of transformer layers")
    parser.add_argument("--model-type", type=str, default="gated_transformer",
                       choices=["transformer", "gated_transformer", "lstm", "tcn", "hybrid"],
                       help="Model architecture type")
    
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


def create_equity_curve_plot(values, title="Portfolio Equity Curve"):
    """Create equity curve plot for wandb"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(values, linewidth=2, color='#2ecc71')
    ax.fill_between(range(len(values)), values, alpha=0.3, color='#2ecc71')
    ax.set_xlabel('Step')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_positions_plot(positions_history, symbols):
    """Create stacked area plot for position allocations"""
    if not positions_history:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Convert to arrays
    data = {s: [] for s in symbols}
    for pos in positions_history:
        for s in symbols:
            data[s].append(pos.get(s, 0))
    
    # Stack plot
    x = range(len(positions_history))
    ax.stackplot(x, *[data[s] for s in symbols], labels=symbols, alpha=0.7)
    ax.legend(loc='upper right')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position Weight')
    ax.set_title('Position Allocations Over Time')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_drawdown_plot(drawdowns):
    """Create drawdown plot"""
    fig, ax = plt.subplots(figsize=(10, 4))
    drawdowns_pct = [d * 100 for d in drawdowns]
    ax.fill_between(range(len(drawdowns_pct)), drawdowns_pct, 0, 
                   color='#e74c3c', alpha=0.3)
    ax.plot(drawdowns_pct, color='#e74c3c', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Portfolio Drawdown')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_leverage_plot(leverages):
    """Create leverage usage plot"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(leverages, color='#3498db', linewidth=1)
    ax.axhline(y=np.mean(leverages), color='orange', linestyle='--', 
               label=f'Avg: {np.mean(leverages):.2f}x')
    ax.set_xlabel('Step')
    ax.set_ylabel('Leverage (x)')
    ax.set_title('Leverage Usage Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def run_full_evaluation(agent, env, device, symbols):
    """Run full evaluation episode and collect all data for visualization"""
    agent.eval()
    
    # Reset environment to start
    obs, info = env.reset()
    obs, info = env.reset()
    env.current_step = 0
    env.portfolio.reset()
    
    # Tracking data
    portfolio_values = [env.portfolio.total_value]
    daily_returns = []
    drawdowns = [0.0]
    positions_history = []
    leverages = [0.0]
    dates = []
    actions_taken = []
    
    done = False
    truncated = False
    step = 0
    
    while not done and not truncated:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action, _, _, _ = agent(obs_tensor, deterministic=True)
        
        action_np = action.cpu().numpy()[0]
        obs, reward, done, truncated, info = env.step(action_np)
        
        # Track data
        portfolio_values.append(info["total_value"])
        daily_returns.append(info["daily_return"])
        drawdowns.append(info["drawdown"])
        positions_history.append(info["positions"])
        leverages.append(info["leverage"])
        dates.append(info["date"])
        actions_taken.append({
            "step": step,
            "action": action_np.tolist(),
            "reward": reward,
            "value": info["total_value"],
            "leverage": info["leverage"],
        })
        
        step += 1
    
    # Calculate metrics
    returns_arr = np.array(daily_returns)
    
    metrics = {
        "total_return": (portfolio_values[-1] / portfolio_values[0]) - 1,
        "sharpe_ratio": (returns_arr.mean() / returns_arr.std() * np.sqrt(252)) if returns_arr.std() > 0 else 0,
        "max_drawdown": min(drawdowns),
        "volatility": returns_arr.std() * np.sqrt(252),
        "final_value": portfolio_values[-1],
        "n_steps": len(daily_returns),
    }
    
    return {
        "metrics": metrics,
        "portfolio_values": portfolio_values,
        "daily_returns": daily_returns,
        "drawdowns": drawdowns,
        "positions_history": positions_history,
        "leverages": leverages,
        "dates": dates,
        "actions": actions_taken,
    }


def log_evaluation_to_wandb(eval_data, prefix="val", symbols=None):
    """Log comprehensive evaluation data to wandb"""
    symbols = symbols or ["SPY", "QQQ", "DIA", "IWM"]
    
    # Log metrics
    metrics = eval_data["metrics"]
    wandb.log({
        f"{prefix}/total_return": metrics["total_return"],
        f"{prefix}/sharpe_ratio": metrics["sharpe_ratio"],
        f"{prefix}/max_drawdown": metrics["max_drawdown"],
        f"{prefix}/volatility": metrics["volatility"],
        f"{prefix}/final_value": metrics["final_value"],
    })
    
    # Log equity curve
    equity_fig = create_equity_curve_plot(
        eval_data["portfolio_values"], 
        title=f"{prefix.upper()} Portfolio Equity Curve"
    )
    wandb.log({f"{prefix}/equity_curve": wandb.Image(equity_fig)})
    plt.close(equity_fig)
    
    # Log positions
    if eval_data["positions_history"]:
        positions_fig = create_positions_plot(eval_data["positions_history"], symbols)
        if positions_fig:
            wandb.log({f"{prefix}/positions": wandb.Image(positions_fig)})
            plt.close(positions_fig)
    
    # Log drawdown
    drawdown_fig = create_drawdown_plot(eval_data["drawdowns"])
    wandb.log({f"{prefix}/drawdown": wandb.Image(drawdown_fig)})
    plt.close(drawdown_fig)
    
    # Log leverage
    leverage_fig = create_leverage_plot(eval_data["leverages"])
    wandb.log({f"{prefix}/leverage": wandb.Image(leverage_fig)})
    plt.close(leverage_fig)
    
    # Log trade table
    if eval_data["actions"]:
        actions_df = pd.DataFrame(eval_data["actions"][:100])  # First 100 actions
        wandb.log({f"{prefix}/trades_table": wandb.Table(dataframe=actions_df)})
    
    # Log returns distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.array(eval_data["daily_returns"]) * 100, bins=50, 
            color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Daily Returns Distribution')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({f"{prefix}/returns_distribution": wandb.Image(fig)})
    plt.close(fig)


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
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Model size: d={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
    
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
    
    symbols = env_config.symbols
    
    logger.info(f"Train env: {len(train_features)} samples")
    logger.info(f"Val env: {len(val_features)} samples")
    logger.info(f"Train period: {split.train_dates}")
    logger.info(f"Val period: {split.val_dates}")
    
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
    n_params = count_parameters(agent)
    logger.info(f"Agent parameters: {n_params:,}")
    
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
    
    # Setup wandb with enhanced config
    if args.use_wandb:
        run_name = args.run_name or f"train-{args.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **vars(args),
                "n_parameters": n_params,
                "train_samples": len(train_features),
                "val_samples": len(val_features),
                "train_period": split.train_dates,
                "val_period": split.val_dates,
            },
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
    
    # Episode tracking for wandb
    episode_values = []
    episode_positions = []
    episode_leverages = []
    episode_drawdowns = []
    
    obs, info = train_env.reset()
    
    while global_step < args.total_timesteps:
        # Collect rollout
        rollout_buffer.reset()
        agent.eval()
        
        episode_reward = 0
        episode_returns = []
        
        # Clear episode tracking
        episode_values = [train_env.portfolio.total_value]
        episode_positions = []
        episode_leverages = [train_env.portfolio.current_leverage]
        episode_drawdowns = [0]
        
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
            
            # Track for visualization
            episode_values.append(info.get("total_value", 0))
            episode_positions.append(info.get("positions", {}))
            episode_leverages.append(info.get("leverage", 0))
            episode_drawdowns.append(info.get("drawdown", 0))
            
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
        returns_arr = np.array(episode_returns)
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            episode_sharpe = returns_arr.mean() / returns_arr.std() * (252 ** 0.5)
        else:
            episode_sharpe = 0
        
        metrics_tracker.update({
            "episode_reward": episode_reward,
            "episode_sharpe": episode_sharpe,
            "episode_return": (episode_values[-1] / episode_values[0]) - 1 if episode_values[0] > 0 else 0,
            "episode_max_dd": min(episode_drawdowns) if episode_drawdowns else 0,
            "episode_avg_leverage": np.mean(episode_leverages) if episode_leverages else 0,
            **train_metrics,
        })
        
        # Logging
        if global_step % args.log_freq == 0:
            avg_metrics = metrics_tracker.get_averages()
            
            logger.info(f"\n[Step {global_step:,}] Episode {episode_count}")
            logger.info(f"  Reward: {avg_metrics.get('episode_reward', 0):.2f}")
            logger.info(f"  Return: {avg_metrics.get('episode_return', 0):.2%}")
            logger.info(f"  Sharpe: {avg_metrics.get('episode_sharpe', 0):.2f}")
            logger.info(f"  Max DD: {avg_metrics.get('episode_max_dd', 0):.2%}")
            logger.info(f"  Avg Leverage: {avg_metrics.get('episode_avg_leverage', 0):.2f}x")
            logger.info(f"  Policy Loss: {avg_metrics.get('policy_loss', 0):.4f}")
            logger.info(f"  Value Loss: {avg_metrics.get('value_loss', 0):.4f}")
            
            if args.use_wandb:
                # Log scalar metrics
                wandb.log({
                    "train/step": global_step,
                    "train/episode": episode_count,
                    "train/reward": avg_metrics.get('episode_reward', 0),
                    "train/return": avg_metrics.get('episode_return', 0),
                    "train/sharpe": avg_metrics.get('episode_sharpe', 0),
                    "train/max_drawdown": avg_metrics.get('episode_max_dd', 0),
                    "train/avg_leverage": avg_metrics.get('episode_avg_leverage', 0),
                    "train/policy_loss": avg_metrics.get('policy_loss', 0),
                    "train/value_loss": avg_metrics.get('value_loss', 0),
                    "train/entropy": avg_metrics.get('entropy', 0),
                    "train/approx_kl": avg_metrics.get('approx_kl', 0),
                    "train/lr": avg_metrics.get('lr', args.lr),
                })
                
                # Log episode visualizations periodically
                if global_step % (args.log_freq * 10) == 0 and len(episode_values) > 10:
                    # Equity curve
                    eq_fig = create_equity_curve_plot(episode_values, "Training Episode Equity")
                    wandb.log({"train/episode_equity": wandb.Image(eq_fig)})
                    plt.close(eq_fig)
                    
                    # Positions
                    if episode_positions:
                        pos_fig = create_positions_plot(episode_positions, symbols)
                        if pos_fig:
                            wandb.log({"train/episode_positions": wandb.Image(pos_fig)})
                            plt.close(pos_fig)
            
            metrics_tracker.reset()
        
        # Full Evaluation
        if global_step % args.eval_freq == 0:
            logger.info("\n--- Full Validation ---")
            
            # Run comprehensive evaluation
            eval_data = run_full_evaluation(agent, val_env, device, symbols)
            metrics = eval_data["metrics"]
            
            logger.info(f"  Return: {metrics['total_return']:.2%}")
            logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Max DD: {metrics['max_drawdown']:.2%}")
            logger.info(f"  Volatility: {metrics['volatility']:.2%}")
            logger.info(f"  Final Value: ${metrics['final_value']:,.2f}")
            
            if args.use_wandb:
                log_evaluation_to_wandb(eval_data, prefix="val", symbols=symbols)
            
            # Save best
            val_sharpe = metrics['sharpe_ratio']
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                checkpoint_manager.save(
                    agent, ppo.optimizer, global_step,
                    {"val_sharpe": val_sharpe, "val_return": metrics['total_return']},
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
    
    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Final Evaluation")
    logger.info("=" * 60)
    
    final_eval = run_full_evaluation(agent, val_env, device, symbols)
    final_metrics = final_eval["metrics"]
    
    logger.info(f"Final Return: {final_metrics['total_return']:.2%}")
    logger.info(f"Final Sharpe: {final_metrics['sharpe_ratio']:.2f}")
    logger.info(f"Final Max DD: {final_metrics['max_drawdown']:.2%}")
    
    if args.use_wandb:
        log_evaluation_to_wandb(final_eval, prefix="final", symbols=symbols)
    
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
