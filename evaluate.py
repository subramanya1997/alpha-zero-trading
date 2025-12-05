#!/usr/bin/env python
"""
Evaluation script for trained AlphaZero Trading Agent
"""
import argparse
from pathlib import Path

import torch
import wandb
import pandas as pd

from config import DEFAULT_CONFIG
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.preprocessing import DataPreprocessor
from environment.trading_env import TradingEnv, EnvConfig
from models.alphazero_agent import AlphaZeroAgent, AgentConfig
from evaluation.backtest import Backtester
from evaluation.metrics import PerformanceMetrics
from evaluation.visualization import Visualizer
from training.checkpointing import CheckpointManager
from utils.helpers import get_device
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero Trading Agent")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint to evaluate")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda, mps, cpu)")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Log to wandb")
    parser.add_argument("--wandb-project", type=str, default="alphazero-trading",
                       help="Wandb project name")
    
    # Model config (should match training)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    
    # Trading config
    parser.add_argument("--initial-capital", type=float, default=10000)
    parser.add_argument("--max-leverage", type=float, default=10)
    parser.add_argument("--min-leverage", type=float, default=5)
    parser.add_argument("--max-drawdown", type=float, default=0.05)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = get_device(args.device)
    logger = setup_logger()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("AlphaZero Trading Agent - Evaluation")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Load data
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
    
    # Create test environment
    env_config = EnvConfig(
        initial_capital=args.initial_capital,
        max_leverage=args.max_leverage,
        min_leverage=args.min_leverage,
        max_drawdown=args.max_drawdown,
    )
    
    test_features, test_prices, test_returns, test_dates = sequences["test"]
    test_env = TradingEnv(test_features, test_prices, test_returns, test_dates, env_config)
    
    logger.info(f"Test period: {split.test_dates[0]} to {split.test_dates[1]}")
    logger.info(f"Test samples: {len(test_features)}")
    
    # Create agent and load checkpoint
    logger.info("\nLoading agent...")
    input_dim = test_features.shape[2] + 10
    
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
    
    checkpoint_manager = CheckpointManager()
    checkpoint = checkpoint_manager.load(args.checkpoint, agent, device=device)
    
    if checkpoint:
        logger.info(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    else:
        logger.error("Failed to load checkpoint!")
        return
    
    # Run backtest
    logger.info("\n" + "=" * 60)
    logger.info("Running backtest...")
    logger.info("=" * 60)
    
    backtester = Backtester(agent, device)
    result = backtester.run(test_env, deterministic=True)
    
    # Print results
    logger.info("\n--- Agent Performance ---")
    logger.info(f"Total Return: {result.total_return:.2%}")
    logger.info(f"CAGR: {result.cagr:.2%}")
    logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    logger.info(f"Sortino Ratio: {result.sortino_ratio:.2f}")
    logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
    logger.info(f"Volatility: {result.volatility:.2%}")
    logger.info(f"Calmar Ratio: {result.calmar_ratio:.2f}")
    logger.info(f"Win Rate: {result.win_rate:.2%}")
    logger.info(f"Profit Factor: {result.profit_factor:.2f}")
    logger.info(f"Trades: {result.n_trades}")
    logger.info(f"Avg Leverage: {result.avg_leverage:.2f}x")
    logger.info(f"Max Leverage: {result.max_leverage:.2f}x")
    
    # Run baselines
    logger.info("\n--- Baseline Performance ---")
    baselines = backtester.run_baselines(test_env)
    
    for name, baseline_result in baselines.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Return: {baseline_result.total_return:.2%}")
        logger.info(f"  Sharpe: {baseline_result.sharpe_ratio:.2f}")
        logger.info(f"  Max DD: {baseline_result.max_drawdown:.2%}")
    
    # Comparison table
    comparison = backtester.compare_results(result, baselines)
    logger.info("\n--- Comparison ---")
    logger.info(comparison.to_string())
    
    # Save comparison to CSV
    comparison.to_csv(output_dir / "comparison.csv")
    
    # Additional metrics
    logger.info("\n--- Additional Metrics ---")
    returns_arr = torch.tensor(result.daily_returns).numpy()
    
    if "buy_hold_spy" in baselines:
        benchmark_returns = torch.tensor(baselines["buy_hold_spy"].daily_returns).numpy()
        if len(benchmark_returns) == len(returns_arr):
            advanced_metrics = PerformanceMetrics.calculate_comprehensive(
                returns_arr, benchmark_returns
            )
            for name, value in advanced_metrics.items():
                logger.info(f"  {name}: {value:.4f}")
    
    # Create visualizations
    logger.info("\n--- Creating Visualizations ---")
    viz = Visualizer()
    
    figures = viz.create_full_report(
        result,
        baselines,
        save_dir=str(output_dir / "plots"),
    )
    logger.info(f"Saved {len(figures)} plots to {output_dir / 'plots'}")
    
    # Save detailed results
    results_dict = {
        "total_return": result.total_return,
        "cagr": result.cagr,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "max_drawdown": result.max_drawdown,
        "volatility": result.volatility,
        "calmar_ratio": result.calmar_ratio,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "n_trades": result.n_trades,
        "avg_leverage": result.avg_leverage,
        "max_leverage": result.max_leverage,
    }
    
    pd.Series(results_dict).to_csv(output_dir / "metrics.csv")
    
    # Save time series data
    ts_data = pd.DataFrame({
        "date": result.dates,
        "portfolio_value": result.portfolio_values[:len(result.dates)],
        "daily_return": [0] + result.daily_returns,
        "drawdown": result.drawdowns[:len(result.dates)],
        "leverage": result.leverages[:len(result.dates)],
    })
    ts_data.to_csv(output_dir / "timeseries.csv", index=False)
    
    # Log to wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"eval-{Path(args.checkpoint).stem}",
            config=vars(args),
        )
        
        wandb.log(results_dict)
        
        # Log plots
        for i, fig in enumerate(figures):
            if fig:
                wandb.log({f"plot_{i}": wandb.Image(fig)})
        
        # Log comparison table
        wandb.log({"comparison": wandb.Table(dataframe=comparison.reset_index())})
        
        wandb.finish()
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

