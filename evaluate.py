#!/usr/bin/env python
"""
Evaluation script for trained AlphaZero Trading Agent
Enhanced with comprehensive wandb logging and visualizations
"""
import argparse
import os
from pathlib import Path
from datetime import datetime

# Fix matplotlib backend BEFORE importing (fixes Colab/Jupyter issues)
if os.environ.get('MPLBACKEND', '').startswith('module://'):
    os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch
import wandb
import pandas as pd

from config import DEFAULT_CONFIG
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.preprocessing import DataPreprocessor
from environment.trading_env import TradingEnv, EnvConfig
from models.alphazero_agent import AlphaZeroAgent, AgentConfig
from evaluation.backtest import Backtester, BacktestResult
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
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    
    # Trading config
    parser.add_argument("--initial-capital", type=float, default=10000)
    parser.add_argument("--max-leverage", type=float, default=10)
    parser.add_argument("--min-leverage", type=float, default=5)
    parser.add_argument("--max-drawdown", type=float, default=0.05)
    
    return parser.parse_args()


def create_equity_comparison_plot(agent_result, baselines, title="Equity Curve Comparison"):
    """Create equity curve comparison with all strategies"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Normalize all to start at 1
    agent_values = np.array(agent_result.portfolio_values)
    agent_norm = agent_values / agent_values[0]
    ax.plot(agent_norm, label=f"Agent ({agent_result.total_return:.1%})", 
            linewidth=2.5, color='#2ecc71')
    
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    for (name, result), color in zip(baselines.items(), colors):
        values = np.array(result.portfolio_values)
        norm = values / values[0]
        ax.plot(norm, label=f"{name} ({result.total_return:.1%})", 
                linewidth=1.5, color=color, alpha=0.7)
    
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Portfolio Value (Normalized)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return fig


def create_monthly_returns_heatmap(result, title="Monthly Returns Heatmap"):
    """Create monthly returns heatmap"""
    if not result.dates or None in result.dates:
        return None
    
    try:
        # Create DataFrame
        df = pd.DataFrame({
            'date': result.dates[1:],
            'return': result.daily_returns
        })
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Monthly returns
        monthly = df.resample('M')['return'].apply(lambda x: (1 + x).prod() - 1)
        
        if len(monthly) < 2:
            return None
        
        # Pivot
        monthly_df = pd.DataFrame(monthly)
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=0, ax=ax, cbar_kws={'label': 'Return (%)'})
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error creating monthly heatmap: {e}")
        return None


def create_positions_timeline(result, symbols):
    """Create position allocation timeline"""
    if not result.positions:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Convert positions to DataFrame
    pos_data = {s: [] for s in symbols}
    for pos in result.positions:
        for s in symbols:
            pos_data[s].append(pos.get(s, 0))
    
    # Create stacked area plot
    x = range(len(result.positions))
    ax.stackplot(x, *[pos_data[s] for s in symbols], 
                 labels=symbols, alpha=0.7)
    
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Position Weight', fontsize=12)
    ax.set_title('Position Allocations Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_drawdown_analysis(result):
    """Create detailed drawdown analysis plot"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Equity curve
    ax1 = axes[0]
    values = np.array(result.portfolio_values)
    ax1.plot(values, color='#2ecc71', linewidth=1.5)
    ax1.fill_between(range(len(values)), values, alpha=0.3, color='#2ecc71')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.set_title('Equity Curve & Drawdown Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2 = axes[1]
    drawdowns = np.array(result.drawdowns) * 100
    ax2.fill_between(range(len(drawdowns)), drawdowns, 0, 
                     color='#e74c3c', alpha=0.5)
    ax2.plot(drawdowns, color='#e74c3c', linewidth=1)
    
    # Highlight max drawdown
    max_dd_idx = np.argmin(drawdowns)
    max_dd = drawdowns[max_dd_idx]
    ax2.axhline(y=max_dd, color='red', linestyle='--', alpha=0.7)
    ax2.scatter([max_dd_idx], [max_dd], color='red', s=100, zorder=5,
               label=f'Max DD: {max_dd:.1f}%')
    
    ax2.set_xlabel('Trading Days', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_leverage_analysis(result):
    """Create leverage usage analysis"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Leverage over time
    ax1 = axes[0]
    leverages = np.array(result.leverages)
    ax1.plot(leverages, color='#3498db', linewidth=1)
    ax1.axhline(y=np.mean(leverages), color='orange', linestyle='--',
               label=f'Avg: {np.mean(leverages):.2f}x')
    ax1.set_xlabel('Trading Days', fontsize=12)
    ax1.set_ylabel('Leverage (x)', fontsize=12)
    ax1.set_title('Leverage Usage Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Leverage distribution
    ax2 = axes[1]
    ax2.hist(leverages, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(leverages), color='orange', linestyle='--',
               label=f'Mean: {np.mean(leverages):.2f}x')
    ax2.axvline(x=np.median(leverages), color='green', linestyle='--',
               label=f'Median: {np.median(leverages):.2f}x')
    ax2.set_xlabel('Leverage (x)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Leverage Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_returns_analysis(result):
    """Create returns distribution analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    returns = np.array(result.daily_returns) * 100
    
    # Histogram
    ax1 = axes[0]
    n, bins, patches = ax1.hist(returns, bins=50, density=True, alpha=0.7)
    
    # Color positive/negative
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor('#e74c3c')
        else:
            patch.set_facecolor('#2ecc71')
    
    # Normal overlay
    mu, sigma = np.mean(returns), np.std(returns)
    x = np.linspace(min(returns), max(returns), 100)
    normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax1.plot(x, normal, 'k--', linewidth=2, label=f'Normal (μ={mu:.2f}%, σ={sigma:.2f}%)')
    
    ax1.set_xlabel('Daily Return (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # QQ plot
    ax2 = axes[1]
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (vs Normal)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_trade_analysis_table(result, n_rows=50):
    """Create trade analysis DataFrame for wandb table"""
    if not result.positions or not result.dates:
        return None
    
    trades = []
    prev_pos = {s: 0 for s in result.positions[0].keys()} if result.positions else {}
    
    for i, (pos, date, ret, dd, lev) in enumerate(zip(
        result.positions[:n_rows],
        result.dates[1:n_rows+1] if len(result.dates) > 1 else [None] * n_rows,
        result.daily_returns[:n_rows],
        result.drawdowns[1:n_rows+1],
        result.leverages[1:n_rows+1],
    )):
        # Calculate position changes
        trades.append({
            'step': i,
            'date': str(date)[:10] if date else '',
            'SPY': f"{pos.get('SPY', 0):.2%}",
            'QQQ': f"{pos.get('QQQ', 0):.2%}",
            'DIA': f"{pos.get('DIA', 0):.2%}",
            'IWM': f"{pos.get('IWM', 0):.2%}",
            'total_exposure': f"{sum(pos.values()):.2%}",
            'leverage': f"{lev:.2f}x",
            'daily_return': f"{ret:.2%}",
            'drawdown': f"{dd:.2%}",
            'portfolio_value': f"${result.portfolio_values[i+1]:,.2f}",
        })
        prev_pos = pos
    
    return pd.DataFrame(trades)


def create_metrics_comparison_table(agent_result, baselines):
    """Create comprehensive metrics comparison table"""
    data = {
        'Metric': [
            'Total Return',
            'CAGR',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Max Drawdown',
            'Volatility',
            'Calmar Ratio',
            'Win Rate',
            'Profit Factor',
        ],
        'Agent': [
            f"{agent_result.total_return:.2%}",
            f"{agent_result.cagr:.2%}",
            f"{agent_result.sharpe_ratio:.2f}",
            f"{agent_result.sortino_ratio:.2f}",
            f"{agent_result.max_drawdown:.2%}",
            f"{agent_result.volatility:.2%}",
            f"{agent_result.calmar_ratio:.2f}",
            f"{agent_result.win_rate:.2%}",
            f"{agent_result.profit_factor:.2f}",
        ]
    }
    
    for name, result in baselines.items():
        data[name] = [
            f"{result.total_return:.2%}",
            f"{result.cagr:.2%}",
            f"{result.sharpe_ratio:.2f}",
            f"{result.sortino_ratio:.2f}",
            f"{result.max_drawdown:.2%}",
            f"{result.volatility:.2%}",
            f"{result.calmar_ratio:.2f}" if hasattr(result, 'calmar_ratio') else "N/A",
            f"{result.win_rate:.2%}" if hasattr(result, 'win_rate') else "N/A",
            f"{result.profit_factor:.2f}" if hasattr(result, 'profit_factor') else "N/A",
        ]
    
    return pd.DataFrame(data)


def log_comprehensive_to_wandb(result, baselines, symbols, output_dir):
    """Log all evaluation data comprehensively to wandb"""
    
    # 1. Summary metrics
    wandb.log({
        "eval/total_return": result.total_return,
        "eval/cagr": result.cagr,
        "eval/sharpe_ratio": result.sharpe_ratio,
        "eval/sortino_ratio": result.sortino_ratio,
        "eval/max_drawdown": result.max_drawdown,
        "eval/volatility": result.volatility,
        "eval/calmar_ratio": result.calmar_ratio,
        "eval/win_rate": result.win_rate,
        "eval/profit_factor": result.profit_factor,
        "eval/n_trades": result.n_trades,
        "eval/avg_leverage": result.avg_leverage,
        "eval/max_leverage": result.max_leverage,
    })
    
    # 2. Equity curve comparison
    equity_fig = create_equity_comparison_plot(result, baselines)
    wandb.log({"charts/equity_comparison": wandb.Image(equity_fig)})
    equity_fig.savefig(output_dir / "equity_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(equity_fig)
    
    # 3. Position allocations
    pos_fig = create_positions_timeline(result, symbols)
    if pos_fig:
        wandb.log({"charts/positions_timeline": wandb.Image(pos_fig)})
        pos_fig.savefig(output_dir / "positions_timeline.png", dpi=150, bbox_inches='tight')
        plt.close(pos_fig)
    
    # 4. Drawdown analysis
    dd_fig = create_drawdown_analysis(result)
    wandb.log({"charts/drawdown_analysis": wandb.Image(dd_fig)})
    dd_fig.savefig(output_dir / "drawdown_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(dd_fig)
    
    # 5. Leverage analysis
    lev_fig = create_leverage_analysis(result)
    wandb.log({"charts/leverage_analysis": wandb.Image(lev_fig)})
    lev_fig.savefig(output_dir / "leverage_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(lev_fig)
    
    # 6. Returns analysis
    ret_fig = create_returns_analysis(result)
    wandb.log({"charts/returns_analysis": wandb.Image(ret_fig)})
    ret_fig.savefig(output_dir / "returns_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(ret_fig)
    
    # 7. Monthly returns heatmap
    monthly_fig = create_monthly_returns_heatmap(result)
    if monthly_fig:
        wandb.log({"charts/monthly_returns": wandb.Image(monthly_fig)})
        monthly_fig.savefig(output_dir / "monthly_returns.png", dpi=150, bbox_inches='tight')
        plt.close(monthly_fig)
    
    # 8. Trade analysis table
    trades_df = create_trade_analysis_table(result, n_rows=100)
    if trades_df is not None:
        wandb.log({"tables/trades": wandb.Table(dataframe=trades_df)})
        trades_df.to_csv(output_dir / "trades.csv", index=False)
    
    # 9. Metrics comparison table
    comparison_df = create_metrics_comparison_table(result, baselines)
    wandb.log({"tables/metrics_comparison": wandb.Table(dataframe=comparison_df)})
    comparison_df.to_csv(output_dir / "metrics_comparison.csv", index=False)
    
    # 10. Time series data
    ts_df = pd.DataFrame({
        'step': range(len(result.portfolio_values)),
        'portfolio_value': result.portfolio_values,
        'daily_return': [0] + result.daily_returns,
        'cumulative_return': np.cumprod([1] + [1 + r for r in result.daily_returns]) - 1,
        'drawdown': result.drawdowns,
        'leverage': result.leverages,
    })
    
    # Log as line plots
    wandb.log({
        "timeseries/portfolio_value": wandb.plot.line_series(
            xs=list(range(len(result.portfolio_values))),
            ys=[result.portfolio_values],
            keys=["Portfolio Value"],
            title="Portfolio Value Over Time",
            xname="Step"
        ),
    })


def main():
    args = parse_args()
    
    # Setup
    device = get_device(args.device)
    logger = setup_logger()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    
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
    
    symbols = env_config.symbols
    
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
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"eval-{Path(args.checkpoint).stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                **vars(args),
                "test_period": split.test_dates,
                "test_samples": len(test_features),
            },
        )
    
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
    
    # Additional metrics
    logger.info("\n--- Additional Metrics ---")
    returns_arr = np.array(result.daily_returns)
    
    if "buy_hold_spy" in baselines:
        benchmark_returns = np.array(baselines["buy_hold_spy"].daily_returns)
        if len(benchmark_returns) == len(returns_arr):
            advanced_metrics = PerformanceMetrics.calculate_comprehensive(
                returns_arr, benchmark_returns
            )
            for name, value in advanced_metrics.items():
                logger.info(f"  {name}: {value:.4f}")
    
    # Log to wandb with comprehensive visualizations
    if args.use_wandb:
        logger.info("\n--- Logging to Wandb ---")
        log_comprehensive_to_wandb(result, baselines, symbols, output_dir / "plots")
    
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
        "step": range(len(result.portfolio_values)),
        "portfolio_value": result.portfolio_values,
        "daily_return": [0] + result.daily_returns,
        "drawdown": result.drawdowns,
        "leverage": result.leverages,
    })
    ts_data.to_csv(output_dir / "timeseries.csv", index=False)
    
    if args.use_wandb:
        wandb.finish()
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
