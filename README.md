# AlphaZero Trading System

An AlphaZero-inspired reinforcement learning system for trading major US index ETFs (SPY, QQQ, DIA, IWM) with leverage and risk management.

## Features

- **Hybrid Transformer Architecture**: Combines transformer encoder for temporal pattern recognition with policy/value networks for decision making
- **Continuous Position Sizing**: Learn optimal allocation across 4 major US indexes
- **Leverage Management**: Dynamic leverage between 5x-10x based on market conditions
- **Risk Controls**: Maximum 5% drawdown constraint, position limits, volatility targeting
- **PPO Training**: Proximal Policy Optimization for stable RL training
- **Comprehensive Backtesting**: Compare against buy-and-hold and equal-weight baselines

## Installation

```bash
# Clone the repository
cd alpha-zero-trading

# Install dependencies with UV
uv sync

# Or install with pip
pip install -r requirements.txt
```

## Project Structure

```
alpha-zero-trading/
├── config.py              # Configuration classes
├── train.py               # Main training script
├── evaluate.py            # Evaluation/backtest script
├── data/
│   ├── data_loader.py     # Download and cache market data
│   ├── feature_engineering.py  # Technical indicators
│   └── preprocessing.py   # Data normalization and splits
├── environment/
│   ├── trading_env.py     # Gym-style trading environment
│   └── portfolio_manager.py  # Portfolio tracking with margin
├── models/
│   ├── transformer_encoder.py  # Transformer for state encoding
│   ├── policy_network.py  # Policy head for actions
│   ├── value_network.py   # Value head for returns
│   └── alphazero_agent.py # Combined agent with PPO
├── training/
│   ├── trainer.py         # Training loop
│   ├── metrics.py         # Performance metrics
│   ├── checkpointing.py   # Model saving/loading
│   └── replay_buffer.py   # Experience buffers
├── risk/
│   └── risk_manager.py    # Risk constraints and sizing
├── evaluation/
│   ├── backtest.py        # Backtesting engine
│   ├── metrics.py         # Advanced performance metrics
│   └── visualization.py   # Charts and plots
└── utils/
    ├── logger.py          # Logging utilities
    └── helpers.py         # Helper functions
```

## Usage

### Training

```bash
# Basic training
uv run python train.py --total-timesteps 1000000

# Training with wandb logging
uv run python train.py --use-wandb --wandb-project alphazero-trading

# Custom configuration
uv run python train.py \
    --total-timesteps 2000000 \
    --d-model 256 \
    --n-layers 6 \
    --lr 3e-4 \
    --max-leverage 8 \
    --use-wandb
```

### Evaluation

```bash
# Evaluate trained model
uv run python evaluate.py --checkpoint checkpoints/best_*.pt

# With visualization
uv run python evaluate.py \
    --checkpoint checkpoints/best_*.pt \
    --output-dir results \
    --use-wandb
```

## Configuration

Key parameters in `config.py`:

### Trading Configuration
- `initial_capital`: Starting capital ($10,000 default)
- `max_leverage`: Maximum leverage (10x default)
- `min_leverage`: Minimum leverage (5x default)
- `max_drawdown`: Maximum allowed drawdown (5% default)
- `max_position_size`: Maximum position per asset (50% default)
- `transaction_cost`: Trading cost (0.1% default)

### Model Configuration
- `d_model`: Transformer dimension (128 default)
- `n_heads`: Attention heads (8 default)
- `n_layers`: Transformer layers (4 default)
- `d_ff`: Feed-forward dimension (512 default)

### Training Configuration
- `learning_rate`: Learning rate (1e-4 default)
- `batch_size`: Batch size (64 default)
- `n_epochs`: PPO epochs per update (10 default)
- `clip_epsilon`: PPO clipping (0.2 default)

## Data

The system downloads and caches historical daily data for:
- **SPY**: S&P 500 ETF (since 1993)
- **QQQ**: NASDAQ-100 ETF (since 1999)
- **DIA**: Dow Jones ETF (since 1998)
- **IWM**: Russell 2000 ETF (since 2000)

Features include:
- OHLCV price data
- 60+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Market regime indicators
- Cross-asset correlations

## Risk Management

The system implements multiple risk controls:

1. **Position Limits**: Maximum 50% allocation per asset
2. **Leverage Control**: Dynamic leverage 5x-10x based on volatility
3. **Drawdown Control**: Episode terminates if drawdown exceeds 5%
4. **Volatility Targeting**: Scale positions to target volatility
5. **Margin Monitoring**: Track margin utilization

## Expected Results

With proper training, the model should achieve:
- **Sharpe Ratio**: > 1.5
- **Maximum Drawdown**: < 5%
- **CAGR**: Outperform buy-and-hold
- **Win Rate**: > 50%

## Monitoring with Wandb

Training and evaluation metrics are logged to [Weights & Biases](https://wandb.ai):

- Training loss curves
- Episode rewards and Sharpe ratios
- Position allocations over time
- Leverage usage
- Equity curves
- Comparison vs baselines

## Hardware Requirements

- **Minimum**: 16GB RAM, CPU-only (slow)
- **Recommended**: 32GB RAM, GPU (CUDA) or Apple Silicon (MPS)
- **Training Time**: ~2-3 days for 1M timesteps on GPU

## License

MIT License

