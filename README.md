# AlphaZero Trading System

An AlphaZero-inspired reinforcement learning system for trading major US index ETFs (SPY, QQQ, DIA, IWM) with leverage and risk management.

## Features

- **Gated Transformer Architecture**: 256-dim, 6-layer Transformer with gating for robust temporal pattern recognition
- **Self-Play Training**: Data augmentation, curriculum learning, and diverse scenario exploration
- **Differential Sharpe Ratio**: Direct optimization of risk-adjusted returns
- **Continuous Position Sizing**: Learn optimal allocation across 4 major US indexes
- **Leverage Management**: Dynamic leverage between 1x-10x based on market conditions
- **Risk Controls**: Maximum 5% drawdown constraint, position limits, volatility targeting
- **PPO Training**: Proximal Policy Optimization for stable RL training
- **Comprehensive Backtesting**: Compare against buy-and-hold and equal-weight baselines
- **Wandb Integration**: Full experiment tracking and visualization

## Installation

```bash
# Clone the repository
git clone git@github.com:subramanya1997/alpha-zero-trading.git
cd alpha-zero-trading

# Install dependencies with UV (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

### Download Data & Train

```bash
# Basic training (standard PPO)
uv run python train.py --total-timesteps 500000 --d-model 256 --n-layers 6

# Self-play training with curriculum learning (recommended)
uv run python train_selfplay.py \
    --total-timesteps 1000000 \
    --n-steps 2048 \
    --d-model 256 \
    --n-layers 6 \
    --use-curriculum \
    --use-augmentation \
    --use-wandb \
    --wandb-project alphazero-trading
```

### Evaluate

```bash
# Evaluate trained model
uv run python evaluate.py --checkpoint checkpoints/best_*.pt --use-wandb
```

## Project Structure

```
alpha-zero-trading/
├── config.py                  # Configuration classes
├── train.py                   # Standard PPO training
├── train_selfplay.py          # Self-play training with augmentation
├── evaluate.py                # Evaluation/backtest script
├── data/
│   ├── data_loader.py         # Download and cache market data
│   ├── feature_engineering.py # Technical indicators
│   └── preprocessing.py       # Data normalization and splits
├── environment/
│   ├── trading_env.py         # Gym-style trading environment
│   └── portfolio_manager.py   # Portfolio tracking with margin
├── models/
│   ├── transformer_encoder.py # Transformer for state encoding
│   ├── policy_network.py      # Policy head for actions
│   ├── value_network.py       # Value head for returns
│   ├── alphazero_agent.py     # Combined agent with PPO
│   └── improved_models.py     # Advanced model architectures
├── training/
│   ├── trainer.py             # Training loop
│   ├── self_play.py           # Self-play components
│   ├── differential_sharpe.py # DSR reward signal
│   ├── metrics.py             # Performance metrics
│   ├── checkpointing.py       # Model saving/loading
│   └── replay_buffer.py       # Experience buffers
├── risk/
│   └── risk_manager.py        # Risk constraints and sizing
├── evaluation/
│   ├── backtest.py            # Backtesting engine
│   ├── metrics.py             # Advanced performance metrics
│   └── visualization.py       # Charts and plots
└── utils/
    ├── logger.py              # Logging utilities
    └── helpers.py             # Helper functions
```

## Training Approaches

### 1. Standard PPO Training (`train.py`)

Basic PPO training with shaped rewards:

```bash
uv run python train.py \
    --total-timesteps 1000000 \
    --n-steps 2048 \
    --d-model 256 \
    --n-layers 6 \
    --lr 3e-4 \
    --use-wandb
```

### 2. Self-Play Training (`train_selfplay.py`)

Advanced training with:
- **Curriculum Learning**: Start with relaxed constraints, gradually increase difficulty
- **Data Augmentation**: Noise injection, time warping, feature scaling
- **Differential Sharpe Ratio**: Direct optimization of risk-adjusted returns
- **Prioritized Experience Replay**: Focus on important transitions

```bash
uv run python train_selfplay.py \
    --total-timesteps 2000000 \
    --n-steps 2048 \
    --d-model 256 \
    --n-layers 6 \
    --use-curriculum \
    --use-augmentation \
    --initial-leverage-mult 0.3 \
    --use-wandb
```

## Model Architecture

### Recommended: Gated Transformer
- **Dimension**: 256
- **Layers**: 6
- **Heads**: 8
- **Parameters**: ~6.5M

The Gated Transformer uses learnable gating mechanisms to control information flow, improving robustness to noisy financial data.

### Alternatives (in `models/improved_models.py`):
- **Decision Transformer**: Goal-conditioned, can target specific Sharpe ratios
- **Temporal Convolutional Network (TCN)**: Faster inference, good for production
- **LSTM**: Simpler but effective baseline
- **Hybrid CNN+Transformer**: Best of both worlds

## Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--total-timesteps` | 500000 | Total environment steps |
| `--n-steps` | 2048 | Steps per rollout |
| `--d-model` | 256 | Transformer dimension |
| `--n-layers` | 6 | Number of Transformer layers |
| `--lr` | 3e-4 | Learning rate |
| `--max-leverage` | 10 | Maximum leverage allowed |
| `--max-drawdown` | 0.10 | Max drawdown before termination |
| `--use-curriculum` | True | Enable curriculum learning |
| `--use-augmentation` | True | Enable data augmentation |

## Data Pipeline

### Features (248 total)

The system creates a rich feature set for each trading day:

1. **Price Features** (per asset): Returns, log returns, OHLC ratios
2. **Technical Indicators** (per asset):
   - RSI (14), MACD (12,26,9), ATR (14)
   - Bollinger Bands (20,2), Moving Averages (5,10,20,50)
   - Volume indicators, momentum
3. **Cross-Asset Features**: Correlations between all pairs
4. **Market Regime**: VIX levels, trend indicators

### Data Splits

- **Train**: 2001-2018 (70%)
- **Validation**: 2018-2022 (15%)
- **Test**: 2022-2025 (15%)

## Challenges & Solutions

### Challenge 1: Low Signal-to-Noise Ratio

Financial markets have very low signal-to-noise ratios (~0.1 Sharpe is achievable for daily returns).

**Solutions:**
- Use Differential Sharpe Ratio (DSR) for reward shaping
- Longer episode lengths with curriculum learning
- Multiple timescale averaging

### Challenge 2: Non-Stationarity

Market regimes change over time.

**Solutions:**
- Data augmentation to simulate different regimes
- Cross-asset correlation features
- Rolling normalization

### Challenge 3: Sparse Rewards

Good trading strategies may have only slightly positive returns.

**Solutions:**
- Survival bonus (reward for staying in the game)
- Shaped rewards based on risk-adjusted metrics
- Curriculum learning (start easy, get harder)

## Risk Management

The system implements multiple risk controls:

1. **Position Limits**: Maximum 50% allocation per asset
2. **Leverage Control**: Dynamic leverage 1x-10x based on volatility
3. **Drawdown Control**: Episode terminates if drawdown exceeds limit
4. **Warmup Period**: 50 steps before early termination is enforced
5. **Margin Monitoring**: Track margin utilization

## Monitoring with Wandb

Training metrics logged:
- Policy loss, value loss, entropy
- Episode rewards, lengths
- Rolling Sharpe ratio
- Portfolio value, leverage, drawdown
- Position allocations

Evaluation metrics logged:
- Equity curve
- Drawdown chart
- Comparison vs baselines (SPY, equal-weight)
- Trade analysis

## GPU Recommendations

| GPU | Training Time (1M steps) | Notes |
|-----|--------------------------|-------|
| Apple M1/M2/M3 (MPS) | 8-12 hours | Good for development |
| RTX 4090 | 3-4 hours | Recommended for serious training |
| A100 (40GB) | 2-3 hours | Best for large experiments |
| CPU only | 24-48 hours | Not recommended |

## Expected Results

With proper training (1M+ steps), the model should achieve:
- **Sharpe Ratio**: 0.5-1.5 (validation)
- **Maximum Drawdown**: < 10%
- **Win Rate**: ~50-55%

**Note**: Financial RL is challenging. Even small positive Sharpe ratios are meaningful given the noise in market data.

## Troubleshooting

### Sharpe Always 0

This usually means episodes are too short. Try:
1. Increase `--max-drawdown` to 0.15 or 0.20
2. Use `--use-curriculum` to start with relaxed constraints
3. Increase `--warmup-steps` in the environment config

### High Value Loss

The value function is hard to learn in financial RL. Try:
1. Decrease learning rate: `--lr 1e-4`
2. Increase entropy coefficient for more exploration
3. Use normalized advantage estimation

### Memory Issues

For large models or long training:
1. Reduce `--batch-size`
2. Use gradient checkpointing
3. Reduce `--n-steps`

## References

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [Differential Sharpe Ratio](https://arxiv.org/abs/1901.00137)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [Decision Transformer](https://arxiv.org/abs/2106.01345)

## License

MIT License
