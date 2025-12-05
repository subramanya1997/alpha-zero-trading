"""
Configuration for AlphaZero Trading System
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TradingConfig:
    """Trading environment configuration"""
    # Capital and Leverage
    initial_capital: float = 10_000.0
    min_leverage: float = 5.0
    max_leverage: float = 10.0
    
    # Risk Management
    max_drawdown: float = 0.05  # 5% maximum drawdown (hard limit)
    max_position_size: float = 0.50  # 50% max per single position
    transaction_cost: float = 0.001  # 0.1% per trade
    margin_maintenance: float = 0.25  # 25% maintenance margin
    
    # Indexes to trade
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "DIA", "IWM"])
    
    # Data parameters
    lookback_window: int = 60  # Days of history for state
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Episode parameters
    episode_length: int = 252  # ~1 trading year
    
    
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Encoder type: "gated_transformer", "transformer", "lstm", "tcn", "hybrid"
    encoder_type: str = "gated_transformer"
    
    # Transformer/Encoder settings (optimized for Gated Transformer)
    d_model: int = 256  # Model dimension (increased from 128)
    n_heads: int = 8  # Number of attention heads
    n_layers: int = 6  # Number of transformer layers (increased from 4)
    d_ff: int = 1024  # Feed-forward dimension (increased from 512)
    dropout: float = 0.1
    
    # Policy/Value heads
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    
    # Number of actions (4 indexes + cash allocation)
    n_assets: int = 4
    

@dataclass  
class TrainingConfig:
    """Training configuration"""
    # PPO hyperparameters
    learning_rate: float = 3e-4  # Increased for faster convergence
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Training loop
    n_epochs: int = 10  # PPO epochs per iteration
    batch_size: int = 64
    n_steps: int = 2048  # Steps per rollout
    total_timesteps: int = 1_000_000
    
    # Checkpointing
    save_freq: int = 50_000
    eval_freq: int = 10_000
    log_freq: int = 1_000
    
    # Device
    device: str = "mps"  # Use MPS for Apple Silicon, change to "cuda" for NVIDIA
    
    # Wandb
    project_name: str = "alphazero-trading"
    run_name: str = None  # Auto-generated if None
    

@dataclass
class Config:
    """Main configuration combining all configs"""
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    data_dir: str = "data/cache"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


# Default configuration
DEFAULT_CONFIG = Config()

