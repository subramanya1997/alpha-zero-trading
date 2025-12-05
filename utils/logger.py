"""
Logging utilities
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "alphazero_trading",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        console: Whether to also log to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"{name}_{timestamp}.log"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """Logger for training progress"""
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
    ):
        self.logger = setup_logger(log_dir=log_dir)
        self.use_wandb = use_wandb
        
        self.step = 0
        self.episode = 0
    
    def log_step(self, metrics: dict, step: Optional[int] = None):
        """Log training step metrics"""
        step = step or self.step
        self.step = step
        
        msg = f"Step {step}"
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" | {key}: {value:.4f}"
            else:
                msg += f" | {key}: {value}"
        
        self.logger.info(msg)
    
    def log_episode(self, metrics: dict, episode: Optional[int] = None):
        """Log episode metrics"""
        episode = episode or self.episode
        self.episode = episode
        
        msg = f"Episode {episode}"
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" | {key}: {value:.4f}"
            else:
                msg += f" | {key}: {value}"
        
        self.logger.info(msg)
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)

