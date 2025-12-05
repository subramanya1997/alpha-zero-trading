"""
Helper utilities
"""
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(preferred: Optional[str] = None) -> str:
    """
    Get best available device
    
    Args:
        preferred: Preferred device ('cuda', 'mps', 'cpu')
        
    Returns:
        Device string
    """
    if preferred:
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif preferred == "cpu":
            return "cpu"
    
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_number(num: float) -> str:
    """Format large numbers with K/M/B suffix"""
    if abs(num) >= 1e9:
        return f"{num/1e9:.1f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.1f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return f"{num:.1f}"


def ensure_dir(path: str) -> str:
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)
    return path


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str = "", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' or 'min' - maximize or minimize metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, metric: float) -> bool:
        score = metric if self.mode == "max" else -metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.should_stop


if __name__ == "__main__":
    # Test utilities
    print("=== Utility Tests ===")
    
    # Set seed
    set_seed(42)
    print(f"Random seed set")
    
    # Get device
    device = get_device()
    print(f"Device: {device}")
    
    # Format tests
    print(f"\nFormat time: {format_time(3723)} = 1.0h")
    print(f"Format number: {format_number(1234567)} = 1.2M")
    
    # Average meter
    meter = AverageMeter("loss")
    for i in range(10):
        meter.update(random.random())
    print(f"\nAverage meter: {meter}")
    
    # Early stopping
    early_stop = EarlyStopping(patience=3, mode="max")
    values = [0.5, 0.6, 0.7, 0.65, 0.64, 0.63, 0.62]
    for i, v in enumerate(values):
        stop = early_stop(v)
        print(f"Step {i}: metric={v:.2f}, counter={early_stop.counter}, stop={stop}")

