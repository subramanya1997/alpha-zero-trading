"""
Metrics tracking for training
"""
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class MetricsTracker:
    """Tracks and aggregates training metrics"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker
        
        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.episode_metrics: List[Dict[str, float]] = []
    
    def reset(self):
        """Reset current window metrics"""
        self.metrics = defaultdict(list)
    
    def update(self, metrics: Dict[str, float]):
        """
        Update metrics
        
        Args:
            metrics: Dictionary of metric name to value
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.metrics[name].append(value)
    
    def update_episode(self, episode_metrics: Dict[str, float]):
        """Store complete episode metrics"""
        self.episode_metrics.append(episode_metrics)
        if len(self.episode_metrics) > 1000:
            self.episode_metrics = self.episode_metrics[-1000:]
    
    def get_averages(self) -> Dict[str, float]:
        """Get average of all tracked metrics"""
        return {
            name: np.mean(values[-self.window_size:])
            for name, values in self.metrics.items()
            if values
        }
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest value for all metrics"""
        return {
            name: values[-1] if values else 0.0
            for name, values in self.metrics.items()
        }
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                arr = np.array(values[-self.window_size:])
                summary[name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "last": float(arr[-1]),
                }
        return summary


class TradingMetrics:
    """Calculate trading-specific performance metrics"""
    
    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate annualized Sharpe ratio
        
        Args:
            returns: Array of returns
            risk_free: Risk-free rate (annual)
            periods_per_year: Trading periods per year
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free / periods_per_year
        if np.std(returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)
    
    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate annualized Sortino ratio (downside risk only)
        
        Args:
            returns: Array of returns
            risk_free: Risk-free rate (annual)
            periods_per_year: Trading periods per year
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free / periods_per_year
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return np.mean(excess_returns) * np.sqrt(periods_per_year) * 100
        
        downside_std = np.std(downside_returns)
        return np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
    
    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            returns: Array of returns
            
        Returns:
            Maximum drawdown (negative value)
        """
        if len(returns) < 2:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return float(np.min(drawdown))
    
    @staticmethod
    def calmar_ratio(
        returns: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown)
        
        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year
            
        Returns:
            Calmar ratio
        """
        if len(returns) < 2:
            return 0.0
        
        annual_return = np.mean(returns) * periods_per_year
        max_dd = abs(TradingMetrics.max_drawdown(returns))
        
        if max_dd == 0:
            return annual_return * 100
        
        return annual_return / max_dd
    
    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """Calculate win rate (percentage of positive returns)"""
        if len(returns) == 0:
            return 0.0
        return float(np.sum(returns > 0) / len(returns))
    
    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(losses) == 0 or np.sum(np.abs(losses)) == 0:
            return np.sum(gains) * 100 if len(gains) > 0 else 0.0
        
        return np.sum(gains) / np.sum(np.abs(losses))
    
    @staticmethod
    def cagr(
        returns: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Compound Annual Growth Rate
        
        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year
            
        Returns:
            CAGR as decimal
        """
        if len(returns) < 2:
            return 0.0
        
        total_return = np.prod(1 + returns) - 1
        n_years = len(returns) / periods_per_year
        
        if n_years <= 0 or total_return <= -1:
            return 0.0
        
        return (1 + total_return) ** (1 / n_years) - 1
    
    @staticmethod
    def volatility(
        returns: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized volatility"""
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns) * np.sqrt(periods_per_year))
    
    @staticmethod
    def calculate_all(
        returns: np.ndarray,
        risk_free: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate all metrics
        
        Args:
            returns: Array of returns
            risk_free: Risk-free rate
            
        Returns:
            Dictionary of all metrics
        """
        tm = TradingMetrics
        return {
            "sharpe_ratio": tm.sharpe_ratio(returns, risk_free),
            "sortino_ratio": tm.sortino_ratio(returns, risk_free),
            "max_drawdown": tm.max_drawdown(returns),
            "calmar_ratio": tm.calmar_ratio(returns),
            "win_rate": tm.win_rate(returns),
            "profit_factor": tm.profit_factor(returns),
            "cagr": tm.cagr(returns),
            "volatility": tm.volatility(returns),
            "total_return": float(np.prod(1 + returns) - 1),
            "n_trades": len(returns),
        }


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Simulate returns
    returns = np.random.normal(0.0005, 0.01, 252)  # 1 year of daily returns
    
    print("=== Trading Metrics Test ===")
    print(f"Sample size: {len(returns)} days")
    print(f"Total return: {np.prod(1 + returns) - 1:.2%}")
    
    metrics = TradingMetrics.calculate_all(returns)
    print("\nAll metrics:")
    for name, value in metrics.items():
        if "ratio" in name or "return" in name or "cagr" in name:
            print(f"  {name}: {value:.4f}")
        elif "rate" in name:
            print(f"  {name}: {value:.2%}")
        else:
            print(f"  {name}: {value:.4f}")
    
    # Test tracker
    print("\n=== Metrics Tracker Test ===")
    tracker = MetricsTracker()
    
    for i in range(100):
        tracker.update({
            "reward": np.random.randn() + 0.1,
            "loss": np.random.rand(),
            "sharpe": np.random.randn() * 0.5 + 1,
        })
    
    print("Averages:", tracker.get_averages())
    print("Latest:", tracker.get_latest())

