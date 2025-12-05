"""
Performance metrics for evaluation
"""
import numpy as np
from typing import Dict, List, Tuple, Optional


class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def alpha_beta(
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        risk_free: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Calculate alpha and beta vs benchmark
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            risk_free: Risk-free rate
            
        Returns:
            Tuple of (alpha, beta)
        """
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0, 1.0
        
        # Calculate beta (covariance / variance)
        cov = np.cov(returns, benchmark_returns)[0, 1]
        var = np.var(benchmark_returns)
        beta = cov / var if var > 0 else 1.0
        
        # Calculate alpha
        rf_daily = risk_free / 252
        strategy_excess = np.mean(returns) - rf_daily
        benchmark_excess = np.mean(benchmark_returns) - rf_daily
        alpha = (strategy_excess - beta * benchmark_excess) * 252  # Annualized
        
        return alpha, beta
    
    @staticmethod
    def information_ratio(
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """
        Calculate information ratio
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)
        
        if tracking_error == 0:
            return 0.0
        
        return np.mean(active_returns) * 252 / tracking_error
    
    @staticmethod
    def treynor_ratio(
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        risk_free: float = 0.0,
    ) -> float:
        """
        Calculate Treynor ratio
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            risk_free: Risk-free rate
            
        Returns:
            Treynor ratio
        """
        _, beta = PerformanceMetrics.alpha_beta(returns, benchmark_returns, risk_free)
        
        if beta == 0:
            return 0.0
        
        excess_return = np.mean(returns) * 252 - risk_free
        return excess_return / beta
    
    @staticmethod
    def ulcer_index(returns: np.ndarray) -> float:
        """
        Calculate Ulcer Index (measure of downside volatility)
        
        Args:
            returns: Strategy returns
            
        Returns:
            Ulcer index
        """
        if len(returns) < 2:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        
        return np.sqrt(np.mean(drawdown ** 2))
    
    @staticmethod
    def ulcer_performance_index(
        returns: np.ndarray,
        risk_free: float = 0.0,
    ) -> float:
        """
        Calculate Ulcer Performance Index (Martin ratio)
        
        Args:
            returns: Strategy returns
            risk_free: Risk-free rate
            
        Returns:
            Ulcer performance index
        """
        ulcer = PerformanceMetrics.ulcer_index(returns)
        
        if ulcer == 0:
            return np.mean(returns) * 252 * 100
        
        excess_return = np.mean(returns) * 252 - risk_free
        return excess_return / ulcer
    
    @staticmethod
    def omega_ratio(
        returns: np.ndarray,
        threshold: float = 0.0,
    ) -> float:
        """
        Calculate Omega ratio
        
        Args:
            returns: Strategy returns
            threshold: Return threshold
            
        Returns:
            Omega ratio
        """
        if len(returns) < 1:
            return 1.0
        
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        sum_losses = np.sum(losses)
        if sum_losses == 0:
            return np.sum(gains) * 100 if len(gains) > 0 else 1.0
        
        return np.sum(gains) / sum_losses
    
    @staticmethod
    def tail_ratio(returns: np.ndarray) -> float:
        """
        Calculate tail ratio (right tail / left tail)
        
        Args:
            returns: Strategy returns
            
        Returns:
            Tail ratio
        """
        if len(returns) < 20:
            return 1.0
        
        right_tail = np.percentile(returns, 95)
        left_tail = np.percentile(returns, 5)
        
        if left_tail == 0:
            return abs(right_tail) * 100
        
        return abs(right_tail / left_tail)
    
    @staticmethod
    def common_sense_ratio(returns: np.ndarray) -> float:
        """
        Calculate common sense ratio (tail ratio * gain/loss ratio)
        
        Args:
            returns: Strategy returns
            
        Returns:
            Common sense ratio
        """
        tail = PerformanceMetrics.tail_ratio(returns)
        
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(losses) == 0:
            return tail * 100
        
        win_rate = len(gains) / len(returns)
        loss_rate = len(losses) / len(returns)
        
        if loss_rate == 0:
            return tail * 100
        
        return tail * (win_rate / loss_rate)
    
    @staticmethod
    def skewness(returns: np.ndarray) -> float:
        """Calculate return skewness"""
        if len(returns) < 3:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        return np.mean(((returns - mean) / std) ** 3)
    
    @staticmethod
    def kurtosis(returns: np.ndarray) -> float:
        """Calculate return kurtosis (excess)"""
        if len(returns) < 4:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        return np.mean(((returns - mean) / std) ** 4) - 3
    
    @staticmethod
    def calculate_comprehensive(
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        risk_free: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics
        
        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns
            risk_free: Risk-free rate
            
        Returns:
            Dictionary of all metrics
        """
        pm = PerformanceMetrics
        
        metrics = {
            "ulcer_index": pm.ulcer_index(returns),
            "ulcer_performance": pm.ulcer_performance_index(returns, risk_free),
            "omega_ratio": pm.omega_ratio(returns),
            "tail_ratio": pm.tail_ratio(returns),
            "common_sense_ratio": pm.common_sense_ratio(returns),
            "skewness": pm.skewness(returns),
            "kurtosis": pm.kurtosis(returns),
        }
        
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            alpha, beta = pm.alpha_beta(returns, benchmark_returns, risk_free)
            metrics.update({
                "alpha": alpha,
                "beta": beta,
                "information_ratio": pm.information_ratio(returns, benchmark_returns),
                "treynor_ratio": pm.treynor_ratio(returns, benchmark_returns, risk_free),
            })
        
        return metrics


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Generate sample returns
    returns = np.random.normal(0.0005, 0.01, 252)
    benchmark = np.random.normal(0.0003, 0.012, 252)
    
    print("=== Performance Metrics Test ===")
    
    metrics = PerformanceMetrics.calculate_comprehensive(returns, benchmark)
    
    print("\nAll metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Individual tests
    print(f"\nAlpha: {metrics.get('alpha', 0):.2%}")
    print(f"Beta: {metrics.get('beta', 0):.2f}")
    print(f"Information Ratio: {metrics.get('information_ratio', 0):.2f}")
    print(f"Omega Ratio: {metrics.get('omega_ratio', 0):.2f}")

