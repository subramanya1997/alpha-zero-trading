"""
Differential Sharpe Ratio (DSR) for RL Trading
Based on the paper: "Reinforcement Learning for Optimized Trade Execution"

The DSR provides an immediate reward signal that corresponds to the
infinitesimal change in Sharpe ratio due to a single trade.
"""
import numpy as np
from typing import Tuple


class DifferentialSharpeRatio:
    """
    Calculates Differential Sharpe Ratio for incremental reward signal.
    
    The DSR is defined as:
        DSR = dSR/dt ≈ (A' - 0.5 * A * B') / (B - A²)^(3/2)
    
    Where:
        A = exponential moving average of returns
        B = exponential moving average of squared returns
        A', B' = updates to A and B
        
    This gives a reward signal that directly optimizes Sharpe ratio.
    """
    
    def __init__(self, eta: float = 0.01, warmup_steps: int = 20):
        """
        Initialize DSR calculator.
        
        Args:
            eta: Learning rate for exponential moving averages (0.01-0.1)
            warmup_steps: Number of steps before DSR is valid
        """
        self.eta = eta
        self.warmup_steps = warmup_steps
        
        # Running statistics
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns
        self.step_count = 0
    
    def reset(self):
        """Reset running statistics"""
        self.A = 0.0
        self.B = 0.0
        self.step_count = 0
    
    def step(self, return_t: float) -> Tuple[float, float]:
        """
        Calculate DSR for a single step.
        
        Args:
            return_t: Return at time t
            
        Returns:
            Tuple of (differential_sharpe_ratio, current_sharpe_estimate)
        """
        self.step_count += 1
        
        # Update statistics with exponential moving average
        delta_A = return_t - self.A
        delta_B = return_t ** 2 - self.B
        
        self.A = self.A + self.eta * delta_A
        self.B = self.B + self.eta * delta_B
        
        # Calculate variance estimate
        variance = self.B - self.A ** 2
        
        # Guard against numerical issues
        if variance < 1e-10 or self.step_count < self.warmup_steps:
            return 0.0, 0.0
        
        # Calculate DSR
        denominator = variance ** 1.5
        if denominator < 1e-10:
            return 0.0, 0.0
        
        dsr = (self.B * delta_A - 0.5 * self.A * delta_B) / denominator
        
        # Current Sharpe estimate (annualized)
        sharpe = self.A / np.sqrt(variance) * np.sqrt(252)
        
        return dsr, sharpe
    
    def get_sharpe(self) -> float:
        """Get current Sharpe estimate"""
        variance = self.B - self.A ** 2
        if variance < 1e-10:
            return 0.0
        return self.A / np.sqrt(variance) * np.sqrt(252)


class AdaptiveDSR:
    """
    Adaptive DSR with multi-timescale tracking and risk adjustment.
    """
    
    def __init__(
        self,
        etas: Tuple[float, ...] = (0.01, 0.05, 0.1),
        risk_aversion: float = 0.5,
        max_drawdown_penalty: float = 10.0,
    ):
        """
        Initialize Adaptive DSR.
        
        Args:
            etas: Multiple learning rates for different timescales
            risk_aversion: Weight on downside variance (0=neutral, 1=very risk averse)
            max_drawdown_penalty: Penalty scale for drawdown
        """
        self.dsrs = [DifferentialSharpeRatio(eta=eta) for eta in etas]
        self.risk_aversion = risk_aversion
        self.max_drawdown_penalty = max_drawdown_penalty
        
        # Track for Sortino-style downside calculation
        self.downside_ema = 0.0
        self.downside_eta = 0.05
        
        # Drawdown tracking
        self.peak_value = 1.0
        self.current_value = 1.0
    
    def reset(self):
        """Reset all trackers"""
        for dsr in self.dsrs:
            dsr.reset()
        self.downside_ema = 0.0
        self.peak_value = 1.0
        self.current_value = 1.0
    
    def step(self, return_t: float, portfolio_value: float = None) -> float:
        """
        Calculate adaptive DSR reward.
        
        Args:
            return_t: Return at time t
            portfolio_value: Optional portfolio value for drawdown tracking
            
        Returns:
            Combined reward signal
        """
        # Update all DSR trackers
        rewards = []
        sharpes = []
        for dsr in self.dsrs:
            r, s = dsr.step(return_t)
            rewards.append(r)
            sharpes.append(s)
        
        # Average DSR across timescales
        avg_dsr = np.mean(rewards)
        
        # Update downside tracking (for Sortino-style adjustment)
        if return_t < 0:
            self.downside_ema = (1 - self.downside_eta) * self.downside_ema + \
                                self.downside_eta * return_t ** 2
        
        # Downside penalty
        downside_penalty = 0.0
        if return_t < 0:
            downside_penalty = self.risk_aversion * abs(return_t) * 10
        
        # Drawdown penalty (if portfolio value provided)
        dd_penalty = 0.0
        if portfolio_value is not None:
            self.current_value = portfolio_value
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
            else:
                drawdown = (self.peak_value - portfolio_value) / self.peak_value
                if drawdown > 0.01:  # Only penalize significant drawdowns
                    dd_penalty = self.max_drawdown_penalty * drawdown ** 2
        
        # Combined reward
        reward = avg_dsr * 100  # Scale DSR
        reward -= downside_penalty
        reward -= dd_penalty
        
        # Small survival bonus
        reward += 0.05
        
        return float(np.clip(reward, -10, 10))
    
    def get_sharpes(self) -> dict:
        """Get Sharpe estimates at different timescales"""
        return {f"sharpe_eta_{dsr.eta}": dsr.get_sharpe() for dsr in self.dsrs}


class OnlineSharpeTracker:
    """
    Track Sharpe ratio online with proper variance estimation.
    Uses Welford's algorithm for numerical stability.
    """
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared deviations
        
        # For downside variance
        self.downside_M2 = 0.0
        self.downside_count = 0
    
    def reset(self):
        """Reset all statistics"""
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.downside_M2 = 0.0
        self.downside_count = 0
    
    def update(self, return_t: float):
        """Update with new return using Welford's algorithm"""
        self.n += 1
        delta = return_t - self.mean
        self.mean += delta / self.n
        delta2 = return_t - self.mean
        self.M2 += delta * delta2
        
        # Update downside variance
        if return_t < 0:
            self.downside_count += 1
            old_mean = self.mean - delta / self.n
            if self.downside_count > 1:
                self.downside_M2 += return_t ** 2
    
    @property
    def variance(self) -> float:
        """Get current variance estimate"""
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)
    
    @property
    def std(self) -> float:
        """Get current standard deviation"""
        return np.sqrt(self.variance)
    
    @property
    def sharpe(self) -> float:
        """Get current Sharpe ratio (annualized)"""
        if self.n < 10 or self.std < 1e-8:
            return 0.0
        return self.mean / self.std * np.sqrt(252)
    
    @property
    def sortino(self) -> float:
        """Get current Sortino ratio (annualized)"""
        if self.n < 10 or self.downside_count < 5:
            return 0.0
        downside_std = np.sqrt(self.downside_M2 / self.downside_count)
        if downside_std < 1e-8:
            return 0.0
        return self.mean / downside_std * np.sqrt(252)


if __name__ == "__main__":
    print("=== Testing Differential Sharpe Ratio ===\n")
    
    # Test with simulated returns
    np.random.seed(42)
    
    # Scenario 1: Random walk (no edge)
    print("Scenario 1: Random Walk (no edge)")
    dsr = DifferentialSharpeRatio(eta=0.05)
    returns = np.random.normal(0, 0.01, 100)
    
    for i, r in enumerate(returns):
        dsr_val, sharpe = dsr.step(r)
        if i % 20 == 0:
            print(f"  Step {i}: return={r:.4f}, DSR={dsr_val:.4f}, Sharpe={sharpe:.2f}")
    
    print(f"  Final Sharpe: {dsr.get_sharpe():.2f}\n")
    
    # Scenario 2: Positive edge
    print("Scenario 2: Positive Edge (mean=0.1%)")
    dsr = DifferentialSharpeRatio(eta=0.05)
    returns = np.random.normal(0.001, 0.01, 100)  # 0.1% daily expected return
    
    for i, r in enumerate(returns):
        dsr_val, sharpe = dsr.step(r)
        if i % 20 == 0:
            print(f"  Step {i}: return={r:.4f}, DSR={dsr_val:.4f}, Sharpe={sharpe:.2f}")
    
    print(f"  Final Sharpe: {dsr.get_sharpe():.2f}\n")
    
    # Scenario 3: Negative edge
    print("Scenario 3: Negative Edge (mean=-0.1%)")
    dsr = DifferentialSharpeRatio(eta=0.05)
    returns = np.random.normal(-0.001, 0.01, 100)
    
    for i, r in enumerate(returns):
        dsr_val, sharpe = dsr.step(r)
        if i % 20 == 0:
            print(f"  Step {i}: return={r:.4f}, DSR={dsr_val:.4f}, Sharpe={sharpe:.2f}")
    
    print(f"  Final Sharpe: {dsr.get_sharpe():.2f}\n")
    
    # Test Adaptive DSR
    print("=== Testing Adaptive DSR ===\n")
    
    adsr = AdaptiveDSR(risk_aversion=0.5)
    rewards = []
    value = 10000
    
    for i in range(100):
        r = np.random.normal(0.001, 0.015)  # Volatile but positive edge
        value *= (1 + r)
        reward = adsr.step(r, value)
        rewards.append(reward)
        
        if i % 20 == 0:
            sharpes = adsr.get_sharpes()
            print(f"  Step {i}: return={r:.4f}, reward={reward:.4f}, value=${value:,.0f}")
            print(f"           Sharpes: {sharpes}")
    
    print(f"\n  Mean reward: {np.mean(rewards):.4f}")
    print(f"  Reward std: {np.std(rewards):.4f}")
    
    # Test Online Tracker
    print("\n=== Testing Online Sharpe Tracker ===\n")
    
    tracker = OnlineSharpeTracker()
    returns = np.random.normal(0.0005, 0.01, 252)  # 1 year of daily returns
    
    for i, r in enumerate(returns):
        tracker.update(r)
        if i % 50 == 0 and i > 0:
            print(f"  Day {i}: Sharpe={tracker.sharpe:.2f}, Sortino={tracker.sortino:.2f}")
    
    print(f"\n  Final Sharpe: {tracker.sharpe:.2f}")
    print(f"  Final Sortino: {tracker.sortino:.2f}")
    print(f"  Mean return: {tracker.mean:.5f}")
    print(f"  Volatility: {tracker.std:.5f}")

