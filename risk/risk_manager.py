"""
Risk management module for position sizing and constraints
"""
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_drawdown: float = 0.05  # 5% max drawdown
    max_position_size: float = 0.50  # 50% max per position
    max_leverage: float = 10.0
    min_leverage: float = 5.0
    max_daily_var: float = 0.02  # 2% daily VaR at 99%
    volatility_target: float = 0.20  # 20% annual volatility target
    min_cash_buffer: float = 0.05  # 5% cash buffer


class RiskManager:
    """
    Risk management for leveraged trading
    
    Enforces position limits, volatility targeting, and drawdown controls
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager
        
        Args:
            limits: Risk limit configuration
        """
        self.limits = limits or RiskLimits()
        
        # Track historical data for risk calculations
        self.returns_history = []
        self.volatility_history = []
        self.drawdown_history = []
        
    def reset(self):
        """Reset historical tracking"""
        self.returns_history = []
        self.volatility_history = []
        self.drawdown_history = []
    
    def calculate_volatility(
        self,
        returns: np.ndarray,
        window: int = 20,
    ) -> float:
        """
        Calculate rolling volatility
        
        Args:
            returns: Array of daily returns
            window: Rolling window size
            
        Returns:
            Annualized volatility
        """
        if len(returns) < window:
            return 0.2  # Default 20% vol
        
        recent_returns = returns[-window:]
        daily_vol = np.std(recent_returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        return annual_vol
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.99,
        window: int = 60,
    ) -> float:
        """
        Calculate Value at Risk (parametric)
        
        Args:
            returns: Array of daily returns
            confidence: Confidence level (e.g., 0.99 for 99%)
            window: Rolling window size
            
        Returns:
            VaR as positive value (maximum expected loss)
        """
        if len(returns) < window:
            return 0.02  # Default 2%
        
        recent_returns = returns[-window:]
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        # Z-score for confidence level
        z_score = abs(np.percentile(
            np.random.standard_normal(10000), 
            (1 - confidence) * 100
        ))
        
        var = -(mean_return - z_score * std_return)
        return max(var, 0)
    
    def adjust_weights_for_volatility(
        self,
        weights: Dict[str, float],
        asset_volatilities: Dict[str, float],
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Adjust position weights to target portfolio volatility
        
        Args:
            weights: Target weights per asset
            asset_volatilities: Volatility per asset
            correlation_matrix: Correlation matrix between assets
            
        Returns:
            Adjusted weights
        """
        symbols = list(weights.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return weights
        
        # Convert to arrays
        w = np.array([weights[s] for s in symbols])
        vols = np.array([asset_volatilities.get(s, 0.2) for s in symbols])
        
        # Use identity correlation if not provided
        if correlation_matrix is None:
            corr = np.eye(n_assets)
        else:
            corr = correlation_matrix
        
        # Calculate portfolio volatility
        cov = np.outer(vols, vols) * corr
        port_vol = np.sqrt(w @ cov @ w)
        
        if port_vol > 0:
            # Scale weights to hit volatility target
            scale = self.limits.volatility_target / port_vol
            scale = min(scale, 1.0)  # Don't scale up
            w = w * scale
        
        return {s: float(w[i]) for i, s in enumerate(symbols)}
    
    def enforce_position_limits(
        self,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Enforce maximum position size limits
        
        Args:
            weights: Target weights per asset
            
        Returns:
            Weights with limits enforced
        """
        adjusted = {}
        
        for symbol, weight in weights.items():
            # Clip to max position size
            adjusted[symbol] = min(weight, self.limits.max_position_size)
        
        return adjusted
    
    def calculate_safe_leverage(
        self,
        current_volatility: float,
        current_drawdown: float,
        target_leverage: float,
    ) -> float:
        """
        Calculate safe leverage based on current market conditions
        
        Args:
            current_volatility: Current annualized volatility
            current_drawdown: Current drawdown (negative value)
            target_leverage: Requested leverage
            
        Returns:
            Safe leverage level
        """
        # Base leverage within limits
        leverage = np.clip(
            target_leverage,
            self.limits.min_leverage,
            self.limits.max_leverage
        )
        
        # Reduce leverage in high volatility
        vol_ratio = current_volatility / self.limits.volatility_target
        if vol_ratio > 1.5:
            leverage *= (1.5 / vol_ratio)
        
        # Reduce leverage when approaching max drawdown
        if current_drawdown < 0:
            dd_ratio = abs(current_drawdown) / self.limits.max_drawdown
            if dd_ratio > 0.5:
                leverage *= (1 - (dd_ratio - 0.5))
        
        # Final clamp
        return np.clip(leverage, self.limits.min_leverage, self.limits.max_leverage)
    
    def check_risk_breach(
        self,
        drawdown: float,
        daily_var: float,
        leverage: float,
    ) -> Tuple[bool, str]:
        """
        Check if any risk limits are breached
        
        Args:
            drawdown: Current drawdown (negative value)
            daily_var: Current daily VaR
            leverage: Current leverage
            
        Returns:
            Tuple of (is_breached, reason)
        """
        if drawdown <= -self.limits.max_drawdown:
            return True, f"Max drawdown breached: {drawdown:.2%}"
        
        if daily_var > self.limits.max_daily_var:
            return True, f"Daily VaR exceeded: {daily_var:.2%}"
        
        if leverage > self.limits.max_leverage * 1.1:  # 10% buffer
            return True, f"Leverage too high: {leverage:.2f}x"
        
        return False, ""
    
    def apply_risk_adjustments(
        self,
        action: np.ndarray,
        portfolio_returns: np.ndarray,
        current_drawdown: float,
        asset_volatilities: Dict[str, float],
        symbols: list,
    ) -> Tuple[np.ndarray, float]:
        """
        Apply all risk adjustments to proposed action
        
        Args:
            action: Raw action [w1, w2, w3, w4, leverage_factor]
            portfolio_returns: Historical portfolio returns
            current_drawdown: Current portfolio drawdown
            asset_volatilities: Volatility per asset
            symbols: List of asset symbols
            
        Returns:
            Tuple of (adjusted_action, safe_leverage)
        """
        # Extract weights and leverage
        weights = action[:-1]
        leverage_factor = action[-1]
        
        # Convert to dict
        weight_dict = {s: weights[i] for i, s in enumerate(symbols)}
        
        # 1. Enforce position limits
        weight_dict = self.enforce_position_limits(weight_dict)
        
        # 2. Adjust for volatility targeting
        weight_dict = self.adjust_weights_for_volatility(
            weight_dict, asset_volatilities
        )
        
        # 3. Calculate safe leverage
        current_vol = self.calculate_volatility(portfolio_returns)
        target_leverage = 5.0 + leverage_factor * 5.0  # 5-10x range
        safe_leverage = self.calculate_safe_leverage(
            current_vol, current_drawdown, target_leverage
        )
        
        # Convert back to array
        adjusted_weights = np.array([weight_dict[s] for s in symbols])
        
        # Ensure weights sum to <= 1
        total_weight = np.sum(adjusted_weights)
        if total_weight > 1.0:
            adjusted_weights = adjusted_weights / total_weight
        
        # Recombine
        adjusted_action = np.concatenate([
            adjusted_weights,
            [safe_leverage / 10.0]  # Normalize back to [0, 1]
        ])
        
        return adjusted_action, safe_leverage
    
    def get_risk_metrics(
        self,
        portfolio_returns: np.ndarray,
        current_drawdown: float,
        current_leverage: float,
    ) -> Dict[str, float]:
        """
        Get current risk metrics
        
        Args:
            portfolio_returns: Historical returns
            current_drawdown: Current drawdown
            current_leverage: Current leverage
            
        Returns:
            Dictionary of risk metrics
        """
        volatility = self.calculate_volatility(portfolio_returns)
        var = self.calculate_var(portfolio_returns)
        
        return {
            "volatility": volatility,
            "var_99": var,
            "drawdown": current_drawdown,
            "leverage": current_leverage,
            "vol_ratio": volatility / self.limits.volatility_target,
            "dd_ratio": abs(current_drawdown) / self.limits.max_drawdown if current_drawdown < 0 else 0,
        }


if __name__ == "__main__":
    # Test risk manager
    rm = RiskManager()
    
    # Simulate some returns
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 100)
    
    print("=== Risk Manager Test ===")
    
    # Test volatility calculation
    vol = rm.calculate_volatility(returns)
    print(f"\nVolatility: {vol:.2%}")
    
    # Test VaR calculation
    var = rm.calculate_var(returns)
    print(f"VaR (99%): {var:.2%}")
    
    # Test position limit enforcement
    weights = {"SPY": 0.6, "QQQ": 0.4, "DIA": 0.3, "IWM": 0.2}
    adjusted = rm.enforce_position_limits(weights)
    print(f"\nOriginal weights: {weights}")
    print(f"Adjusted weights: {adjusted}")
    
    # Test volatility adjustment
    asset_vols = {"SPY": 0.15, "QQQ": 0.25, "DIA": 0.14, "IWM": 0.22}
    vol_adjusted = rm.adjust_weights_for_volatility(adjusted, asset_vols)
    print(f"Vol-adjusted weights: {vol_adjusted}")
    
    # Test safe leverage
    leverage = rm.calculate_safe_leverage(0.25, -0.03, 8.0)
    print(f"\nSafe leverage (high vol, some DD): {leverage:.2f}x")
    
    leverage = rm.calculate_safe_leverage(0.15, 0.0, 8.0)
    print(f"Safe leverage (low vol, no DD): {leverage:.2f}x")
    
    # Test risk breach check
    breached, reason = rm.check_risk_breach(-0.06, 0.01, 8.0)
    print(f"\nRisk breach (6% DD): {breached}, {reason}")
    
    breached, reason = rm.check_risk_breach(-0.03, 0.01, 8.0)
    print(f"Risk breach (3% DD): {breached}, {reason}")
    
    # Test full adjustment
    action = np.array([0.3, 0.3, 0.2, 0.2, 0.6])  # weights + leverage factor
    symbols = ["SPY", "QQQ", "DIA", "IWM"]
    
    adjusted_action, safe_leverage = rm.apply_risk_adjustments(
        action, returns, -0.02, asset_vols, symbols
    )
    print(f"\nOriginal action: {action}")
    print(f"Adjusted action: {adjusted_action}")
    print(f"Safe leverage: {safe_leverage:.2f}x")
    
    # Get risk metrics
    metrics = rm.get_risk_metrics(returns, -0.02, safe_leverage)
    print(f"\nRisk metrics: {metrics}")

