"""
Portfolio manager for tracking positions, P&L, and margin
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Position:
    """Single position in an asset"""
    symbol: str
    shares: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost
    
    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis


@dataclass 
class PortfolioState:
    """Current state of the portfolio"""
    cash: float
    positions: Dict[str, Position]
    total_value: float
    peak_value: float
    leverage: float
    margin_used: float
    buying_power: float
    realized_pnl: float
    unrealized_pnl: float
    drawdown: float
    daily_return: float
    cumulative_return: float


class PortfolioManager:
    """Manages portfolio state with leverage and margin"""
    
    def __init__(
        self,
        initial_capital: float = 10_000.0,
        max_leverage: float = 10.0,
        min_leverage: float = 5.0,
        maintenance_margin: float = 0.25,
        transaction_cost: float = 0.001,
        symbols: Optional[List[str]] = None,
    ):
        """
        Initialize portfolio manager
        
        Args:
            initial_capital: Starting capital
            max_leverage: Maximum leverage allowed
            min_leverage: Minimum leverage to use
            maintenance_margin: Maintenance margin requirement
            transaction_cost: Cost per trade as fraction
            symbols: List of tradeable symbols
        """
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.maintenance_margin = maintenance_margin
        self.transaction_cost = transaction_cost
        self.symbols = symbols or ["SPY", "QQQ", "DIA", "IWM"]
        
        self.reset()
        
    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {
            symbol: Position(symbol=symbol) for symbol in self.symbols
        }
        self.peak_value = self.initial_capital
        self.realized_pnl = 0.0
        self.prev_total_value = self.initial_capital
        self.trade_count = 0
        self.total_costs = 0.0
        
        # Track history
        self.value_history: List[float] = [self.initial_capital]
        self.return_history: List[float] = [0.0]
        
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)"""
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value
    
    @property
    def gross_exposure(self) -> float:
        """Total absolute exposure (sum of position values)"""
        return sum(abs(pos.market_value) for pos in self.positions.values())
    
    @property
    def current_leverage(self) -> float:
        """Current leverage ratio"""
        equity = self.total_value
        if equity <= 0:
            return float("inf")
        return self.gross_exposure / equity
    
    @property
    def margin_used(self) -> float:
        """Margin currently in use"""
        return self.gross_exposure * self.maintenance_margin
    
    @property
    def buying_power(self) -> float:
        """Available buying power with leverage"""
        equity = self.total_value
        max_exposure = equity * self.max_leverage
        return max(0, max_exposure - self.gross_exposure)
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def drawdown(self) -> float:
        """Current drawdown from peak"""
        if self.peak_value <= 0:
            return 0.0
        return (self.total_value - self.peak_value) / self.peak_value
    
    @property
    def daily_return(self) -> float:
        """Return since last update"""
        if self.prev_total_value <= 0:
            return 0.0
        return (self.total_value - self.prev_total_value) / self.prev_total_value
    
    @property
    def cumulative_return(self) -> float:
        """Total return since inception"""
        return (self.total_value - self.initial_capital) / self.initial_capital
    
    def get_position_weights(self) -> Dict[str, float]:
        """Get current position weights"""
        total = self.total_value
        if total <= 0:
            return {symbol: 0.0 for symbol in self.symbols}
        return {
            symbol: pos.market_value / total 
            for symbol, pos in self.positions.items()
        }
    
    def execute_rebalance(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        target_leverage: float = 1.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Rebalance portfolio to target weights
        
        Args:
            target_weights: Target weight for each symbol (can sum > 1 with leverage)
            prices: Current prices for each symbol
            target_leverage: Target leverage multiplier
            
        Returns:
            Tuple of (total_cost, trades_executed)
        """
        # Update prices first
        self.update_prices(prices)
        
        # Clamp leverage
        target_leverage = np.clip(target_leverage, self.min_leverage, self.max_leverage)
        
        # Calculate target values
        equity = self.total_value
        target_exposure = equity * target_leverage
        
        trades = {}
        total_cost = 0.0
        
        for symbol in self.symbols:
            if symbol not in target_weights or symbol not in prices:
                continue
            
            weight = target_weights.get(symbol, 0.0)
            price = prices[symbol]
            
            # Target position value
            target_value = target_exposure * weight
            current_value = self.positions[symbol].market_value
            
            # Trade amount
            trade_value = target_value - current_value
            
            if abs(trade_value) < 1.0:  # Skip tiny trades
                trades[symbol] = 0.0
                continue
            
            # Calculate shares to trade
            shares_to_trade = trade_value / price
            
            # Transaction cost
            cost = abs(trade_value) * self.transaction_cost
            total_cost += cost
            self.total_costs += cost
            
            # Execute trade
            self._execute_trade(symbol, shares_to_trade, price)
            trades[symbol] = shares_to_trade
            
            if shares_to_trade != 0:
                self.trade_count += 1
        
        # Deduct costs from cash
        self.cash -= total_cost
        
        return total_cost, trades
    
    def _execute_trade(self, symbol: str, shares: float, price: float):
        """Execute a single trade"""
        position = self.positions[symbol]
        
        if shares > 0:  # Buying
            # Update average cost
            total_shares = position.shares + shares
            if total_shares > 0:
                total_cost = position.cost_basis + (shares * price)
                position.avg_cost = total_cost / total_shares
            position.shares = total_shares
            self.cash -= shares * price
            
        else:  # Selling
            shares_to_sell = abs(shares)
            if position.shares > 0:
                # Realize P&L
                pnl_per_share = price - position.avg_cost
                realized = min(shares_to_sell, position.shares) * pnl_per_share
                self.realized_pnl += realized
            
            position.shares -= shares_to_sell
            self.cash += shares_to_sell * price
            
            # Reset avg cost if position closed
            if position.shares <= 0:
                position.shares = 0
                position.avg_cost = 0
        
        position.current_price = price
    
    def step(self, prices: Dict[str, float]):
        """
        Update portfolio state at end of day
        
        Args:
            prices: End of day prices
        """
        self.update_prices(prices)
        
        # Update peak value
        current_value = self.total_value
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Track history
        daily_return = (current_value - self.prev_total_value) / self.prev_total_value if self.prev_total_value > 0 else 0
        self.value_history.append(current_value)
        self.return_history.append(daily_return)
        
        self.prev_total_value = current_value
    
    def get_state(self) -> PortfolioState:
        """Get current portfolio state"""
        return PortfolioState(
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=self.total_value,
            peak_value=self.peak_value,
            leverage=self.current_leverage,
            margin_used=self.margin_used,
            buying_power=self.buying_power,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            drawdown=self.drawdown,
            daily_return=self.daily_return,
            cumulative_return=self.cumulative_return,
        )
    
    def check_margin_call(self) -> bool:
        """Check if margin call is triggered"""
        equity = self.total_value
        if equity <= 0:
            return True
        
        # Margin call if equity < maintenance margin requirement
        required_margin = self.gross_exposure * self.maintenance_margin
        return equity < required_margin
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        returns = np.array(self.return_history[1:])  # Skip initial 0
        
        if len(returns) < 2:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": self.drawdown,
                "volatility": 0.0,
                "total_return": self.cumulative_return,
                "trade_count": self.trade_count,
                "total_costs": self.total_costs,
            }
        
        # Sharpe ratio (annualized)
        mean_return = np.mean(returns) * 252
        std_return = np.std(returns) * np.sqrt(252)
        sharpe = mean_return / std_return if std_return > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        sortino = mean_return / downside_std if downside_std > 0 else 0.0
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - peak) / peak
        max_dd = np.min(drawdowns)
        
        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "volatility": std_return,
            "total_return": self.cumulative_return,
            "trade_count": self.trade_count,
            "total_costs": self.total_costs,
        }


if __name__ == "__main__":
    # Test portfolio manager
    pm = PortfolioManager(initial_capital=10000, max_leverage=10)
    
    # Simulate some trading
    prices = {"SPY": 450, "QQQ": 380, "DIA": 350, "IWM": 200}
    
    print("Initial state:")
    print(f"  Total value: ${pm.total_value:,.2f}")
    print(f"  Cash: ${pm.cash:,.2f}")
    print(f"  Leverage: {pm.current_leverage:.2f}x")
    
    # Rebalance to 5x leverage
    target_weights = {"SPY": 0.3, "QQQ": 0.3, "DIA": 0.2, "IWM": 0.2}
    cost, trades = pm.execute_rebalance(target_weights, prices, target_leverage=5.0)
    
    print(f"\nAfter 5x leverage rebalance:")
    print(f"  Total value: ${pm.total_value:,.2f}")
    print(f"  Cash: ${pm.cash:,.2f}")
    print(f"  Leverage: {pm.current_leverage:.2f}x")
    print(f"  Trades: {trades}")
    print(f"  Cost: ${cost:.2f}")
    
    # Simulate price move
    new_prices = {"SPY": 455, "QQQ": 385, "DIA": 352, "IWM": 202}
    pm.step(new_prices)
    
    print(f"\nAfter price move:")
    print(f"  Total value: ${pm.total_value:,.2f}")
    print(f"  Daily return: {pm.daily_return:.2%}")
    print(f"  Cumulative return: {pm.cumulative_return:.2%}")
    print(f"  Drawdown: {pm.drawdown:.2%}")
    
    # Get metrics
    metrics = pm.get_metrics()
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

