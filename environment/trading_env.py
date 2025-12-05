"""
Trading environment for reinforcement learning
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

try:
    from .portfolio_manager import PortfolioManager
except ImportError:
    from portfolio_manager import PortfolioManager


@dataclass
class EnvConfig:
    """Environment configuration"""
    initial_capital: float = 10_000.0
    max_leverage: float = 10.0
    min_leverage: float = 1.0  # Start with lower leverage
    max_drawdown: float = 0.05  # 5% max drawdown
    max_position_size: float = 0.50  # 50% max per position
    transaction_cost: float = 0.001  # 0.1% per trade
    maintenance_margin: float = 0.25
    lookback_window: int = 60
    symbols: List[str] = None
    
    # Curriculum/training settings
    warmup_steps: int = 50  # Steps before termination is enforced
    allow_early_termination: bool = True
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["SPY", "QQQ", "DIA", "IWM"]


class TradingEnv(gym.Env):
    """
    Trading environment with leverage and risk management
    
    State: Historical features + portfolio state
    Action: Position weights for each asset + leverage factor
    Reward: Risk-adjusted returns with drawdown penalties
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        returns: np.ndarray,
        dates: List,
        config: Optional[EnvConfig] = None,
    ):
        """
        Initialize trading environment
        
        Args:
            features: Array of shape (n_samples, lookback, n_features)
            prices: Array of shape (n_samples, n_symbols)
            returns: Array of shape (n_samples, n_symbols)
            dates: List of dates
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config or EnvConfig()
        self.features = features
        self.prices = prices
        self.returns = returns
        self.dates = dates
        
        self.n_samples = len(features)
        self.n_symbols = len(self.config.symbols)
        self.n_features = features.shape[2]
        
        # Portfolio manager
        self.portfolio = PortfolioManager(
            initial_capital=self.config.initial_capital,
            max_leverage=self.config.max_leverage,
            min_leverage=self.config.min_leverage,
            maintenance_margin=self.config.maintenance_margin,
            transaction_cost=self.config.transaction_cost,
            symbols=self.config.symbols,
        )
        
        # Action space: weights for each asset + leverage factor
        # Weights are in [0, 1], leverage factor in [0, 1] (scaled to min-max)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_symbols + 1,),  # 4 weights + 1 leverage
            dtype=np.float32,
        )
        
        # Observation space: flattened features + portfolio state
        portfolio_state_dim = 10  # cash_ratio, leverage, drawdown, etc.
        obs_dim = self.n_features + portfolio_state_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.lookback_window, obs_dim),
            dtype=np.float32,
        )
        
        # Internal state
        self.current_step = 0
        self.done = False
        self.truncated = False
        
        # Episode tracking
        self.episode_returns = []
        self.episode_drawdowns = []
        
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset portfolio
        self.portfolio.reset()
        
        # Random starting point (leave room for episode)
        max_start = self.n_samples - 252  # At least 1 year of trading
        if max_start <= 0:
            max_start = 1
        
        self.current_step = self.np_random.integers(0, max_start)
        self.done = False
        self.truncated = False
        
        # Reset tracking
        self.episode_returns = []
        self.episode_drawdowns = []
        
        # Initialize prices
        prices_dict = self._get_prices(self.current_step)
        self.portfolio.update_prices(prices_dict)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Array of [weight1, weight2, weight3, weight4, leverage_factor]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done or self.truncated:
            # Return terminal state
            return self._get_observation(), 0.0, self.done, self.truncated, self._get_info()
        
        # Parse action
        weights = action[:self.n_symbols]
        leverage_factor = action[self.n_symbols]
        
        # Normalize weights to sum to 1 (or less)
        weights = self._normalize_weights(weights)
        
        # Scale leverage factor to actual leverage
        target_leverage = self.config.min_leverage + leverage_factor * (
            self.config.max_leverage - self.config.min_leverage
        )
        
        # Get current prices
        prices_dict = self._get_prices(self.current_step)
        
        # Execute rebalance
        target_weights = {
            symbol: weights[i] for i, symbol in enumerate(self.config.symbols)
        }
        
        cost, trades = self.portfolio.execute_rebalance(
            target_weights, prices_dict, target_leverage
        )
        
        # Move to next step
        self.current_step += 1
        
        # Check if we've reached the end of data
        if self.current_step >= self.n_samples - 1:
            self.truncated = True
            obs = self._get_observation()
            reward = self._calculate_reward(cost)
            return obs, reward, self.done, self.truncated, self._get_info()
        
        # Update portfolio with new prices
        new_prices = self._get_prices(self.current_step)
        self.portfolio.step(new_prices)
        
        # Track episode metrics
        self.episode_returns.append(self.portfolio.daily_return)
        self.episode_drawdowns.append(self.portfolio.drawdown)
        
        # Check risk constraints
        if self._check_termination():
            self.done = True
        
        # Calculate reward
        reward = self._calculate_reward(cost)
        
        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, self.done, self.truncated, info
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to respect position limits"""
        # Clip individual weights to max position size
        weights = np.clip(weights, 0, self.config.max_position_size)
        
        # Ensure sum doesn't exceed 1
        total = np.sum(weights)
        if total > 1.0:
            weights = weights / total
        
        return weights
    
    def _get_prices(self, step: int) -> Dict[str, float]:
        """Get prices at given step"""
        return {
            symbol: self.prices[step, i]
            for i, symbol in enumerate(self.config.symbols)
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Market features (already lookback window)
        market_features = self.features[self.current_step]  # (lookback, n_features)
        
        # Portfolio state (broadcasted to lookback dimension)
        portfolio_state = self._get_portfolio_state()  # (portfolio_dim,)
        portfolio_state = np.tile(portfolio_state, (self.config.lookback_window, 1))
        
        # Combine
        obs = np.concatenate([market_features, portfolio_state], axis=1)
        
        return obs.astype(np.float32)
    
    def _get_portfolio_state(self) -> np.ndarray:
        """Get portfolio state as observation"""
        state = self.portfolio.get_state()
        
        # Normalize portfolio state features
        total_value = state.total_value
        
        return np.array([
            state.cash / total_value if total_value > 0 else 1.0,  # Cash ratio
            state.leverage / self.config.max_leverage,  # Normalized leverage
            state.drawdown / self.config.max_drawdown,  # Normalized drawdown
            state.daily_return,  # Daily return
            state.cumulative_return,  # Cumulative return
            state.margin_used / total_value if total_value > 0 else 0,  # Margin ratio
            state.unrealized_pnl / self.config.initial_capital,  # Normalized unrealized PnL
            state.realized_pnl / self.config.initial_capital,  # Normalized realized PnL
            min(self.portfolio.trade_count / 100, 1.0),  # Normalized trade count
            self.current_step / self.n_samples,  # Progress through data
        ], dtype=np.float32)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate due to risk constraints"""
        # During warmup period, don't terminate (let agent explore)
        episode_length = len(self.episode_returns)
        if not self.config.allow_early_termination or episode_length < self.config.warmup_steps:
            # Still check for catastrophic failure during warmup
            if self.portfolio.total_value <= self.config.initial_capital * 0.3:
                return True
            return False
        
        # Max drawdown exceeded
        if self.portfolio.drawdown <= -self.config.max_drawdown:
            return True
        
        # Margin call
        if self.portfolio.check_margin_call():
            return True
        
        # Portfolio value collapsed
        if self.portfolio.total_value <= self.config.initial_capital * 0.5:
            return True
        
        return False
    
    def _calculate_reward(self, transaction_cost: float) -> float:
        """
        Calculate reward based on risk-adjusted returns
        
        Reward components:
        1. Risk-adjusted daily return (scaled by volatility)
        2. Drawdown penalty (progressive)
        3. Transaction cost penalty
        4. Rolling Sharpe bonus (main learning signal)
        5. Survival bonus
        """
        daily_return = self.portfolio.daily_return
        drawdown = self.portfolio.drawdown
        episode_len = len(self.episode_returns)
        
        reward = 0.0
        
        # 1. Survival bonus (encourage longer episodes)
        reward += 0.1
        
        # 2. Risk-adjusted return
        if episode_len >= 5:
            recent_returns = np.array(self.episode_returns[-5:])
            vol = np.std(recent_returns) + 1e-6
            risk_adjusted = daily_return / vol
            reward += np.clip(risk_adjusted * 5, -2, 2)  # Scaled and clipped
        else:
            reward += daily_return * 50  # Early stage: raw return signal
        
        # 3. Rolling Sharpe bonus (main objective)
        if episode_len >= 20:
            returns = np.array(self.episode_returns[-20:])
            sharpe = self._calculate_rolling_sharpe(returns)
            # Sharpe bonus: positive sharpe is rewarded, negative is penalized
            reward += sharpe * 2.0
        
        # 4. Drawdown penalty (progressive, not binary)
        if drawdown < 0:
            dd_ratio = abs(drawdown) / self.config.max_drawdown
            if dd_ratio < 0.5:
                # Mild penalty for small drawdowns
                drawdown_penalty = dd_ratio * 2
            elif dd_ratio < 0.8:
                # Moderate penalty
                drawdown_penalty = dd_ratio ** 2 * 5
            else:
                # Severe penalty near max drawdown
                drawdown_penalty = dd_ratio ** 3 * 10
            reward -= drawdown_penalty
        
        # 5. Transaction cost penalty (scaled down)
        cost_penalty = (transaction_cost / self.config.initial_capital) * 20
        reward -= cost_penalty
        
        # 6. Terminal rewards/penalties
        if self.done:
            if drawdown <= -self.config.max_drawdown:
                # Graduated penalty based on episode length
                reward -= max(10, 50 - episode_len * 0.5)
            elif self.portfolio.check_margin_call():
                reward -= 30.0
        
        # 7. End of episode bonus (survived until data end)
        if self.truncated and not self.done:
            final_return = self.portfolio.cumulative_return
            if final_return > 0:
                reward += 20 * final_return  # Bonus for positive returns
            # Bonus based on final Sharpe
            if episode_len >= 20:
                full_sharpe = self._calculate_rolling_sharpe(np.array(self.episode_returns))
                if full_sharpe > 0:
                    reward += full_sharpe * 10
        
        return float(np.clip(reward, -100, 100))
    
    def _calculate_rolling_sharpe(self, returns: np.ndarray) -> float:
        """Calculate rolling Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns) * 252  # Annualized
        std_ret = np.std(returns) * np.sqrt(252)
        if std_ret == 0:
            return 0.0
        return mean_ret / std_ret
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info"""
        metrics = self.portfolio.get_metrics()
        state = self.portfolio.get_state()
        
        return {
            "date": self.dates[self.current_step] if self.current_step < len(self.dates) else None,
            "step": self.current_step,
            "total_value": state.total_value,
            "cash": state.cash,
            "leverage": state.leverage,
            "drawdown": state.drawdown,
            "daily_return": state.daily_return,
            "cumulative_return": state.cumulative_return,
            "trade_count": self.portfolio.trade_count,
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "total_costs": metrics["total_costs"],
            "positions": self.portfolio.get_position_weights(),
        }
    
    def render(self, mode: str = "human"):
        """Render the environment"""
        info = self._get_info()
        print(f"\n=== Step {info['step']} | {info['date']} ===")
        print(f"Total Value: ${info['total_value']:,.2f}")
        print(f"Leverage: {info['leverage']:.2f}x")
        print(f"Daily Return: {info['daily_return']:.2%}")
        print(f"Cumulative Return: {info['cumulative_return']:.2%}")
        print(f"Drawdown: {info['drawdown']:.2%}")
        print(f"Positions: {info['positions']}")


def create_env(
    features: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    dates: List,
    config: Optional[EnvConfig] = None,
) -> TradingEnv:
    """Factory function to create trading environment"""
    return TradingEnv(features, prices, returns, dates, config)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
    
    from data.data_loader import DataLoader
    from data.feature_engineering import FeatureEngineer
    from data.preprocessing import DataPreprocessor
    
    # Load and process data
    print("Loading data...")
    loader = DataLoader()
    data = loader.download_all()
    
    print("Processing features...")
    engineer = FeatureEngineer()
    processed = engineer.process_all_symbols(data)
    processed = engineer.add_cross_asset_features(processed)
    
    print("Preprocessing...")
    preprocessor = DataPreprocessor()
    split, sequences = preprocessor.prepare_training_data(processed)
    
    # Create environment with training data
    features, prices, returns, dates = sequences["train"]
    
    print(f"\nCreating environment...")
    print(f"  Features shape: {features.shape}")
    print(f"  Prices shape: {prices.shape}")
    
    config = EnvConfig(
        initial_capital=10000,
        max_leverage=10,
        min_leverage=5,
        max_drawdown=0.05,
    )
    
    env = TradingEnv(features, prices, returns, dates, config)
    
    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Test environment
    print("\n--- Running test episode ---")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    total_reward = 0
    for step in range(100):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            env.render()
        
        if done or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"Reason: {'Done (risk breach)' if done else 'Truncated (end of data)'}")
            break
    
    print(f"\n--- Episode Summary ---")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final info: {info}")

