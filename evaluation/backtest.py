"""
Backtesting module for evaluating trained models
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.trading_env import TradingEnv, EnvConfig
from models.alphazero_agent import AlphaZeroAgent, AgentConfig


@dataclass
class BacktestResult:
    """Container for backtest results"""
    # Performance metrics
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    
    # Trade statistics
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    
    # Risk metrics
    avg_leverage: float = 0.0
    max_leverage: float = 0.0
    avg_drawdown: float = 0.0
    
    # Time series data
    dates: List = field(default_factory=list)
    portfolio_values: List = field(default_factory=list)
    daily_returns: List = field(default_factory=list)
    drawdowns: List = field(default_factory=list)
    positions: List = field(default_factory=list)
    leverages: List = field(default_factory=list)


class Backtester:
    """
    Backtester for evaluating trading models
    """
    
    def __init__(
        self,
        agent: AlphaZeroAgent,
        device: str = "cpu",
    ):
        """
        Initialize backtester
        
        Args:
            agent: Trained agent to evaluate
            device: Device to run on
        """
        self.agent = agent
        self.device = device
        self.agent.to(device)
        self.agent.eval()
    
    def run(
        self,
        env: TradingEnv,
        deterministic: bool = True,
    ) -> BacktestResult:
        """
        Run backtest on environment
        
        Args:
            env: Trading environment with test data
            deterministic: Whether to use deterministic actions
            
        Returns:
            BacktestResult with all metrics
        """
        result = BacktestResult()
        
        # Reset environment
        obs, info = env.reset()
        obs, info = env.reset()  # Double reset to start from beginning
        
        # Override random start
        env.current_step = 0
        env.portfolio.reset()
        
        done = False
        truncated = False
        
        # Track data
        values = [env.portfolio.total_value]
        returns = []
        drawdowns = [0.0]
        positions_hist = []
        leverages = [0.0]
        dates = [env.dates[0] if env.dates else None]
        
        while not done and not truncated:
            # Get action from agent
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, _, _, _ = self.agent(obs_tensor, deterministic=deterministic)
            
            action_np = action.cpu().numpy()[0]
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action_np)
            
            # Track data
            values.append(info["total_value"])
            returns.append(info["daily_return"])
            drawdowns.append(info["drawdown"])
            positions_hist.append(info["positions"])
            leverages.append(info["leverage"])
            dates.append(info["date"])
        
        # Calculate metrics
        result.dates = dates
        result.portfolio_values = values
        result.daily_returns = returns
        result.drawdowns = drawdowns
        result.positions = positions_hist
        result.leverages = leverages
        
        # Performance metrics
        returns_arr = np.array(returns)
        
        result.total_return = (values[-1] / values[0]) - 1
        result.sharpe_ratio = self._calculate_sharpe(returns_arr)
        result.sortino_ratio = self._calculate_sortino(returns_arr)
        result.max_drawdown = min(drawdowns)
        result.volatility = np.std(returns_arr) * np.sqrt(252)
        
        # CAGR
        n_years = len(returns_arr) / 252
        if n_years > 0 and result.total_return > -1:
            result.cagr = (1 + result.total_return) ** (1 / n_years) - 1
        
        # Calmar ratio
        if result.max_drawdown != 0:
            result.calmar_ratio = result.cagr / abs(result.max_drawdown)
        
        # Trade statistics
        result.n_trades = env.portfolio.trade_count
        result.win_rate = np.sum(returns_arr > 0) / len(returns_arr) if len(returns_arr) > 0 else 0
        
        gains = returns_arr[returns_arr > 0]
        losses = returns_arr[returns_arr < 0]
        if len(losses) > 0 and np.sum(np.abs(losses)) > 0:
            result.profit_factor = np.sum(gains) / np.sum(np.abs(losses))
        
        result.avg_trade_return = np.mean(returns_arr) if len(returns_arr) > 0 else 0
        
        # Leverage stats
        leverages_arr = np.array(leverages)
        result.avg_leverage = np.mean(leverages_arr)
        result.max_leverage = np.max(leverages_arr)
        result.avg_drawdown = np.mean(drawdowns)
        
        return result
    
    def _calculate_sharpe(self, returns: np.ndarray, rf: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        excess = returns - rf / 252
        return np.mean(excess) / np.std(returns) * np.sqrt(252)
    
    def _calculate_sortino(self, returns: np.ndarray, rf: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0
        excess = returns - rf / 252
        downside = returns[returns < 0]
        if len(downside) == 0 or np.std(downside) == 0:
            return np.mean(excess) * np.sqrt(252) * 100
        return np.mean(excess) / np.std(downside) * np.sqrt(252)
    
    def run_baselines(
        self,
        env: TradingEnv,
    ) -> Dict[str, BacktestResult]:
        """
        Run baseline strategies for comparison
        
        Args:
            env: Trading environment
            
        Returns:
            Dictionary of strategy name to result
        """
        results = {}
        
        # Buy and hold SPY
        results["buy_hold_spy"] = self._run_static_strategy(
            env, {"SPY": 1.0, "QQQ": 0.0, "DIA": 0.0, "IWM": 0.0}, leverage=1.0
        )
        
        # Equal weight
        results["equal_weight"] = self._run_static_strategy(
            env, {"SPY": 0.25, "QQQ": 0.25, "DIA": 0.25, "IWM": 0.25}, leverage=1.0
        )
        
        # Equal weight with 5x leverage
        results["equal_weight_5x"] = self._run_static_strategy(
            env, {"SPY": 0.25, "QQQ": 0.25, "DIA": 0.25, "IWM": 0.25}, leverage=5.0
        )
        
        return results
    
    def _run_static_strategy(
        self,
        env: TradingEnv,
        weights: Dict[str, float],
        leverage: float = 1.0,
    ) -> BacktestResult:
        """Run a static allocation strategy"""
        result = BacktestResult()
        
        # Create action array
        symbols = env.config.symbols
        action = np.zeros(len(symbols) + 1)
        for i, symbol in enumerate(symbols):
            action[i] = weights.get(symbol, 0.0)
        action[-1] = (leverage - 5) / 5  # Normalize leverage factor
        
        # Reset environment
        obs, info = env.reset()
        env.current_step = 0
        env.portfolio.reset()
        
        values = [env.portfolio.total_value]
        returns = []
        drawdowns = [0.0]
        
        done = False
        truncated = False
        
        while not done and not truncated:
            obs, reward, done, truncated, info = env.step(action)
            values.append(info["total_value"])
            returns.append(info["daily_return"])
            drawdowns.append(info["drawdown"])
        
        # Calculate metrics
        returns_arr = np.array(returns)
        
        result.total_return = (values[-1] / values[0]) - 1
        result.sharpe_ratio = self._calculate_sharpe(returns_arr)
        result.sortino_ratio = self._calculate_sortino(returns_arr)
        result.max_drawdown = min(drawdowns)
        result.volatility = np.std(returns_arr) * np.sqrt(252)
        result.portfolio_values = values
        result.daily_returns = returns
        
        n_years = len(returns_arr) / 252
        if n_years > 0 and result.total_return > -1:
            result.cagr = (1 + result.total_return) ** (1 / n_years) - 1
        
        return result
    
    def compare_results(
        self,
        agent_result: BacktestResult,
        baseline_results: Dict[str, BacktestResult],
    ) -> pd.DataFrame:
        """
        Compare agent results with baselines
        
        Args:
            agent_result: Agent backtest result
            baseline_results: Dictionary of baseline results
            
        Returns:
            Comparison DataFrame
        """
        data = {
            "Agent": {
                "Total Return": f"{agent_result.total_return:.2%}",
                "CAGR": f"{agent_result.cagr:.2%}",
                "Sharpe Ratio": f"{agent_result.sharpe_ratio:.2f}",
                "Sortino Ratio": f"{agent_result.sortino_ratio:.2f}",
                "Max Drawdown": f"{agent_result.max_drawdown:.2%}",
                "Volatility": f"{agent_result.volatility:.2%}",
                "Win Rate": f"{agent_result.win_rate:.2%}",
            }
        }
        
        for name, result in baseline_results.items():
            data[name] = {
                "Total Return": f"{result.total_return:.2%}",
                "CAGR": f"{result.cagr:.2%}",
                "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
                "Sortino Ratio": f"{result.sortino_ratio:.2f}",
                "Max Drawdown": f"{result.max_drawdown:.2%}",
                "Volatility": f"{result.volatility:.2%}",
                "Win Rate": "-",
            }
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    from data.data_loader import DataLoader
    from data.feature_engineering import FeatureEngineer
    from data.preprocessing import DataPreprocessor
    
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
    
    # Create test environment
    env_config = EnvConfig(
        initial_capital=10000,
        max_leverage=10,
        min_leverage=5,
        max_drawdown=0.05,
    )
    
    test_env = TradingEnv(*sequences["test"], config=env_config)
    
    # Create agent (random for testing)
    agent_config = AgentConfig(
        input_dim=258,
        seq_len=60,
        n_assets=4,
    )
    agent = AlphaZeroAgent(agent_config)
    
    # Run backtest
    backtester = Backtester(agent)
    
    print("\n--- Running Agent Backtest ---")
    result = backtester.run(test_env)
    
    print(f"\nAgent Results:")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  CAGR: {result.cagr:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Win Rate: {result.win_rate:.2%}")
    print(f"  Trades: {result.n_trades}")
    
    # Run baselines
    print("\n--- Running Baselines ---")
    baselines = backtester.run_baselines(test_env)
    
    for name, res in baselines.items():
        print(f"\n{name}:")
        print(f"  Total Return: {res.total_return:.2%}")
        print(f"  Sharpe: {res.sharpe_ratio:.2f}")
        print(f"  Max DD: {res.max_drawdown:.2%}")
    
    # Comparison
    print("\n--- Comparison ---")
    comparison = backtester.compare_results(result, baselines)
    print(comparison.to_string())

