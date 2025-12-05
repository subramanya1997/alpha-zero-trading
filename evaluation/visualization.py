"""
Visualization module for backtest results
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.backtest import BacktestResult


class Visualizer:
    """Visualization tools for backtest results"""
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use("seaborn-v0_8")
        
        self.colors = {
            "agent": "#2ecc71",
            "benchmark": "#3498db",
            "drawdown": "#e74c3c",
            "positive": "#2ecc71",
            "negative": "#e74c3c",
        }
    
    def plot_equity_curve(
        self,
        results: Dict[str, BacktestResult],
        title: str = "Portfolio Equity Curve",
        figsize: Tuple[int, int] = (14, 7),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot equity curves for multiple strategies
        
        Args:
            results: Dictionary of strategy name to BacktestResult
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for (name, result), color in zip(results.items(), colors):
            values = np.array(result.portfolio_values)
            normalized = values / values[0]  # Normalize to start at 1
            
            ax.plot(normalized, label=f"{name} ({result.total_return:.1%})", 
                   color=color, linewidth=2)
        
        ax.set_xlabel("Trading Days", fontsize=12)
        ax.set_ylabel("Portfolio Value (Normalized)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}x'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig
    
    def plot_drawdowns(
        self,
        result: BacktestResult,
        title: str = "Portfolio Drawdown",
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot drawdown over time
        
        Args:
            result: BacktestResult
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        drawdowns = np.array(result.drawdowns) * 100  # Convert to percentage
        
        ax.fill_between(range(len(drawdowns)), drawdowns, 0, 
                       color=self.colors["drawdown"], alpha=0.3)
        ax.plot(drawdowns, color=self.colors["drawdown"], linewidth=1)
        
        # Highlight max drawdown
        max_dd_idx = np.argmin(drawdowns)
        max_dd = drawdowns[max_dd_idx]
        ax.axhline(y=max_dd, color="red", linestyle="--", alpha=0.7,
                  label=f"Max DD: {max_dd:.1f}%")
        ax.scatter([max_dd_idx], [max_dd], color="red", s=100, zorder=5)
        
        ax.set_xlabel("Trading Days", fontsize=12)
        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig
    
    def plot_returns_distribution(
        self,
        result: BacktestResult,
        title: str = "Daily Returns Distribution",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot distribution of daily returns
        
        Args:
            result: BacktestResult
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        returns = np.array(result.daily_returns) * 100
        
        # Histogram
        n, bins, patches = ax.hist(returns, bins=50, density=True, 
                                   alpha=0.7, color=self.colors["agent"])
        
        # Color positive/negative differently
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge < 0:
                patch.set_facecolor(self.colors["negative"])
            else:
                patch.set_facecolor(self.colors["positive"])
        
        # Normal distribution overlay
        mu, sigma = np.mean(returns), np.std(returns)
        x = np.linspace(min(returns), max(returns), 100)
        normal_dist = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, normal_dist, 'k--', linewidth=2, label=f'Normal (μ={mu:.2f}%, σ={sigma:.2f}%)')
        
        # Add statistics text
        stats_text = f"Mean: {mu:.3f}%\nStd: {sigma:.3f}%\nSkew: {pd.Series(returns).skew():.2f}\nKurt: {pd.Series(returns).kurtosis():.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel("Daily Return (%)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig
    
    def plot_position_allocations(
        self,
        result: BacktestResult,
        title: str = "Position Allocations Over Time",
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot position allocations over time
        
        Args:
            result: BacktestResult
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not result.positions:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert positions to DataFrame
        positions_df = pd.DataFrame(result.positions)
        
        # Stack area plot
        positions_df.plot.area(ax=ax, alpha=0.7, linewidth=0)
        
        ax.set_xlabel("Trading Days", fontsize=12)
        ax.set_ylabel("Position Weight", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig
    
    def plot_leverage_history(
        self,
        result: BacktestResult,
        title: str = "Leverage Over Time",
        figsize: Tuple[int, int] = (14, 4),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot leverage usage over time
        
        Args:
            result: BacktestResult
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        leverages = np.array(result.leverages)
        
        ax.fill_between(range(len(leverages)), leverages, 0,
                       color=self.colors["agent"], alpha=0.3)
        ax.plot(leverages, color=self.colors["agent"], linewidth=1)
        
        # Add average line
        avg_lev = np.mean(leverages)
        ax.axhline(y=avg_lev, color="orange", linestyle="--",
                  label=f"Avg: {avg_lev:.1f}x")
        
        ax.set_xlabel("Trading Days", fontsize=12)
        ax.set_ylabel("Leverage (x)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig
    
    def plot_monthly_returns(
        self,
        result: BacktestResult,
        title: str = "Monthly Returns Heatmap",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot monthly returns heatmap
        
        Args:
            result: BacktestResult
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not result.dates or None in result.dates:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create DataFrame with dates
        df = pd.DataFrame({
            "date": result.dates[1:],  # Skip first date (no return)
            "return": result.daily_returns,
        })
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        # Calculate monthly returns
        monthly = df.resample("M")["return"].apply(lambda x: (1 + x).prod() - 1)
        
        # Pivot to year/month format
        monthly_df = pd.DataFrame(monthly)
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        
        pivot = monthly_df.pivot(index="year", columns="month", values="return")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        # Heatmap
        sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn",
                   center=0, ax=ax, cbar_kws={"label": "Return (%)"})
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Year", fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig
    
    def create_full_report(
        self,
        agent_result: BacktestResult,
        baseline_results: Dict[str, BacktestResult],
        save_dir: Optional[str] = None,
    ) -> List[plt.Figure]:
        """
        Create full visualization report
        
        Args:
            agent_result: Agent backtest result
            baseline_results: Dictionary of baseline results
            save_dir: Optional directory to save figures
            
        Returns:
            List of figures
        """
        figures = []
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Equity curves comparison
        all_results = {"Agent": agent_result, **baseline_results}
        fig = self.plot_equity_curve(
            all_results,
            title="Equity Curve Comparison",
            save_path=f"{save_dir}/equity_curve.png" if save_dir else None,
        )
        figures.append(fig)
        
        # 2. Agent drawdown
        fig = self.plot_drawdowns(
            agent_result,
            title="Agent Portfolio Drawdown",
            save_path=f"{save_dir}/drawdown.png" if save_dir else None,
        )
        figures.append(fig)
        
        # 3. Returns distribution
        fig = self.plot_returns_distribution(
            agent_result,
            title="Agent Daily Returns Distribution",
            save_path=f"{save_dir}/returns_dist.png" if save_dir else None,
        )
        figures.append(fig)
        
        # 4. Position allocations
        fig = self.plot_position_allocations(
            agent_result,
            save_path=f"{save_dir}/positions.png" if save_dir else None,
        )
        if fig:
            figures.append(fig)
        
        # 5. Leverage history
        fig = self.plot_leverage_history(
            agent_result,
            save_path=f"{save_dir}/leverage.png" if save_dir else None,
        )
        figures.append(fig)
        
        # 6. Monthly returns
        fig = self.plot_monthly_returns(
            agent_result,
            save_path=f"{save_dir}/monthly_returns.png" if save_dir else None,
        )
        if fig:
            figures.append(fig)
        
        return figures


if __name__ == "__main__":
    # Test visualization
    np.random.seed(42)
    
    # Create dummy results
    n_days = 252
    
    agent_result = BacktestResult(
        total_return=0.35,
        sharpe_ratio=1.5,
        max_drawdown=-0.08,
        portfolio_values=list(10000 * np.cumprod(1 + np.random.normal(0.001, 0.01, n_days))),
        daily_returns=list(np.random.normal(0.001, 0.01, n_days)),
        drawdowns=list(np.random.uniform(-0.08, 0, n_days)),
        positions=[{"SPY": 0.3, "QQQ": 0.3, "DIA": 0.2, "IWM": 0.2} for _ in range(n_days)],
        leverages=list(np.random.uniform(5, 8, n_days)),
    )
    
    benchmark_result = BacktestResult(
        total_return=0.15,
        sharpe_ratio=0.8,
        max_drawdown=-0.12,
        portfolio_values=list(10000 * np.cumprod(1 + np.random.normal(0.0005, 0.012, n_days))),
        daily_returns=list(np.random.normal(0.0005, 0.012, n_days)),
        drawdowns=list(np.random.uniform(-0.12, 0, n_days)),
    )
    
    # Create visualizer
    viz = Visualizer()
    
    # Plot equity curves
    print("Creating equity curve plot...")
    fig = viz.plot_equity_curve(
        {"Agent": agent_result, "Benchmark": benchmark_result},
        save_path="test_equity.png",
    )
    
    print("Creating drawdown plot...")
    fig = viz.plot_drawdowns(agent_result, save_path="test_drawdown.png")
    
    print("Creating returns distribution...")
    fig = viz.plot_returns_distribution(agent_result, save_path="test_returns.png")
    
    print("Creating position allocations...")
    fig = viz.plot_position_allocations(agent_result, save_path="test_positions.png")
    
    print("Creating leverage history...")
    fig = viz.plot_leverage_history(agent_result, save_path="test_leverage.png")
    
    print("\nVisualization test complete!")
    plt.close("all")

