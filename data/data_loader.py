"""
Data loader for downloading and caching historical index data
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import yfinance as yf


class DataLoader:
    """Downloads and caches historical data for US index ETFs"""
    
    # ETF inception dates (approximate)
    INCEPTION_DATES = {
        "SPY": "1993-01-29",  # S&P 500
        "QQQ": "1999-03-10",  # NASDAQ-100
        "DIA": "1998-01-14",  # Dow Jones
        "IWM": "2000-05-22",  # Russell 2000
    }
    
    def __init__(self, cache_dir: str = "data/cache", symbols: Optional[List[str]] = None):
        """
        Initialize the data loader
        
        Args:
            cache_dir: Directory to cache downloaded data
            symbols: List of symbols to download (defaults to all 4 indexes)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.symbols = symbols or list(self.INCEPTION_DATES.keys())
        
    def download_symbol(
        self, 
        symbol: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Download historical data for a single symbol
        
        Args:
            symbol: Ticker symbol (e.g., 'SPY')
            start_date: Start date (defaults to inception)
            end_date: End date (defaults to today)
            force_refresh: If True, redownload even if cached
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_file = self.cache_dir / f"{symbol}_daily.parquet"
        
        # Check cache first
        if cache_file.exists() and not force_refresh:
            df = pd.read_parquet(cache_file)
            print(f"Loaded {symbol} from cache: {len(df)} rows")
            
            # Update if data is stale (more than 1 day old)
            last_date = df.index.max()
            if (datetime.now() - last_date).days > 1:
                print(f"Updating {symbol} data...")
                new_df = self._fetch_data(symbol, str(last_date.date()), end_date)
                if not new_df.empty:
                    df = pd.concat([df, new_df]).drop_duplicates()
                    df.to_parquet(cache_file)
            return df
        
        # Download from yfinance
        start = start_date or self.INCEPTION_DATES.get(symbol, "1990-01-01")
        df = self._fetch_data(symbol, start, end_date)
        
        if not df.empty:
            df.to_parquet(cache_file)
            print(f"Downloaded {symbol}: {len(df)} rows from {df.index.min()} to {df.index.max()}")
        
        return df
    
    def _fetch_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                print(f"Warning: No data returned for {symbol}")
                return pd.DataFrame()
            
            # Rename columns to lowercase for consistency
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            
            # Keep only OHLCV columns
            keep_cols = ["open", "high", "low", "close", "volume"]
            df = df[[c for c in keep_cols if c in df.columns]]
            
            # Ensure datetime index
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)  # Remove timezone info
            
            return df
            
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return pd.DataFrame()
    
    def download_all(
        self, 
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for all symbols
        
        Args:
            force_refresh: If True, redownload even if cached
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        for symbol in self.symbols:
            df = self.download_symbol(symbol, force_refresh=force_refresh)
            if not df.empty:
                data[symbol] = df
        return data
    
    def get_aligned_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get all symbols aligned to the same dates
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            force_refresh: If True, redownload data
            
        Returns:
            MultiIndex DataFrame with (date, symbol) as index
        """
        data = self.download_all(force_refresh=force_refresh)
        
        if not data:
            raise ValueError("No data downloaded")
        
        # Find common date range
        common_start = max(df.index.min() for df in data.values())
        common_end = min(df.index.max() for df in data.values())
        
        if start_date:
            common_start = max(common_start, pd.Timestamp(start_date))
        if end_date:
            common_end = min(common_end, pd.Timestamp(end_date))
        
        print(f"Common date range: {common_start.date()} to {common_end.date()}")
        
        # Align all data to common dates
        aligned = {}
        for symbol, df in data.items():
            mask = (df.index >= common_start) & (df.index <= common_end)
            aligned[symbol] = df[mask]
        
        # Create multi-index DataFrame
        dfs = []
        for symbol, df in aligned.items():
            df = df.copy()
            df["symbol"] = symbol
            dfs.append(df)
        
        combined = pd.concat(dfs)
        combined = combined.reset_index()
        combined = combined.rename(columns={"index": "date", "Date": "date"})
        combined = combined.set_index(["date", "symbol"])
        combined = combined.sort_index()
        
        return combined
    
    def get_prices_matrix(
        self,
        price_col: str = "close",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get price matrix with dates as index and symbols as columns
        
        Args:
            price_col: Price column to use ('open', 'high', 'low', 'close')
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with shape (n_dates, n_symbols)
        """
        data = self.download_all()
        
        if not data:
            raise ValueError("No data downloaded")
        
        # Create price matrix
        prices = pd.DataFrame()
        for symbol, df in data.items():
            if price_col in df.columns:
                prices[symbol] = df[price_col]
        
        # Apply date filters
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]
        
        # Drop rows with any NaN (non-trading days)
        prices = prices.dropna()
        
        return prices


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    
    # Download all data
    data = loader.download_all()
    
    for symbol, df in data.items():
        print(f"\n{symbol}:")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
    
    # Get aligned data
    print("\n--- Aligned Data ---")
    aligned = loader.get_aligned_data()
    print(f"Shape: {aligned.shape}")
    print(aligned.head())
    
    # Get price matrix
    print("\n--- Price Matrix ---")
    prices = loader.get_prices_matrix()
    print(f"Shape: {prices.shape}")
    print(prices.tail())

