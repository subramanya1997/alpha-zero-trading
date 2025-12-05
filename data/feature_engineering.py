"""
Feature engineering for technical indicators and market features
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional


class FeatureEngineer:
    """Generates technical indicators and market features for trading"""
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize feature engineer
        
        Args:
            symbols: List of symbols to process
        """
        self.symbols = symbols or ["SPY", "QQQ", "DIA", "IWM"]
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to a single symbol's OHLCV data
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Price-based features
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f"sma_{period}"] = ta.sma(df["close"], length=period)
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)
        
        # Price relative to moving averages
        df["price_sma20_ratio"] = df["close"] / df["sma_20"]
        df["price_sma50_ratio"] = df["close"] / df["sma_50"]
        df["price_sma200_ratio"] = df["close"] / df["sma_200"]
        
        # Moving average crossovers
        df["sma_5_20_cross"] = df["sma_5"] / df["sma_20"]
        df["sma_20_50_cross"] = df["sma_20"] / df["sma_50"]
        df["sma_50_200_cross"] = df["sma_50"] / df["sma_200"]
        
        # Momentum indicators
        df["rsi_14"] = ta.rsi(df["close"], length=14)
        df["rsi_7"] = ta.rsi(df["close"], length=7)
        df["rsi_21"] = ta.rsi(df["close"], length=21)
        
        # MACD
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            macd_cols = macd.columns.tolist()
            macd_col = [c for c in macd_cols if c.startswith("MACD_") or c == "MACD"][0] if any(c.startswith("MACD") for c in macd_cols) else None
            macd_signal_col = [c for c in macd_cols if "MACDs" in c or "signal" in c.lower()][0] if any("MACDs" in c for c in macd_cols) else None
            macd_hist_col = [c for c in macd_cols if "MACDh" in c or "hist" in c.lower()][0] if any("MACDh" in c for c in macd_cols) else None
            
            if macd_col:
                df["macd"] = macd[macd_col]
            if macd_signal_col:
                df["macd_signal"] = macd[macd_signal_col]
            if macd_hist_col:
                df["macd_hist"] = macd[macd_hist_col]
        
        # Stochastic
        stoch = ta.stoch(df["high"], df["low"], df["close"])
        if stoch is not None and not stoch.empty:
            stoch_cols = stoch.columns.tolist()
            stoch_k_col = [c for c in stoch_cols if "k" in c.lower()][0] if any("k" in c.lower() for c in stoch_cols) else None
            stoch_d_col = [c for c in stoch_cols if "d" in c.lower()][0] if any("d" in c.lower() for c in stoch_cols) else None
            if stoch_k_col:
                df["stoch_k"] = stoch[stoch_k_col]
            if stoch_d_col:
                df["stoch_d"] = stoch[stoch_d_col]
        
        # Bollinger Bands
        bbands = ta.bbands(df["close"], length=20, std=2)
        if bbands is not None and not bbands.empty:
            # Find the correct column names (they vary by pandas-ta version)
            bb_cols = bbands.columns.tolist()
            bb_upper_col = [c for c in bb_cols if c.startswith("BBU")][0] if any(c.startswith("BBU") for c in bb_cols) else None
            bb_middle_col = [c for c in bb_cols if c.startswith("BBM")][0] if any(c.startswith("BBM") for c in bb_cols) else None
            bb_lower_col = [c for c in bb_cols if c.startswith("BBL")][0] if any(c.startswith("BBL") for c in bb_cols) else None
            
            if bb_upper_col and bb_middle_col and bb_lower_col:
                df["bb_upper"] = bbands[bb_upper_col]
                df["bb_middle"] = bbands[bb_middle_col]
                df["bb_lower"] = bbands[bb_lower_col]
                df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
                df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # ATR (Average True Range) - volatility
        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_7"] = ta.atr(df["high"], df["low"], df["close"], length=7)
        df["atr_pct"] = df["atr_14"] / df["close"]  # ATR as percentage of price
        
        # Volatility features
        df["volatility_20"] = df["returns"].rolling(window=20).std() * np.sqrt(252)
        df["volatility_60"] = df["returns"].rolling(window=60).std() * np.sqrt(252)
        
        # Volume features
        df["volume_sma_20"] = ta.sma(df["volume"], length=20)
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        df["volume_change"] = df["volume"].pct_change()
        
        # OBV (On Balance Volume)
        df["obv"] = ta.obv(df["close"], df["volume"])
        df["obv_sma"] = ta.sma(df["obv"], length=20)
        
        # ADX (Average Directional Index) - trend strength
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None and not adx.empty:
            adx_cols = adx.columns.tolist()
            adx_col = [c for c in adx_cols if c.startswith("ADX")][0] if any(c.startswith("ADX") for c in adx_cols) else None
            dmp_col = [c for c in adx_cols if c.startswith("DMP")][0] if any(c.startswith("DMP") for c in adx_cols) else None
            dmn_col = [c for c in adx_cols if c.startswith("DMN")][0] if any(c.startswith("DMN") for c in adx_cols) else None
            
            if adx_col:
                df["adx"] = adx[adx_col]
            if dmp_col:
                df["di_plus"] = adx[dmp_col]
            if dmn_col:
                df["di_minus"] = adx[dmn_col]
        
        # CCI (Commodity Channel Index)
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
        
        # Williams %R
        df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=14)
        
        # Rate of Change
        df["roc_5"] = ta.roc(df["close"], length=5)
        df["roc_10"] = ta.roc(df["close"], length=10)
        df["roc_20"] = ta.roc(df["close"], length=20)
        
        # Candle patterns
        df["candle_body"] = (df["close"] - df["open"]) / df["open"]
        df["candle_wick_upper"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
        df["candle_wick_lower"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
        
        # High-Low range
        df["daily_range"] = (df["high"] - df["low"]) / df["close"]
        
        # Cumulative returns
        df["cum_return_5"] = df["close"] / df["close"].shift(5) - 1
        df["cum_return_10"] = df["close"] / df["close"].shift(10) - 1
        df["cum_return_20"] = df["close"] / df["close"].shift(20) - 1
        df["cum_return_60"] = df["close"] / df["close"].shift(60) - 1
        
        # Drawdown features
        rolling_max = df["close"].rolling(window=252, min_periods=1).max()
        df["drawdown"] = (df["close"] - rolling_max) / rolling_max
        
        return df
    
    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime indicators
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with regime features
        """
        df = df.copy()
        
        # Trend regime (using SMA slope)
        if "sma_50" in df.columns and "sma_200" in df.columns:
            df["sma_50_slope"] = df["sma_50"].pct_change(periods=10)
            df["trend_regime"] = np.where(
                df["close"] > df["sma_200"], 
                np.where(df["sma_50"] > df["sma_200"], 1, 0.5),  # Bullish / Weak bullish
                np.where(df["sma_50"] < df["sma_200"], -1, -0.5)  # Bearish / Weak bearish
            )
        
        # Volatility regime
        if "volatility_20" in df.columns:
            vol_20 = df["volatility_20"]
            vol_median = vol_20.rolling(window=252).median()
            df["vol_regime"] = np.where(vol_20 > vol_median * 1.5, 2,  # High vol
                              np.where(vol_20 < vol_median * 0.75, 0, 1))  # Low / Normal vol
        
        # Momentum regime
        if "rsi_14" in df.columns and "macd_hist" in df.columns and "macd" in df.columns:
            df["momentum_regime"] = np.where(
                (df["rsi_14"] > 70) | ((df["macd_hist"] > 0) & (df["macd"] > 0)), 1,  # Strong momentum
                np.where((df["rsi_14"] < 30) | ((df["macd_hist"] < 0) & (df["macd"] < 0)), -1, 0)  # Weak / Neutral
            )
        
        return df
    
    def process_all_symbols(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process all symbols and add features
        
        Args:
            data: Dictionary mapping symbol to OHLCV DataFrame
            
        Returns:
            Dictionary mapping symbol to processed DataFrame
        """
        processed = {}
        for symbol, df in data.items():
            print(f"Processing {symbol}...")
            processed_df = self.add_technical_indicators(df)
            processed_df = self.add_market_regime_features(processed_df)
            processed[symbol] = processed_df
            print(f"  Added {len(processed_df.columns)} features")
        
        return processed
    
    def add_cross_asset_features(
        self, 
        processed_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Add cross-asset correlation and relative strength features
        
        Args:
            processed_data: Dictionary of processed DataFrames
            
        Returns:
            Updated dictionary with cross-asset features
        """
        symbols = list(processed_data.keys())
        
        # Get aligned returns
        returns_df = pd.DataFrame()
        for symbol, df in processed_data.items():
            if "returns" in df.columns:
                returns_df[symbol] = df["returns"]
        
        returns_df = returns_df.dropna()
        
        # Rolling correlations
        for symbol in symbols:
            if symbol not in processed_data:
                continue
            df = processed_data[symbol]
            
            for other_symbol in symbols:
                if other_symbol == symbol:
                    continue
                
                if symbol in returns_df.columns and other_symbol in returns_df.columns:
                    corr_col = f"corr_{symbol}_{other_symbol}"
                    rolling_corr = returns_df[symbol].rolling(window=60).corr(
                        returns_df[other_symbol]
                    )
                    df[corr_col] = rolling_corr.reindex(df.index)
            
            # Relative strength vs SPY
            if symbol != "SPY" and "SPY" in processed_data:
                spy_close = processed_data["SPY"]["close"].reindex(df.index)
                df[f"rel_strength_spy"] = df["close"] / spy_close
                df[f"rel_strength_spy_sma"] = ta.sma(df["rel_strength_spy"], length=20)
            
            processed_data[symbol] = df
        
        return processed_data
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names generated"""
        # Create a dummy dataframe to get feature names
        dummy = pd.DataFrame({
            "open": [100] * 300,
            "high": [101] * 300,
            "low": [99] * 300,
            "close": [100] * 300,
            "volume": [1000000] * 300
        })
        processed = self.add_technical_indicators(dummy)
        processed = self.add_market_regime_features(processed)
        
        return list(processed.columns)


if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    data = loader.download_all()
    
    # Process features
    engineer = FeatureEngineer()
    processed = engineer.process_all_symbols(data)
    processed = engineer.add_cross_asset_features(processed)
    
    # Show sample
    for symbol, df in processed.items():
        print(f"\n{symbol}:")
        print(f"  Shape: {df.shape}")
        print(f"  Features: {list(df.columns[:10])}...")
        print(f"  NaN counts (first 10): {df.isna().sum().head(10).tolist()}")
    
    # Show feature names
    print("\n--- All Features ---")
    features = engineer.get_feature_names()
    print(f"Total features: {len(features)}")
    print(features)

