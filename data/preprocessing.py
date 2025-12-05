"""
Data preprocessing for training
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DataSplit:
    """Container for train/val/test split data"""
    train: Dict[str, pd.DataFrame]
    val: Dict[str, pd.DataFrame]
    test: Dict[str, pd.DataFrame]
    train_dates: Tuple[str, str]
    val_dates: Tuple[str, str]
    test_dates: Tuple[str, str]
    feature_names: List[str]
    scaler_params: Dict[str, Dict[str, np.ndarray]]


class DataPreprocessor:
    """Preprocesses data for model training"""
    
    def __init__(
        self,
        train_split: float = 0.70,
        val_split: float = 0.15,
        test_split: float = 0.15,
        lookback_window: int = 60,
    ):
        """
        Initialize preprocessor
        
        Args:
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            lookback_window: Number of days for lookback window
        """
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.lookback_window = lookback_window
        
        self.scaler_params: Dict[str, Dict[str, np.ndarray]] = {}
        self.feature_names: List[str] = []
        
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns to use as features"""
        exclude = ["open", "high", "low", "close", "volume", "symbol"]
        # Exclude cross-asset correlation features (they differ per symbol)
        return [c for c in df.columns 
                if c not in exclude 
                and not c.startswith("corr_")
                and not c.startswith("rel_strength")
                and not df[c].isna().all()]
    
    def split_data(
        self, 
        processed_data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> DataSplit:
        """
        Split data chronologically into train/val/test sets
        
        Args:
            processed_data: Dictionary mapping symbol to processed DataFrame
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataSplit object with train/val/test data
        """
        # Find common date range
        common_dates = None
        for symbol, df in processed_data.items():
            valid_idx = df.dropna().index
            if common_dates is None:
                common_dates = set(valid_idx)
            else:
                common_dates = common_dates.intersection(set(valid_idx))
        
        common_dates = sorted(list(common_dates))
        
        if start_date:
            common_dates = [d for d in common_dates if d >= pd.Timestamp(start_date)]
        if end_date:
            common_dates = [d for d in common_dates if d <= pd.Timestamp(end_date)]
        
        n_dates = len(common_dates)
        print(f"Total valid dates: {n_dates}")
        
        # Calculate split points
        train_end = int(n_dates * self.train_split)
        val_end = int(n_dates * (self.train_split + self.val_split))
        
        train_dates = common_dates[:train_end]
        val_dates = common_dates[train_end:val_end]
        test_dates = common_dates[val_end:]
        
        print(f"Train: {len(train_dates)} days ({train_dates[0].date()} to {train_dates[-1].date()})")
        print(f"Val: {len(val_dates)} days ({val_dates[0].date()} to {val_dates[-1].date()})")
        print(f"Test: {len(test_dates)} days ({test_dates[0].date()} to {test_dates[-1].date()})")
        
        # Split each symbol's data
        train_data = {}
        val_data = {}
        test_data = {}
        
        for symbol, df in processed_data.items():
            train_data[symbol] = df.loc[df.index.isin(train_dates)]
            val_data[symbol] = df.loc[df.index.isin(val_dates)]
            test_data[symbol] = df.loc[df.index.isin(test_dates)]
        
        # Get feature names from first symbol
        first_symbol = list(processed_data.keys())[0]
        self.feature_names = self.get_feature_columns(processed_data[first_symbol])
        
        # Fit scalers on training data
        self._fit_scalers(train_data)
        
        # Apply scaling
        train_data = self._scale_data(train_data)
        val_data = self._scale_data(val_data)
        test_data = self._scale_data(test_data)
        
        return DataSplit(
            train=train_data,
            val=val_data,
            test=test_data,
            train_dates=(str(train_dates[0].date()), str(train_dates[-1].date())),
            val_dates=(str(val_dates[0].date()), str(val_dates[-1].date())),
            test_dates=(str(test_dates[0].date()), str(test_dates[-1].date())),
            feature_names=self.feature_names,
            scaler_params=self.scaler_params,
        )
    
    def _fit_scalers(self, train_data: Dict[str, pd.DataFrame]):
        """Fit scalers on training data"""
        self.scaler_params = {}
        
        for symbol, df in train_data.items():
            features = df[self.feature_names].values
            
            # Use robust scaling (median and IQR) to handle outliers
            median = np.nanmedian(features, axis=0)
            q75 = np.nanpercentile(features, 75, axis=0)
            q25 = np.nanpercentile(features, 25, axis=0)
            iqr = q75 - q25
            
            # Avoid division by zero
            iqr = np.where(iqr == 0, 1, iqr)
            
            self.scaler_params[symbol] = {
                "median": median,
                "iqr": iqr,
            }
    
    def _scale_data(
        self, 
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Scale data using fitted scalers"""
        scaled_data = {}
        
        for symbol, df in data.items():
            df = df.copy()
            
            if symbol in self.scaler_params:
                params = self.scaler_params[symbol]
                median = params["median"]
                iqr = params["iqr"]
                
                # Scale features
                for i, col in enumerate(self.feature_names):
                    if col in df.columns:
                        df[col] = (df[col] - median[i]) / iqr[i]
            
            scaled_data[symbol] = df
        
        return scaled_data
    
    def create_sequences(
        self,
        data: Dict[str, pd.DataFrame],
        lookback: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """
        Create sequences for training
        
        Args:
            data: Dictionary of processed DataFrames
            lookback: Lookback window (default: self.lookback_window)
            
        Returns:
            Tuple of (features, prices, returns, dates)
            - features: (n_samples, lookback, n_features * n_symbols)
            - prices: (n_samples, n_symbols) - current prices
            - returns: (n_samples, n_symbols) - next day returns
            - dates: List of dates
        """
        lookback = lookback or self.lookback_window
        symbols = list(data.keys())
        n_symbols = len(symbols)
        
        # Align all data to common dates
        common_dates = None
        for symbol, df in data.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        
        common_dates = sorted(list(common_dates))
        
        # Prepare aligned data
        aligned_features = {}
        aligned_prices = {}
        aligned_returns = {}
        
        for symbol in symbols:
            df = data[symbol].loc[common_dates]
            aligned_features[symbol] = df[self.feature_names].values
            aligned_prices[symbol] = df["close"].values
            aligned_returns[symbol] = df["returns"].values
        
        n_dates = len(common_dates)
        n_features = len(self.feature_names)
        
        # Create sequences
        sequences = []
        prices = []
        returns = []
        dates = []
        
        for i in range(lookback, n_dates - 1):  # -1 to have next day return
            # Combine features from all symbols
            seq_features = np.zeros((lookback, n_features * n_symbols))
            
            for j, symbol in enumerate(symbols):
                start_idx = j * n_features
                end_idx = (j + 1) * n_features
                seq_features[:, start_idx:end_idx] = aligned_features[symbol][i-lookback:i]
            
            # Current prices for all symbols
            current_prices = np.array([aligned_prices[symbol][i] for symbol in symbols])
            
            # Next day returns for all symbols
            next_returns = np.array([aligned_returns[symbol][i+1] for symbol in symbols])
            
            sequences.append(seq_features)
            prices.append(current_prices)
            returns.append(next_returns)
            dates.append(common_dates[i])
        
        return (
            np.array(sequences),
            np.array(prices),
            np.array(returns),
            dates,
        )
    
    def prepare_training_data(
        self,
        processed_data: Dict[str, pd.DataFrame],
    ) -> Tuple[DataSplit, Dict[str, Tuple]]:
        """
        Full pipeline to prepare training data
        
        Args:
            processed_data: Dictionary of processed DataFrames
            
        Returns:
            Tuple of (DataSplit, sequences_dict)
            sequences_dict maps 'train', 'val', 'test' to sequence tuples
        """
        # Split data
        split = self.split_data(processed_data)
        
        # Create sequences for each split
        sequences = {}
        for name, data in [("train", split.train), ("val", split.val), ("test", split.test)]:
            print(f"\nCreating {name} sequences...")
            seqs = self.create_sequences(data)
            sequences[name] = seqs
            print(f"  Shape: features={seqs[0].shape}, prices={seqs[1].shape}, returns={seqs[2].shape}")
        
        return split, sequences


if __name__ == "__main__":
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    
    # Load and process data
    loader = DataLoader()
    data = loader.download_all()
    
    engineer = FeatureEngineer()
    processed = engineer.process_all_symbols(data)
    processed = engineer.add_cross_asset_features(processed)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    split, sequences = preprocessor.prepare_training_data(processed)
    
    print("\n--- Data Split ---")
    print(f"Train dates: {split.train_dates}")
    print(f"Val dates: {split.val_dates}")
    print(f"Test dates: {split.test_dates}")
    print(f"Feature names ({len(split.feature_names)}): {split.feature_names[:10]}...")
    
    print("\n--- Training Sequences ---")
    X_train, prices_train, returns_train, dates_train = sequences["train"]
    print(f"X_train: {X_train.shape}")
    print(f"prices_train: {prices_train.shape}")
    print(f"returns_train: {returns_train.shape}")
    print(f"Sample X range: [{X_train.min():.2f}, {X_train.max():.2f}]")

