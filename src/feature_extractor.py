"""
Feature extraction module for stock price analysis and clustering.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple


class FeatureExtractor:
    """
    Extracts features from stock price data for clustering analysis.
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.logger = logging.getLogger(__name__)
        
    def calculate_basic_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic return metrics for each stock.
        
        Args:
            df: DataFrame with stock price data
            
        Returns:
            DataFrame with additional return columns
        """
        df = df.copy()
        
        # Group by symbol to calculate returns per stock
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].sort_values('date')
            
            # Daily returns
            daily_returns = symbol_data['close'].pct_change()
            df.loc[mask, 'daily_return'] = daily_returns
            
            # Log returns
            df.loc[mask, 'log_return'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
            
            # Cumulative returns
            df.loc[mask, 'cumulative_return'] = (1 + daily_returns).cumprod() - 1
            
        return df
    
    def calculate_volatility_features(self, df: pd.DataFrame, windows: List[int] = [30, 90, 252]) -> pd.DataFrame:
        """
        Calculate volatility features using multiple rolling windows.
        
        Args:
            df: DataFrame with stock price data
            windows: List of rolling window sizes in days
            
        Returns:
            DataFrame with volatility features
        """
        df = df.copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].sort_values('date')
            
            for window in windows:
                if len(symbol_data) >= window:
                    # Rolling volatility (standard deviation of returns)
                    rolling_vol = symbol_data['daily_return'].rolling(window=window).std() * np.sqrt(252)
                    df.loc[mask, f'volatility_{window}d'] = rolling_vol
                    
                    # Rolling mean of returns
                    rolling_mean_ret = symbol_data['daily_return'].rolling(window=window).mean() * 252
                    df.loc[mask, f'mean_return_{window}d'] = rolling_mean_ret
                    
                    # Volatility regime changes (change in volatility)
                    df.loc[mask, f'vol_change_{window}d'] = rolling_vol.pct_change()
                    
        return df
    
    def count_fluctuation_cycles(self, df: pd.DataFrame, 
                               min_pct: float = 30, max_pct: float = 70,
                               price_reference: str = 'close') -> Dict[str, Dict]:
        """
        Count fluctuation cycles between percentage thresholds.
        
        Args:
            df: DataFrame with stock price data
            min_pct: Minimum percentage threshold
            max_pct: Maximum percentage threshold  
            price_reference: Price column to use for calculations
            
        Returns:
            Dictionary with fluctuation statistics per symbol
        """
        fluctuation_stats = {}
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('date').copy()
            
            if len(symbol_data) < 100:  # Need sufficient data
                continue
                
            # Calculate percentage from historical minimum
            min_price = symbol_data[price_reference].min()
            max_price = symbol_data[price_reference].max()
            price_range = max_price - min_price
            
            if price_range == 0:
                continue
                
            # Calculate percentage from minimum
            symbol_data['pct_from_min'] = (symbol_data[price_reference] - min_price) / price_range * 100
            
            # Identify crossings of thresholds
            crosses_min = symbol_data['pct_from_min'] >= min_pct
            crosses_max = symbol_data['pct_from_min'] >= max_pct
            
            # Find cycle starts and ends
            cycle_starts = []
            cycle_ends = []
            
            in_cycle = False
            for i in range(len(symbol_data)):
                if not in_cycle and crosses_min.iloc[i]:
                    cycle_starts.append(i)
                    in_cycle = True
                elif in_cycle and crosses_max.iloc[i]:
                    cycle_ends.append(i)
                    in_cycle = False
                    
            # Calculate cycle statistics
            if cycle_starts and cycle_ends:
                cycle_count = min(len(cycle_starts), len(cycle_ends))
                cycle_periods = []
                
                for i in range(cycle_count):
                    period_days = (symbol_data.iloc[cycle_ends[i]]['date'] - 
                                 symbol_data.iloc[cycle_starts[i]]['date']).days
                    cycle_periods.append(period_days)
                
                fluctuation_stats[symbol] = {
                    'fluctuation_count': cycle_count,
                    'avg_fluctuation_period': np.mean(cycle_periods) if cycle_periods else 0,
                    'min_fluctuation_period': min(cycle_periods) if cycle_periods else 0,
                    'max_fluctuation_period': max(cycle_periods) if cycle_periods else 0,
                    'fluctuation_frequency': cycle_count / (len(symbol_data) / 252) if len(symbol_data) >= 252 else 0
                }
            else:
                fluctuation_stats[symbol] = {
                    'fluctuation_count': 0,
                    'avg_fluctuation_period': 0,
                    'min_fluctuation_period': 0,
                    'max_fluctuation_period': 0,
                    'fluctuation_frequency': 0
                }
                
        return fluctuation_stats
    
    def calculate_trend_features(self, df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """
        Calculate trend-related features.
        
        Args:
            df: DataFrame with stock price data
            periods: List of moving average periods
            
        Returns:
            DataFrame with trend features
        """
        df = df.copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].sort_values('date')
            
            for period in periods:
                if len(symbol_data) >= period:
                    # Moving averages
                    ma = symbol_data['close'].rolling(window=period).mean()
                    df.loc[mask, f'ma_{period}'] = ma
                    
                    # Price to MA ratio
                    df.loc[mask, f'price_to_ma_{period}'] = symbol_data['close'] / ma
                    
                    # Trend strength (correlation with time)
                    if len(symbol_data) >= period:
                        trend_window = symbol_data['close'].rolling(window=period)
                        trend_strength = trend_window.apply(
                            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
                        )
                        df.loc[mask, f'trend_strength_{period}'] = trend_strength
                        
        return df
    
    def calculate_drawdown_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate drawdown-related features.
        
        Args:
            df: DataFrame with stock price data
            
        Returns:
            DataFrame with drawdown features
        """
        df = df.copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].sort_values('date')
            
            # Calculate running maximum
            running_max = symbol_data['close'].expanding().max()
            
            # Calculate drawdown
            drawdown = (symbol_data['close'] - running_max) / running_max * 100
            df.loc[mask, 'drawdown'] = drawdown
            
            # Maximum drawdown (rolling)
            df.loc[mask, 'max_drawdown_252d'] = drawdown.rolling(window=252).min()
            
            # Drawdown duration (consecutive days in drawdown)
            in_drawdown = drawdown < 0
            drawdown_duration = in_drawdown.groupby((in_drawdown != in_drawdown.shift()).cumsum()).cumsum()
            df.loc[mask, 'drawdown_duration'] = drawdown_duration
            
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic technical indicators.
        
        Args:
            df: DataFrame with stock price data
            
        Returns:
            DataFrame with technical indicators
        """
        df = df.copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].sort_values('date')
            
            if len(symbol_data) < 14:
                continue
                
            # RSI (Relative Strength Index)
            delta = symbol_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            df.loc[mask, 'rsi'] = rsi
            
            # MACD (Moving Average Convergence Divergence)
            ema_12 = symbol_data['close'].ewm(span=12).mean()
            ema_26 = symbol_data['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal_line = macd.ewm(span=9).mean()
            df.loc[mask, 'macd'] = macd
            df.loc[mask, 'macd_signal'] = signal_line
            df.loc[mask, 'macd_histogram'] = macd - signal_line
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = symbol_data['close'].rolling(window=bb_period).mean()
            bb_std_dev = symbol_data['close'].rolling(window=bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            
            df.loc[mask, 'bb_middle'] = bb_middle
            df.loc[mask, 'bb_upper'] = bb_upper
            df.loc[mask, 'bb_lower'] = bb_lower
            df.loc[mask, 'bb_position'] = (symbol_data['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Volume indicators
            if 'volume' in symbol_data.columns:
                # Volume moving average
                df.loc[mask, 'volume_ma_20'] = symbol_data['volume'].rolling(window=20).mean()
                
                # Volume ratio to average
                df.loc[mask, 'volume_ratio'] = symbol_data['volume'] / df.loc[mask, 'volume_ma_20']
                
        return df
    
    def calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistical features for each stock.
        
        Args:
            df: DataFrame with stock price data
            
        Returns:
            DataFrame with additional statistical features
        """
        df = df.copy()
        
        statistical_features = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if len(symbol_data) < 30:
                continue
                
            returns = symbol_data['daily_return'].dropna()
            
            if len(returns) == 0:
                continue
                
            # Basic statistics
            stats_dict = {
                'symbol': symbol,
                'return_mean': returns.mean(),
                'return_std': returns.std(),
                'return_skew': stats.skew(returns) if len(returns) > 1 else 0,
                'return_kurtosis': stats.kurtosis(returns) if len(returns) > 1 else 0,
                'return_min': returns.min(),
                'return_max': returns.max(),
                'positive_return_pct': (returns > 0).mean() * 100,
                'negative_return_pct': (returns < 0).mean() * 100,
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'var_95': np.percentile(returns, 5),  # 5% VaR
                'var_99': np.percentile(returns, 1),  # 1% VaR
                'total_return': symbol_data['cumulative_return'].iloc[-1] if 'cumulative_return' in symbol_data.columns else 0,
                'annualized_return': ((symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0]) ** (252 / len(symbol_data)) - 1) if len(symbol_data) > 0 and symbol_data['close'].iloc[0] > 0 else 0
            }
            
            statistical_features.append(stats_dict)
        
        # Convert to DataFrame and merge back
        if statistical_features:
            stats_df = pd.DataFrame(statistical_features)
            df = df.merge(stats_df, on='symbol', how='left')
        
        return df
    
    def extract_features_for_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features needed for clustering analysis.
        
        Args:
            df: DataFrame with stock price data
            
        Returns:
            DataFrame with all features extracted
        """
        self.logger.info("Starting feature extraction process...")
        
        # Step 1: Calculate basic returns
        df = self.calculate_basic_returns(df)
        self.logger.info("Basic returns calculated")
        
        # Step 2: Calculate volatility features
        df = self.calculate_volatility_features(df)
        self.logger.info("Volatility features calculated")
        
        # Step 3: Calculate trend features
        df = self.calculate_trend_features(df)
        self.logger.info("Trend features calculated")
        
        # Step 4: Calculate drawdown features
        df = self.calculate_drawdown_features(df)
        self.logger.info("Drawdown features calculated")
        
        # Step 5: Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        self.logger.info("Technical indicators calculated")
        
        # Step 6: Calculate statistical features
        df = self.calculate_statistical_features(df)
        self.logger.info("Statistical features calculated")
        
        return df
    
    def create_feature_matrix(self, df: pd.DataFrame, features_per_symbol: bool = True) -> pd.DataFrame:
        """
        Create feature matrix suitable for clustering.
        
        Args:
            df: DataFrame with all features calculated
            features_per_symbol: Whether to create one row per symbol (aggregated) or per observation
            
        Returns:
            Feature matrix for clustering
        """
        if features_per_symbol:
            # Aggregate features per symbol
            feature_columns = [col for col in df.columns if col not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']]
            
            feature_matrix = df.groupby('symbol')[feature_columns].agg([
                'mean', 'std', 'min', 'max', 'last'
            ]).round(6)
            
            # Flatten column names
            feature_matrix.columns = [f"{col}_{agg}" for col, agg in feature_matrix.columns]
            feature_matrix = feature_matrix.reset_index()
            
        else:
            # Use all observations (time series clustering)
            # Select numeric features only
            exclude_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            feature_matrix = df[['symbol', 'date'] + feature_columns].copy()
        
        self.logger.info(f"Created feature matrix with shape: {feature_matrix.shape}")
        return feature_matrix
    
    def prepare_features_for_clustering(self, feature_matrix: pd.DataFrame, 
                                      handle_missing: str = 'mean') -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Prepare features for clustering (scaling, handling missing values).
        
        Args:
            feature_matrix: Feature matrix
            handle_missing: How to handle missing values ('mean', 'median', 'zero', 'drop')
            
        Returns:
            Tuple of (scaled_features, scaler_object)
        """
        # Remove non-numeric columns for scaling
        non_numeric_cols = ['symbol', 'date']
        numeric_cols = [col for col in feature_matrix.columns if col not in non_numeric_cols]
        
        features_numeric = feature_matrix[numeric_cols].copy()
        
        # Handle missing values
        if handle_missing == 'mean':
            features_numeric = features_numeric.fillna(features_numeric.mean())
        elif handle_missing == 'median':
            features_numeric = features_numeric.fillna(features_numeric.median())
        elif handle_missing == 'zero':
            features_numeric = features_numeric.fillna(0)
        elif handle_missing == 'drop':
            features_numeric = features_numeric.dropna()
        
        # Handle infinite values
        features_numeric = features_numeric.replace([np.inf, -np.inf], 0)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_numeric)
        
        # Convert back to DataFrame
        scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols)
        
        # Add back non-numeric columns
        for col in non_numeric_cols:
            if col in feature_matrix.columns:
                scaled_df[col] = feature_matrix[col].values
        
        self.logger.info(f"Features prepared: {scaled_features.shape[1]} features, {scaled_features.shape[0]} samples")
        
        return scaled_df, scaler