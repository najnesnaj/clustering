"""
Data fetching module for retrieving stock symbols and price data.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.database import get_db_connection
from sqlalchemy import text


class DataFetcher:
    """
    Handles fetching stock symbols from PostgreSQL and price data from Yahoo Finance.
    """
    
    def __init__(self, cache_dir: str = "data/raw", max_workers: int = 10):
        """
        Initialize data fetcher.
        
        Args:
            cache_dir: Directory to cache downloaded data
            max_workers: Maximum number of parallel download threads
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        self.db_connection = get_db_connection()
        
        # Rate limiting for Yahoo Finance API
        self.api_delay = 0.1  # seconds between requests
        
    def fetch_symbols_from_database(self) -> List[str]:
        """
        Fetch all stock symbols from the metrics table in PostgreSQL.
        
        Returns:
            List of stock symbols
        """
        try:
            symbols_df = self.db_connection.fetch_symbols_from_metrics()
            symbols = symbols_df['symbol'].tolist()
            self.logger.info(f"Fetched {len(symbols)} symbols from database")
            return symbols
        except Exception as e:
            self.logger.error(f"Failed to fetch symbols from database: {e}")
            raise
    
    def fetch_price_data_from_database(self, symbols: List[str], period: str = "max") -> Dict[str, pd.DataFrame]:
        """
        Fetch stock price data from the local price_data database.
        
        Args:
            symbols: List of stock symbols
            period: Data period (not used for database fetching)
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        try:
            db = get_db_connection()
            engine = db.connect()
            
            stock_data_dict = {}
            
            with engine.connect() as conn:
                for symbol in symbols:
                    # Fetch data for this symbol
                    query = text('''
                        SELECT symbol, date, open_price, high_price, low_price, close_price, volume, dividends, stock_splits
                        FROM price_data 
                        WHERE symbol = :symbol 
                        ORDER BY date ASC
                    ''')
                    
                    df = pd.read_sql(query, conn, params={'symbol': symbol})
                    
                    if not df.empty:
                        # Rename columns to match expected format
                        df = df.rename(columns={
                            'open_price': 'open',
                            'high_price': 'high', 
                            'low_price': 'low',
                            'close_price': 'close'
                        })
                        
                        stock_data_dict[symbol] = df
                        self.logger.info(f"Fetched {len(df)} rows of data for {symbol}")
                    else:
                        self.logger.warning(f"No data found for symbol {symbol}")
            
            return stock_data_dict
            
        except Exception as e:
            self.logger.error(f"Failed to fetch price data from database: {e}")
            raise
    
    def validate_symbol(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a symbol is available on Yahoo Finance.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try to get basic info to validate symbol
            info = ticker.info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                return True, None
            elif 'shortName' in info:
                return True, None
            else:
                return False, "No market data available"
        except Exception as e:
            return False, str(e)
    
    def fetch_single_stock_data(self, symbol: str, period: str = "max") -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single stock symbol.
        
        Args:
            symbol: Stock symbol
            period: Data period ("max", "10y", "5y", etc.)
            
        Returns:
            DataFrame with historical price data or None if failed
        """
        try:
            # Check cache first
            cache_file = self.cache_dir / f"{symbol}_max.parquet"
            if cache_file.exists():
                cached_data = pd.read_parquet(cache_file)
                # Check if cache is recent (less than 1 day old)
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age.days < 1:
                    self.logger.debug(f"Using cached data for {symbol}")
                    return cached_data
            
            # Fetch from Yahoo Finance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                self.logger.warning(f"No historical data found for {symbol}")
                return None
            
            # Clean up the data
            hist = hist.reset_index()
            hist['Symbol'] = symbol
            
            # Cache the data
            hist.to_parquet(cache_file)
            
            self.logger.debug(f"Fetched {len(hist)} rows of data for {symbol}")
            return hist
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def fetch_multiple_stocks_data(self, symbols: List[str], period: str = "max", 
                                  validate_symbols: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stock symbols in parallel.
        
        Args:
            symbols: List of stock symbols
            period: Data period ("max", "10y", "5y", etc.)
            validate_symbols: Whether to validate symbols before downloading
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        if validate_symbols:
            self.logger.info("Validating symbols before downloading...")
            valid_symbols = []
            for symbol in tqdm(symbols, desc="Validating symbols"):
                is_valid, _ = self.validate_symbol(symbol)
                if is_valid:
                    valid_symbols.append(symbol)
                else:
                    self.logger.warning(f"Skipping invalid symbol: {symbol}")
        else:
            valid_symbols = symbols
        
        self.logger.info(f"Fetching data for {len(valid_symbols)} symbols...")
        
        results = {}
        failed_symbols = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.fetch_single_stock_data, symbol, period): symbol 
                for symbol in valid_symbols
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(valid_symbols), desc="Downloading data") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result()
                        if data is not None:
                            results[symbol] = data
                        else:
                            failed_symbols.append(symbol)
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")
                        failed_symbols.append(symbol)
                    finally:
                        pbar.update(1)
                    # Rate limiting
                    time.sleep(self.api_delay)
        
        self.logger.info(f"Successfully fetched data for {len(results)} symbols")
        if failed_symbols:
            self.logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")
        
        return results
    
    def combine_all_data(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all stock data into a single DataFrame.
        
        Args:
            stock_data: Dictionary of symbol -> DataFrame
            
        Returns:
            Combined DataFrame with all stock data
        """
        if not stock_data:
            raise ValueError("No stock data to combine")
        
        all_data = []
        for symbol, data in stock_data.items():
            if data is not None and not data.empty:
                all_data.append(data)
        
        if not all_data:
            raise ValueError("No valid stock data to combine")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure proper column names
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Dividends': 'dividends',
            'Stock Splits': 'stock_splits',
            'Symbol': 'symbol'
        }
        
        # Rename columns to lowercase
        combined_df = combined_df.rename(columns=column_mapping)
        
        # Convert date column to datetime
        if 'date' in combined_df.columns:
            combined_df['date'] = pd.to_datetime(combined_df['date'], utc=True)
        
        # Sort by symbol and date
        combined_df = combined_df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        self.logger.info(f"Combined data contains {len(combined_df)} rows for {combined_df['symbol'].nunique()} symbols")
        
        return combined_df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the fetched data.
        
        Args:
            df: Combined stock data DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_records': len(df),
            'unique_symbols': df['symbol'].nunique(),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'symbols_with_data': df['symbol'].unique().tolist(),
            'data_completeness': {}
        }
        
        # Check data completeness per symbol
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            total_days = (symbol_data['date'].max() - symbol_data['date'].min()).days
            available_days = len(symbol_data)
            completeness = available_days / max(total_days, 1) * 100
            summary['data_completeness'][symbol] = {
                'available_days': available_days,
                'total_period_days': total_days,
                'completeness_percentage': round(completeness, 2)
            }
        
        return summary
    
    def save_combined_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save combined data to file.
        
        Args:
            df: Combined stock data DataFrame
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"combined_stock_data_{timestamp}.parquet"
        
        filepath = self.cache_dir / filename
        df.to_parquet(filepath, index=False)
        
        self.logger.info(f"Combined data saved to {filepath}")
        return str(filepath)