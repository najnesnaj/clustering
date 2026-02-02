"""
Basic tests for the data fetcher module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_fetcher import DataFetcher


class TestDataFetcher(unittest.TestCase):
    """Test cases for DataFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_fetcher = DataFetcher(cache_dir="test_cache")
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_validate_symbol_valid(self, mock_ticker):
        """Test validation of valid symbol."""
        mock_ticker.return_value.info = {'regularMarketPrice': 100.0}
        
        is_valid, error = self.data_fetcher.validate_symbol('AAPL')
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_validate_symbol_invalid(self, mock_ticker):
        """Test validation of invalid symbol."""
        mock_ticker.side_effect = Exception("Symbol not found")
        
        is_valid, error = self.data_fetcher.validate_symbol('INVALID')
        
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_single_stock_data_success(self, mock_ticker):
        """Test successful data fetching for single stock."""
        # Mock historical data
        mock_hist = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1100]
        })
        mock_hist.index = pd.date_range('2023-01-01', periods=2)
        
        mock_ticker.return_value.history.return_value = mock_hist
        
        result = self.data_fetcher.fetch_single_stock_data('AAPL')
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertIn('symbol', result.columns)
        self.assertEqual(result['symbol'].iloc[0], 'AAPL')
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_single_stock_data_empty(self, mock_ticker):
        """Test data fetching when no data available."""
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        
        result = self.data_fetcher.fetch_single_stock_data('INVALID')
        
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()