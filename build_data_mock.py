#!/usr/bin/env python3

import logging
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock stock symbols for demo
STOCK_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'NVDA', 'JPM', 'BAC', 'JNJ', 'PFE',
    'WMT', 'HD', 'MCD', 'NKE', 'DIS'
]

class SimpleDatabaseManager:
    """Simplified database manager for demo."""
    
    def __init__(self, db_path: str = "data/clustering.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
    def create_database_schema(self):
        """Create database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create stock_data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    dividends REAL,
                    stock_splits REAL,
                    PRIMARY KEY (symbol, date)
                )
            ''')
            
            # Create clusters table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS clusters (
                    symbol TEXT PRIMARY KEY,
                    cluster_id INTEGER,
                    cluster_label TEXT,
                    confidence_score REAL
                )
            ''')
            
            # Create cluster_info table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cluster_info (
                    cluster_id INTEGER PRIMARY KEY,
                    cluster_label TEXT,
                    size INTEGER,
                    percentage REAL,
                    avg_volatility REAL,
                    avg_return REAL,
                    description TEXT
                )
            ''')
            
            # Create metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            
            conn.commit()
    
    def insert_stock_data(self, df: pd.DataFrame):
        """Insert stock data."""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql('stock_data', conn, if_exists='replace', index=False)
            logger.info(f"Inserted {len(df)} rows of stock data")
    
    def insert_cluster_assignments(self, clusters_df: pd.DataFrame):
        """Insert cluster assignments."""
        with sqlite3.connect(self.db_path) as conn:
            clusters_df.to_sql('clusters', conn, if_exists='replace', index=False)
            logger.info(f"Inserted cluster assignments for {len(clusters_df)} symbols")
    
    def insert_cluster_info(self, cluster_info_dict: dict):
        """Insert cluster information."""
        cluster_info_list = []
        for cluster_id, info in cluster_info_dict.items():
            cluster_info_list.append({
                'cluster_id': cluster_id,
                'cluster_label': info.get('label', f'Cluster {cluster_id}'),
                'size': info.get('size', 0),
                'percentage': info.get('percentage', 0),
                'avg_volatility': info.get('avg_volatility', 0),
                'avg_return': info.get('avg_return', 0),
                'description': info.get('description', '')
            })
        
        if cluster_info_list:
            df = pd.DataFrame(cluster_info_list)
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql('cluster_info', conn, if_exists='replace', index=False)
                logger.info(f"Inserted information for {len(cluster_info_list)} clusters")
    
    def insert_metadata(self, metadata_dict: dict):
        """Insert metadata."""
        with sqlite3.connect(self.db_path) as conn:
            for key, value in metadata_dict.items():
                conn.execute('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)', 
                           (key, str(value)))
            logger.info("Metadata updated successfully")

def generate_mock_stock_data():
    """Generate realistic mock stock data."""
    logger.info(f"Generating mock data for {len(STOCK_SYMBOLS)} symbols...")
    
    all_data = []
    
    # Base date range
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 1, 1)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    for symbol in STOCK_SYMBOLS:
        # Generate realistic stock price data
        np.random.seed(hash(symbol) % 2**32)  # Deterministic but different per symbol
        
        # Starting price based on symbol
        base_price = 50 + hash(symbol) % 150
        
        # Generate returns with some trend
        trend = np.random.normal(0.0005, 0.001, len(date_range))  # Slight upward trend
        volatility = np.random.uniform(0.01, 0.03)  # Different volatility per symbol
        
        daily_returns = np.random.normal(trend, volatility, len(date_range))
        prices = [base_price]
        
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = prices[1:]  # Remove initial price
        
        # Create OHLC data
        for i, date in enumerate(date_range):
            price = prices[i]
            
            # Simple OHLC generation
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = low + (high - low) * np.random.random()
            close = price
            
            # Volume
            volume = int(np.random.normal(1000000, 500000))
            
            all_data.append({
                'symbol': symbol,
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': max(volume, 0),
                'dividends': 0,
                'stock_splits': 0
            })
    
    df = pd.DataFrame(all_data)
    logger.info(f"Generated {len(df)} records for {len(STOCK_SYMBOLS)} symbols")
    return df

def extract_simple_features(df: pd.DataFrame):
    """Extract simple features for clustering."""
    logger.info("Extracting simple features...")
    
    features_list = []
    
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].sort_values('date')
        
        if len(symbol_data) < 100:
            continue
        
        # Calculate daily returns
        daily_returns = symbol_data['close'].pct_change().dropna()
        
        # Simple features
        features = {
            'symbol': symbol,
            'avg_return': daily_returns.mean(),
            'volatility': daily_returns.std() * np.sqrt(252),  # Annualized volatility
            'total_return': (symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0] - 1) * 100,
            'max_drawdown': ((symbol_data['close'].cummax() - symbol_data['close']) / symbol_data['close'].cummax()).max() * 100,
            'sharpe_ratio': daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0,
            'avg_volume': symbol_data['volume'].mean() if 'volume' in symbol_data.columns else 0
        }
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def perform_clustering(feature_df: pd.DataFrame):
    """Perform simple clustering."""
    logger.info("Performing clustering...")
    
    # Get numeric features
    numeric_cols = ['avg_return', 'volatility', 'total_return', 'max_drawdown', 'sharpe_ratio', 'avg_volume']
    numeric_features = feature_df[numeric_cols].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    
    # Find optimal number of clusters
    silhouette_scores = []
    cluster_range = range(2, min(6, len(numeric_features)))  # 2-5 clusters
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, cluster_labels)
        silhouette_scores.append(score)
        logger.info(f"Clusters: {n_clusters}, Silhouette: {score:.3f}")
    
    # Choose best number of clusters
    best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
    logger.info(f"Best number of clusters: {best_n_clusters}")
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    return cluster_labels, best_n_clusters

def main():
    """Main function to build the data."""
    logger.info("Starting simple stock clustering demo data build...")
    
    try:
        # Initialize database manager
        db_manager = SimpleDatabaseManager()
        db_manager.create_database_schema()
        
        # Step 1: Generate mock stock data
        stock_data = generate_mock_stock_data()
        db_manager.insert_stock_data(stock_data)
        
        # Step 2: Extract features
        feature_df = extract_simple_features(stock_data)
        logger.info(f"Extracted features for {len(feature_df)} symbols")
        
        # Step 3: Perform clustering
        cluster_labels, n_clusters = perform_clustering(feature_df)
        
        # Step 4: Create cluster assignments
        cluster_assignments = pd.DataFrame({
            'symbol': feature_df['symbol'],
            'cluster_id': cluster_labels,
            'cluster_label': [f'Cluster {label}' for label in cluster_labels],
            'confidence_score': 0.85
        })
        db_manager.insert_cluster_assignments(cluster_assignments)
        
        # Step 5: Create cluster info
        cluster_info_dict = {}
        for cluster_id in range(n_clusters):
            cluster_symbols = cluster_assignments[cluster_assignments['cluster_id'] == cluster_id]['symbol'].tolist()
            cluster_features = feature_df[feature_df['symbol'].isin(cluster_symbols)]
            
            cluster_info_dict[cluster_id] = {
                'label': f'Cluster {cluster_id}',
                'size': len(cluster_symbols),
                'percentage': len(cluster_symbols) / len(feature_df) * 100,
                'avg_volatility': cluster_features['volatility'].mean(),
                'avg_return': cluster_features['avg_return'].mean() * 100,
                'description': f"Cluster {cluster_id} with {len(cluster_symbols)} stocks"
            }
        
        db_manager.insert_cluster_info(cluster_info_dict)
        
        # Step 6: Insert metadata
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'total_symbols': len(stock_data['symbol'].unique()),
            'total_records': len(stock_data),
            'date_range_start': stock_data['date'].min().isoformat(),
            'date_range_end': stock_data['date'].max().isoformat(),
            'num_clusters': n_clusters,
            'feature_count': 6  # We used 6 simple features
        }
        db_manager.insert_metadata(metadata)
        
        logger.info("Mock clustering completed successfully!")
        logger.info(f"Created {n_clusters} clusters")
        logger.info(f"Data saved to database: {db_manager.db_path}")
        
        return True
            
    except Exception as e:
        logger.error(f"Error during data build: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)