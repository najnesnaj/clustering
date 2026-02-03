#!/usr/bin/env python3

import logging
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add current directory to path for imports
sys.path.append('/app')

# Import our modules
from src.feature_extractor import FeatureExtractor
from database_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simplified stock list (20 stocks for faster demo)
STOCK_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'NVDA', 'JPM', 'BAC', 'JNJ', 'PFE',
    'WMT', 'HD', 'MCD', 'NKE', 'DIS',
    'SPY', 'QQQ', 'VTI', 'VOO'
]

def fetch_stock_data():
    """Fetch stock data for simplified symbol list."""
    logger.info(f"Fetching data for {len(STOCK_SYMBOLS)} symbols...")
    
    all_data = []
    failed_symbols = []
    
    for symbol in tqdm(STOCK_SYMBOLS, desc="Downloading stock data"):
        try:
            ticker = yf.Ticker(symbol)
            # Get 10 years of data instead of 20 for faster build
            hist = ticker.history(start="2014-01-01", end="2024-01-01")
            
            if hist.empty:
                logger.warning(f"No data found for {symbol}")
                failed_symbols.append(symbol)
                continue
            
            # Reset index to get Date as a column
            hist = hist.reset_index()
            
            # Add symbol column
            hist['symbol'] = symbol
            
            # Rename columns to match our expected format
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            }
            hist = hist.rename(columns=column_mapping)
            
            # Ensure data types are correct
            hist['date'] = pd.to_datetime(hist['date'])
            
            all_data.append(hist)
            
            # Rate limiting to avoid getting blocked
            time.sleep(0.2)
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            failed_symbols.append(symbol)
            continue
    
    if not all_data:
        raise ValueError("No data was successfully fetched")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Successfully fetched data for {len(all_data)} symbols")
    logger.info(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")
    logger.info(f"Total records: {len(combined_df)}")
    
    return combined_df

def simple_clustering_analysis(feature_matrix, prepared_features):
    """Simple clustering analysis without complex dependencies."""
    
    # Get numeric features for clustering
    numeric_features = prepared_features.select_dtypes(include=[np.number])
    
    # Handle any infinite or NaN values
    numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Find optimal number of clusters (simple approach)
    silhouette_scores = []
    cluster_range = range(3, min(8, len(numeric_features)))  # Limit to 7 clusters for demo
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(numeric_features)
        score = silhouette_score(numeric_features, cluster_labels)
        silhouette_scores.append(score)
    
    # Choose best number of clusters
    best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(numeric_features)
    
    # Create cluster analysis
    cluster_analysis = {}
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_symbols = feature_matrix.loc[cluster_mask, 'symbol'].tolist()
        
        # Simple stats - use any available numeric features
        numeric_cols = numeric_features.columns.tolist()
        avg_values = numeric_features.loc[cluster_mask].mean()
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_symbols),
            'percentage': len(cluster_symbols) / len(feature_matrix) * 100,
            'symbols': cluster_symbols,
            'volatility_252d_mean': avg_values.get('volatility_252d_mean', 0.15),
            'return_mean': avg_values.get('return_mean', 0.001)
        }
    
    # Generate simple cluster labels
    cluster_labels_dict = {}
    for cluster_id, analysis in cluster_analysis.items():
        size_desc = "Large" if analysis['size'] > 4 else "Medium" if analysis['size'] > 2 else "Small"
        vol_desc = "High Volatility" if analysis.get('volatility_252d_mean', 0) > 0.2 else "Low Volatility"
        cluster_labels_dict[cluster_id] = f"{size_desc} - {vol_desc} Stocks"
    
    return cluster_labels, cluster_analysis, cluster_labels_dict

def process_data_and_cluster(stock_data):
    """Process stock data, extract features, and perform clustering."""
    logger.info("Starting data processing and clustering...")
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    db_manager = DatabaseManager()
    
    # Create database schema
    db_manager.create_database_schema()
    
    # Insert stock data
    db_manager.insert_stock_data(stock_data)
    
    # Extract features
    logger.info("Extracting features...")
    features_df = feature_extractor.extract_features_for_clustering(stock_data)
    
    # Create feature matrix for clustering (one row per symbol)
    logger.info("Creating feature matrix...")
    feature_matrix = feature_extractor.create_feature_matrix(features_df, features_per_symbol=True)
    
    # Prepare features for clustering
    logger.info("Preparing features for clustering...")
    prepared_features, scaler = feature_extractor.prepare_features_for_clustering(feature_matrix)
    
    # Perform clustering
    logger.info("Performing clustering...")
    cluster_labels, cluster_analysis, cluster_labels_dict = simple_clustering_analysis(feature_matrix, prepared_features)
    
    # Create cluster assignments DataFrame
    cluster_assignments = pd.DataFrame({
        'symbol': feature_matrix['symbol'],
        'cluster_id': cluster_labels,
        'cluster_label': [cluster_labels_dict.get(label, f'Cluster {label}') for label in cluster_labels],
        'confidence_score': 0.85  # Placeholder confidence score
    })
    
    # Prepare cluster info for database
    cluster_info_dict = {}
    for cluster_id, analysis in cluster_analysis.items():
        cluster_info_dict[cluster_id] = {
            'label': cluster_labels_dict.get(cluster_id, f'Cluster {cluster_id}'),
            'size': analysis['size'],
            'percentage': analysis['percentage'],
            'avg_volatility': analysis.get('volatility_252d_mean', 0),
            'avg_return': analysis.get('return_mean', 0),
            'description': f"Cluster with {analysis['size']} stocks ({analysis['percentage']:.1f}%)"
        }
    
    # Insert cluster data into database
    db_manager.insert_cluster_assignments(cluster_assignments)
    db_manager.insert_cluster_info(cluster_info_dict)
    
    # Insert metadata
    metadata = {
        'last_updated': datetime.now().isoformat(),
        'total_symbols': len(stock_data['symbol'].unique()),
        'total_records': len(stock_data),
        'date_range_start': stock_data['date'].min().isoformat(),
        'date_range_end': stock_data['date'].max().isoformat(),
        'num_clusters': len(np.unique(cluster_labels)),
        'feature_count': len([col for col in prepared_features.columns if col not in ['symbol', 'date']])
    }
    db_manager.insert_metadata(metadata)
    
    logger.info(f"Clustering completed successfully!")
    logger.info(f"Created {len(np.unique(cluster_labels))} clusters")
    logger.info(f"Data saved to database: {db_manager.db_path}")
    
    return True

def main():
    """Main function to build the data."""
    logger.info("Starting stock clustering demo data build...")
    
    try:
        # Step 1: Fetch stock data
        stock_data = fetch_stock_data()
        
        # Step 2: Process data and perform clustering
        success = process_data_and_cluster(stock_data)
        
        if success:
            logger.info("Data build completed successfully!")
            logger.info("The Streamlit app can now be started with: streamlit run app.py")
        else:
            logger.error("Data build failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during data build: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()