#!/usr/bin/env python3
"""
Test script for stock clustering with sample data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data_fetcher import DataFetcher
from feature_extractor import FeatureExtractor
from clustering import StockClustering
from visualizer import ClusterVisualizer


def create_sample_data():
    """Create sample stock data for testing."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    sample_data = []
    base_date = datetime(2020, 1, 1)
    
    np.random.seed(42)  # For reproducible results
    
    for symbol in symbols:
        # Generate synthetic price data
        days = 500
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns
        prices = [100.0]
        
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Create DataFrame for this symbol
        for i, price in enumerate(prices[:-1]):
            date = base_date + timedelta(days=i)
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(1000000, 10000000)
            
            sample_data.append({
                'symbol': symbol,
                'date': date,
                'open': price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
                'dividends': 0.0,
                'stock_splits': 0.0
            })
    
    return pd.DataFrame(sample_data)


def test_data_fetcher():
    """Test data fetcher with sample data."""
    logger.info("Testing data fetcher...")
    
    # Create sample data
    sample_data = create_sample_data()
    logger.info(f"Created sample data with {len(sample_data)} rows for {sample_data['symbol'].nunique()} symbols")
    
    # Save sample data as if it was fetched
    data_fetcher = DataFetcher(cache_dir="test_cache")
    cache_file = Path("test_cache") / "sample_data.parquet"
    cache_file.parent.mkdir(exist_ok=True)
    sample_data.to_parquet(cache_file)
    
    return sample_data


def test_feature_extraction(sample_data):
    """Test feature extraction."""
    logger.info("Testing feature extraction...")
    
    feature_extractor = FeatureExtractor()
    
    # Extract features
    features_with_data = feature_extractor.extract_features_for_clustering(sample_data)
    logger.info(f"Features extracted with shape: {features_with_data.shape}")
    
    # Create feature matrix
    feature_matrix = feature_extractor.create_feature_matrix(features_with_data, features_per_symbol=True)
    logger.info(f"Feature matrix created with shape: {feature_matrix.shape}")
    
    # Prepare features for clustering
    clustering_features, scaler = feature_extractor.prepare_features_for_clustering(feature_matrix)
    logger.info(f"Clustering features prepared with shape: {clustering_features.shape}")
    
    return feature_matrix, clustering_features, scaler


def test_clustering(feature_matrix, clustering_features):
    """Test clustering."""
    logger.info("Testing clustering...")
    
    clustering_analyzer = StockClustering(max_clusters=5)
    
    # Perform clustering
    cluster_labels = clustering_analyzer.perform_clustering(
        clustering_features,
        algorithm='kmeans',
        n_clusters=3,  # Force 3 clusters for testing
        auto_optimize=False
    )
    
    logger.info(f"Clustering completed with labels: {np.unique(cluster_labels, return_counts=True)}")
    
    # Analyze clusters
    cluster_analysis = clustering_analyzer.analyze_clusters(feature_matrix, cluster_labels)
    logger.info(f"Cluster analysis completed for {len(cluster_analysis)} clusters")
    
    # Generate labels
    cluster_labels_dict = clustering_analyzer.generate_cluster_labels(cluster_analysis)
    logger.info("Descriptive labels generated")
    
    return clustering_analyzer, cluster_labels, cluster_analysis, cluster_labels_dict


def test_visualization(clustering_analyzer, feature_matrix, cluster_labels, cluster_analysis, cluster_labels_dict, sample_data, clustering_features):
    """Test visualization."""
    logger.info("Testing visualization...")
    
    visualizer = ClusterVisualizer(output_dir="test_reports")
    
    # Get cluster assignments
    cluster_assignments = clustering_analyzer.get_cluster_assignments(feature_matrix)
    cluster_assignments['cluster_label'] = cluster_assignments['cluster'].map(cluster_labels_dict)
    
    # Reduce dimensions
    features_2d, _ = clustering_analyzer.reduce_dimensions(
        clustering_features, 
        method='pca', n_components=2
    )
    
    # Create visualizations
    plots_created = visualizer.create_comprehensive_report(
        features_2d=features_2d,
        cluster_labels=cluster_labels,
        cluster_assignments=cluster_assignments,
        cluster_analysis=cluster_analysis,
        clustering_metrics=clustering_analyzer.evaluate_clustering_quality(
            clustering_features, 
            cluster_labels
        ),
        stock_data=sample_data,
        method='PCA'
    )
    
    logger.info(f"Visualizations created: {list(plots_created.keys())}")
    
    # Save summary table
    summary_path = visualizer.save_cluster_summary_table(cluster_analysis, cluster_labels_dict)
    logger.info(f"Summary table saved to: {summary_path}")
    
    return plots_created


def main():
    """Main test function."""
    logger.info("Starting stock clustering test...")
    
    try:
        # Step 1: Test data fetching
        sample_data = test_data_fetcher()
        
        # Step 2: Test feature extraction
        feature_matrix, clustering_features, scaler = test_feature_extraction(sample_data)
        
        # Step 3: Test clustering
        clustering_analyzer, cluster_labels, cluster_analysis, cluster_labels_dict = test_clustering(feature_matrix, clustering_features)
        
        # Step 4: Test visualization
        plots_created = test_visualization(clustering_analyzer, feature_matrix, cluster_labels, cluster_analysis, cluster_labels_dict, sample_data, clustering_features)
        
        # Summary
        logger.info("=== TEST SUMMARY ===")
        logger.info(f"✓ Data: {len(sample_data)} rows for {sample_data['symbol'].nunique()} symbols")
        logger.info(f"✓ Features: {feature_matrix.shape[1]} features extracted")
        logger.info(f"✓ Clusters: {len(cluster_analysis)} clusters created")
        logger.info(f"✓ Visualizations: {len(plots_created)} plots generated")
        
        logger.info("✓ All tests completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())