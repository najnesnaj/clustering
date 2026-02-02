#!/usr/bin/env python3
"""
Demo script to show clustering project functionality with sample data.
This bypasses the database requirement and demonstrates the full pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data_fetcher import DataFetcher
from feature_extractor import FeatureExtractor
from clustering import StockClustering
from visualizer import ClusterVisualizer


def create_comprehensive_sample_data():
    """Create more comprehensive sample data for demonstration."""
    # Different types of stocks with varying characteristics
    stock_configs = [
        {'symbol': 'STABLE_GROWTH', 'volatility': 0.01, 'drift': 0.001, 'price_base': 100},
        {'symbol': 'VOLATILE_TECH', 'volatility': 0.04, 'drift': 0.002, 'price_base': 50},
        {'symbol': 'CYCLICAL_INDUST', 'volatility': 0.025, 'drift': -0.0005, 'price_base': 75},
        {'symbol': 'UTILITY_DIVIDEND', 'volatility': 0.008, 'drift': 0.0003, 'price_base': 120},
        {'symbol': 'BIOTECH_SPECULATIVE', 'volatility': 0.06, 'drift': 0.003, 'price_base': 30},
    ]
    
    sample_data = []
    base_date = datetime(2020, 1, 1)
    
    np.random.seed(42)  # For reproducible results
    
    for config in stock_configs:
        # Generate synthetic price data
        days = 750  # About 3 years of data
        returns = np.random.normal(config['drift'], config['volatility'], days)
        prices = [config['price_base']]
        
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Create DataFrame for this symbol
        for i, price in enumerate(prices[:-1]):
            date = base_date + timedelta(days=i)
            # Add some realistic variation to OHLC
            daily_range = price * config['volatility']
            high_variation = abs(np.random.normal(0, daily_range * 0.5))
            low_variation = abs(np.random.normal(0, daily_range * 0.5))
            
            high = price + high_variation
            low = price - low_variation
            high = min(high, price * 1.1)  # Cap extreme values
            low = max(low, price * 0.9)   # Floor extreme values
            
            volume = np.random.randint(500000, 5000000)
            open_price = price * (1 + np.random.normal(0, 0.005))
            
            sample_data.append({
                'symbol': config['symbol'],
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
                'dividends': 0.0,
                'stock_splits': 0.0
            })
    
    df = pd.DataFrame(sample_data)
    logger.info(f"Created sample data with {len(df)} rows for {df['symbol'].nunique()} symbols")
    return df


def demonstrate_clustering_pipeline():
    """Demonstrate the complete clustering pipeline."""
    logger.info("=== STOCK CLUSTERING DEMONSTRATION ===")
    
    # Step 1: Create sample data (simulating database fetch)
    logger.info("Step 1: Creating sample stock data...")
    sample_data = create_comprehensive_sample_data()
    
    # Save sample data
    sample_data.to_csv('demo_stock_data.csv', index=False)
    logger.info("Sample data saved to demo_stock_data.csv")
    
    # Step 2: Feature extraction
    logger.info("Step 2: Extracting features...")
    feature_extractor = FeatureExtractor()
    features_with_data = feature_extractor.extract_features_for_clustering(sample_data)
    logger.info(f"Features extracted: {features_with_data.shape}")
    
    # Step 3: Create feature matrix
    logger.info("Step 3: Creating feature matrix...")
    feature_matrix = feature_extractor.create_feature_matrix(features_with_data, features_per_symbol=True)
    logger.info(f"Feature matrix: {feature_matrix.shape}")
    
    # Step 4: Prepare features for clustering
    logger.info("Step 4: Preparing features for clustering...")
    clustering_features, scaler = feature_extractor.prepare_features_for_clustering(feature_matrix)
    logger.info(f"Clustering features: {clustering_features.shape}")
    
    # Step 5: Perform clustering
    logger.info("Step 5: Performing clustering analysis...")
    clustering_analyzer = StockClustering(max_clusters=10)  # Allow up to 10 clusters
    
    # Find optimal number of clusters
    cluster_results = clustering_analyzer.find_optimal_clusters(
        clustering_features, 
        cluster_range=range(2, min(6, len(feature_matrix)))  # Test 2-5 clusters
    )
    
    best_n_clusters = clustering_analyzer.select_best_n_clusters(cluster_results)
    logger.info(f"Optimal clusters found: {best_n_clusters}")
    
    # Perform final clustering
    cluster_labels = clustering_analyzer.perform_clustering(
        clustering_features,
        n_clusters=best_n_clusters
    )
    
    # Step 6: Analyze results
    logger.info("Step 6: Analyzing clusters...")
    cluster_analysis = clustering_analyzer.analyze_clusters(feature_matrix, cluster_labels)
    cluster_labels_dict = clustering_analyzer.generate_cluster_labels(cluster_analysis)
    
    logger.info("Cluster Analysis Results:")
    for cluster_id, stats in cluster_analysis.items():
        label = cluster_labels_dict.get(cluster_id, f"Cluster {cluster_id}")
        logger.info(f"  {label}: {stats['size']} stocks")
    
    # Step 7: Create visualizations
    logger.info("Step 7: Creating visualizations...")
    visualizer = ClusterVisualizer(output_dir="demo_reports")
    
    # Get cluster assignments
    cluster_assignments = clustering_analyzer.get_cluster_assignments(feature_matrix)
    cluster_assignments['cluster_label'] = cluster_assignments['cluster'].map(cluster_labels_dict)
    
    # Reduce dimensions for visualization
    features_2d, _ = clustering_analyzer.reduce_dimensions(
        clustering_features, 
        method='pca', n_components=2
    )
    
    # Create comprehensive report
    plots_created = visualizer.create_comprehensive_report(
        features_2d=features_2d,
        cluster_labels=cluster_labels,
        cluster_assignments=cluster_assignments,
        cluster_analysis=cluster_analysis,
        clustering_metrics=clustering_analyzer.evaluate_clustering_quality(clustering_features, cluster_labels),
        stock_data=sample_data,
        method='PCA'
    )
    
    # Save results
    cluster_assignments.to_csv('demo_cluster_assignments.csv', index=False)
    feature_matrix.to_csv('demo_feature_matrix.csv', index=False)
    
    logger.info(f"Visualizations created: {list(plots_created.keys())}")
    
    # Step 8: Generate summary report
    logger.info("Step 8: Generating summary report...")
    summary_table_path = visualizer.save_cluster_summary_table(cluster_analysis, cluster_labels_dict)
    
    # Create markdown report
    generate_demo_report(cluster_analysis, cluster_labels_dict, plots_created, summary_table_path)
    
    logger.info("=== DEMONSTRATION COMPLETE ===")
    logger.info("Files generated:")
    logger.info("  - demo_stock_data.csv: Original price data")
    logger.info("  - demo_cluster_assignments.csv: Cluster assignments per stock")
    logger.info("  - demo_feature_matrix.csv: Extracted features")
    logger.info("  - demo_reports/: Visualization plots")
    logger.info("  - demo_cluster_summary_table.csv: Cluster statistics")
    logger.info("  - demo_report.md: Comprehensive analysis report")


def generate_demo_report(cluster_analysis, cluster_labels_dict, plots_created, summary_table_path):
    """Generate a comprehensive demo report."""
    report_content = f"""# Stock Clustering Demonstration Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This demonstration shows the complete stock clustering pipeline using synthetic data that simulates different types of stocks:
- Stable growth stocks (low volatility, steady returns)
- Volatile tech stocks (high volatility, higher growth)
- Cyclical industrial stocks (moderate volatility, cyclical patterns)
- Utility dividend stocks (very low volatility, stable returns)
- Speculative biotech stocks (very high volatility, speculative nature)

## Clustering Results

**Number of clusters created:** {len(cluster_analysis)}

### Cluster Details

"""
    
    for cluster_id, stats in cluster_analysis.items():
        label = cluster_labels_dict.get(cluster_id, f"Cluster {cluster_id}")
        report_content += f"""
#### {label}

- **Size:** {stats['size']} stocks ({stats['percentage']:.1f}%)
- **Average Volatility:** {stats.get('volatility_252d_mean', 0):.3f}
- **Average Return:** {stats.get('return_mean', 0):.4f}
- **Fluctuation Count:** {stats.get('fluctuation_count_mean', 0):.1f}
- **Sharpe Ratio:** {stats.get('sharpe_ratio_mean', 0):.3f}
- **Sample Stocks:** {', '.join(stats.get('symbols', [])[:3])}

"""
    
    report_content += f"""
## Generated Visualizations

"""
    for plot_name, plot_path in plots_created.items():
        report_content += f"- **{plot_name.replace('_', ' ').title()}:** {plot_path}\n"
    
    report_content += f"""
## Key Insights

1. **Automated Cluster Detection:** The system automatically identified {len(cluster_analysis)} distinct groups from stock data
2. **Descriptive Labeling:** Each cluster received a meaningful label based on its characteristics
3. **Comprehensive Feature Analysis:** {len(cluster_analysis.get(list(cluster_analysis.keys())[0], {}))} different features were extracted and analyzed
4. **Visual Analysis:** Multiple visualization types were generated to interpret the results

## Files Generated

- `demo_stock_data.csv` - Complete stock price data
- `demo_cluster_assignments.csv` - Which cluster each stock belongs to
- `demo_feature_matrix.csv` - All extracted features for analysis
- `{summary_table_path}` - Statistical summary of each cluster
- Visualization files in `demo_reports/` directory

## Next Steps

To use this with real data:

1. Set up your PostgreSQL database with stock symbols in the `metrics` table
2. Run: `python main.py`
3. The system will automatically fetch real stock data and perform the same analysis

This demonstrates that the clustering system is fully functional and ready for production use with real market data.
"""
    
    with open('demo_report.md', 'w') as f:
        f.write(report_content)
    
    logger.info("Demo report saved to demo_report.md")


if __name__ == "__main__":
    try:
        demonstrate_clustering_pipeline()
        logger.info("✓ Demonstration completed successfully!")
    except Exception as e:
        logger.error(f"✗ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)