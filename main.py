#!/usr/bin/env python3
"""
Main execution script for the stock clustering project.

This script orchestrates the entire clustering pipeline:
1. Fetch symbols from PostgreSQL database
2. Download stock price data from Yahoo Finance
3. Extract features for clustering analysis
4. Perform clustering analysis
5. Generate visualizations and reports
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_fetcher import DataFetcher
from feature_extractor import FeatureExtractor
from clustering import StockClustering
from visualizer import ClusterVisualizer
from config.database import get_db_connection
from sqlalchemy import text


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("clustering_analysis.log")
        ]
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Stock Clustering Analysis")
    parser.add_argument("--max-clusters", type=int, default=50, 
                       help="Maximum number of clusters to create")
    parser.add_argument("--algorithm", choices=["kmeans", "hierarchical", "dbscan"], 
                       default="kmeans", help="Clustering algorithm")
    parser.add_argument("--validate-symbols", action="store_true", default=True,
                       help="Validate symbols before downloading")
    parser.add_argument("--period", default="max", 
                       help="Data period to fetch (max, 10y, 5y, etc.)")
    parser.add_argument("--features-per-symbol", action="store_true", default=True,
                       help="Create one feature row per symbol (vs per observation)")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--output-dir", default="reports", 
                       help="Output directory for results")
    parser.add_argument("--cache-dir", default="data/raw", 
                       help="Cache directory for downloaded data")
    parser.add_argument("--use-cache", action="store_true", 
                       help="Use cached data if available")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Stock Clustering Analysis")
    logger.info(f"Parameters: max_clusters={args.max_clusters}, algorithm={args.algorithm}")
    
    try:
        # Step 1: Fetch symbols from database
        logger.info("Step 1: Fetching symbols from PostgreSQL database...")
        db_connection = get_db_connection()
        
        # Test database connection
        if not db_connection.test_connection():
            logger.error("Failed to connect to database")
            return 1
        
        # Check if price_data table exists and has data
        try:
            db = get_db_connection()
            engine = db.connect()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM price_data"))
                count = result.fetchone()[0]
                if count > 0:
                    logger.info(f"Found {count} price records in database, using database data instead of Yahoo Finance")
                    use_database_data = True
                else:
                    logger.info("No price data found in database, fetching from Yahoo Finance")
                    use_database_data = False
        except:
            logger.info("price_data table not found, will fetch from Yahoo Finance")
            use_database_data = False
        
        # Fetch symbols
        symbols = db_connection.fetch_symbols_from_metrics()['symbol'].tolist()
        logger.info(f"Found {len(symbols)} symbols in database")
        
        if not symbols:
            logger.error("No symbols found in database")
            return 1
        
        # Step 2: Fetch stock price data
        logger.info("Step 2: Fetching stock price data...")
        data_fetcher = DataFetcher(cache_dir=args.cache_dir)
        
        if args.use_cache:
            logger.info("Using cached data where available")
        
        if 'use_database_data' in locals() and use_database_data:
            # Fetch from database instead of Yahoo Finance
            logger.info("Fetching stock price data from local database...")
            stock_data_dict = data_fetcher.fetch_price_data_from_database(symbols)
        else:
            # Fetch from Yahoo Finance
            stock_data_dict = data_fetcher.fetch_multiple_stocks_data(
                symbols, 
                period=args.period,
                validate_symbols=args.validate_symbols
            )
        
        if not stock_data_dict:
            logger.error("No stock data could be fetched")
            return 1
        
        # Combine all data
        stock_data = data_fetcher.combine_all_data(stock_data_dict)
        logger.info(f"Combined data contains {len(stock_data)} rows for {stock_data['symbol'].nunique()} symbols")
        
        # Save combined data
        combined_data_path = data_fetcher.save_combined_data(stock_data)
        logger.info(f"Combined data saved to {combined_data_path}")
        
        # Get data summary
        data_summary = data_fetcher.get_data_summary(stock_data)
        logger.info(f"Data summary: {data_summary}")
        
        # Step 3: Extract features
        logger.info("Step 3: Extracting features for clustering analysis...")
        feature_extractor = FeatureExtractor()
        
        # Extract all features
        features_with_data = feature_extractor.extract_features_for_clustering(stock_data)
        
        # Create feature matrix
        if args.features_per_symbol:
            feature_matrix = feature_extractor.create_feature_matrix(features_with_data, features_per_symbol=True)
        else:
            feature_matrix = feature_extractor.create_feature_matrix(features_with_data, features_per_symbol=False)
        
        # Prepare features for clustering
        clustering_features, scaler = feature_extractor.prepare_features_for_clustering(feature_matrix)
        logger.info(f"Feature matrix created with shape: {clustering_features.shape}")
        
        # Step 4: Perform clustering
        logger.info("Step 4: Performing clustering analysis...")
        clustering_analyzer = StockClustering(max_clusters=args.max_clusters)
        
# Perform clustering
        if args.features_per_symbol:
            # Standard clustering on aggregated features
            cluster_labels = clustering_analyzer.perform_clustering(
                clustering_features,
                algorithm=args.algorithm
            )
        else:
            # Time series clustering
            cluster_labels = clustering_analyzer.perform_time_series_clustering(
                clustering_features
            )
        
        # Analyze clusters
        cluster_analysis = clustering_analyzer.analyze_clusters(feature_matrix, cluster_labels)
        
        # Generate descriptive labels
        cluster_labels_descriptive = clustering_analyzer.generate_cluster_labels(cluster_analysis)
        
        # Evaluate clustering quality
        clustering_metrics = clustering_analyzer.evaluate_clustering_quality(clustering_features, cluster_labels)
        
        # Get cluster assignments
        cluster_assignments = clustering_analyzer.get_cluster_assignments(feature_matrix)
        
        # Add descriptive labels to assignments
        cluster_assignments['cluster_label'] = cluster_assignments['cluster'].map(cluster_labels_descriptive)
        
        logger.info(f"Clustering completed: {len(cluster_analysis)} clusters created")
        
        # Step 5: Create visualizations
        logger.info("Step 5: Creating visualizations and reports...")
        visualizer = ClusterVisualizer(output_dir=args.output_dir)
        
        # Reduce dimensions for visualization
        features_2d, _ = clustering_analyzer.reduce_dimensions(
            clustering_features, method='pca', n_components=2
        )
        
        # Create comprehensive report
        plots_created = visualizer.create_comprehensive_report(
            features_2d=features_2d,
            cluster_labels=cluster_labels,
            cluster_assignments=cluster_assignments,
            cluster_analysis=cluster_analysis,
            clustering_metrics=clustering_metrics,
            stock_data=stock_data if not args.features_per_symbol else None,
            method='PCA'
        )
        
        # Save cluster summary table
        summary_table_path = visualizer.save_cluster_summary_table(
            cluster_analysis, cluster_labels_descriptive
        )
        
        # Save cluster assignments
        assignments_path = Path(args.output_dir) / "cluster_assignments.csv"
        cluster_assignments.to_csv(assignments_path, index=False)
        
        # Save feature matrix
        features_path = Path(args.output_dir) / "feature_matrix.csv"
        feature_matrix.to_csv(features_path, index=False)
        
        # Step 6: Generate summary report
        logger.info("Step 6: Generating final report...")
        generate_text_report(
            output_dir=args.output_dir,
            data_summary=data_summary,
            cluster_analysis=cluster_analysis,
            cluster_labels=cluster_labels_descriptive,
            clustering_metrics=clustering_metrics,
            plots_created=plots_created,
            parameters=args
        )
        
        logger.info("✓ Stock clustering analysis completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


def generate_text_report(output_dir: str, data_summary: dict, cluster_analysis: dict,
                         cluster_labels: dict, clustering_metrics: dict,
                         plots_created: dict, parameters: argparse.Namespace):
    """Generate a comprehensive text report."""
    report_path = Path(output_dir) / "analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Stock Clustering Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Parameters
        f.write("## Analysis Parameters\n\n")
        f.write(f"- Maximum clusters: {parameters.max_clusters}\n")
        f.write(f"- Clustering algorithm: {parameters.algorithm}\n")
        f.write(f"- Data period: {parameters.period}\n")
        f.write(f"- Features per symbol: {parameters.features_per_symbol}\n")
        f.write(f"- Symbol validation: {parameters.validate_symbols}\n\n")
        
        # Data Summary
        f.write("## Data Summary\n\n")
        f.write(f"- Total symbols: {data_summary['unique_symbols']}\n")
        f.write(f"- Total records: {data_summary['total_records']:,}\n")
        f.write(f"- Date range: {data_summary['date_range']['start']} to {data_summary['date_range']['end']}\n\n")
        
        # Clustering Results
        f.write("## Clustering Results\n\n")
        f.write(f"- Number of clusters: {len(cluster_analysis)}\n")
        f.write("- Cluster sizes:\n")
        for cluster_id, stats in cluster_analysis.items():
            label = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
            f.write(f"  - {label}: {stats['size']} stocks ({stats['percentage']:.1f}%)\n")
        f.write("\n")
        
        # Quality Metrics
        f.write("## Clustering Quality Metrics\n\n")
        for metric, value in clustering_metrics.items():
            f.write(f"- {metric}: {value:.4f}\n")
        f.write("\n")
        
        # Cluster Details
        f.write("## Cluster Details\n\n")
        for cluster_id, stats in cluster_analysis.items():
            label = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
            f.write(f"### {label}\n\n")
            f.write(f"- **Size**: {stats['size']} stocks ({stats['percentage']:.1f}%)\n")
            f.write(f"- **Average Volatility**: {stats.get('volatility_252d_mean', 0):.3f}\n")
            f.write(f"- **Average Return**: {stats.get('return_mean', 0):.4f}\n")
            f.write(f"- **Fluctuation Count**: {stats.get('fluctuation_count_mean', 0):.1f}\n")
            f.write(f"- **Sharpe Ratio**: {stats.get('sharpe_ratio_mean', 0):.3f}\n")
            
            # Sample symbols
            symbols = stats.get('symbols', [])
            if symbols:
                sample_symbols = symbols[:10]  # First 10 symbols
                f.write(f"- **Sample Symbols**: {', '.join(sample_symbols)}\n")
            f.write("\n")
        
        # Visualizations
        f.write("## Generated Visualizations\n\n")
        for plot_name, plot_path in plots_created.items():
            f.write(f"- **{plot_name.replace('_', ' ').title()}**: {plot_path}\n")
        f.write("\n")
        
        # Usage Instructions
        f.write("## Using the Results\n\n")
        f.write("The following files have been generated:\n\n")
        f.write("- `cluster_assignments.csv`: Symbol-to-cluster mapping\n")
        f.write("- `feature_matrix.csv`: Complete feature set used for clustering\n")
        f.write("- `cluster_summary_table.csv`: Summary statistics for each cluster\n")
        f.write("- Various visualization plots (.png format)\n\n")
        f.write("You can use the cluster assignments to:\n")
        f.write("1. Create diversified portfolios by selecting stocks from different clusters\n")
        f.write("2. Identify stocks with similar trading patterns\n")
        f.write("3. Perform sector-like analysis without relying on traditional sector classifications\n")
        f.write("4. Risk management by understanding volatility patterns across clusters\n\n")


if __name__ == "__main__":
    sys.exit(main())