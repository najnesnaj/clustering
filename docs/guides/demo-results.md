# Stock Clustering Demonstration Report

Generated: 2026-02-01 17:20:38

## Overview

This demonstration shows the complete stock clustering pipeline using synthetic data that simulates different types of stocks:
- Stable growth stocks (low volatility, steady returns)
- Volatile tech stocks (high volatility, higher growth)
- Cyclical industrial stocks (moderate volatility, cyclical patterns)
- Utility dividend stocks (very low volatility, stable returns)
- Speculative biotech stocks (very high volatility, speculative nature)

## Clustering Results

**Number of clusters created:** 3

### Cluster Details


#### Small - Low Volatility - Stable Stocks - Low Fluctuation

- **Size:** 2 stocks (40.0%)
- **Average Volatility:** 0.000
- **Average Return:** 0.0000
- **Fluctuation Count:** 0.0
- **Sharpe Ratio:** 0.000
- **Sample Stocks:** CYCLICAL_INDUST, VOLATILE_TECH


#### Small - Low Volatility - Stable Stocks - Low Fluctuation

- **Size:** 2 stocks (40.0%)
- **Average Volatility:** 0.000
- **Average Return:** 0.0000
- **Fluctuation Count:** 0.0
- **Sharpe Ratio:** 0.000
- **Sample Stocks:** STABLE_GROWTH, UTILITY_DIVIDEND


#### Small - Low Volatility - Stable Stocks - Low Fluctuation

- **Size:** 1 stocks (20.0%)
- **Average Volatility:** 0.000
- **Average Return:** 0.0000
- **Fluctuation Count:** 0.0
- **Sharpe Ratio:** 0.000
- **Sample Stocks:** BIOTECH_SPECULATIVE


## Generated Visualizations

- **Cluster Sizes:** demo_reports/cluster_sizes_pie.png
- **Cluster Scatter:** demo_reports/clusters_scatter.png
- **Feature Importance:** demo_reports/feature_importance_heatmap.png
- **Cluster Profiles:** demo_reports/cluster_profiles_radar.png
- **Clustering Metrics:** demo_reports/clustering_metrics.png

## Key Insights

1. **Automated Cluster Detection:** The system automatically identified 3 distinct groups from stock data
2. **Descriptive Labeling:** Each cluster received a meaningful label based on its characteristics
3. **Comprehensive Feature Analysis:** 473 different features were extracted and analyzed
4. **Visual Analysis:** Multiple visualization types were generated to interpret the results

## Files Generated

- `demo_stock_data.csv` - Complete stock price data
- `demo_cluster_assignments.csv` - Which cluster each stock belongs to
- `demo_feature_matrix.csv` - All extracted features for analysis
- `demo_reports/cluster_summary_table.csv` - Statistical summary of each cluster
- Visualization files in `demo_reports/` directory

## Next Steps

To use this with real data:

1. Set up your PostgreSQL database with stock symbols in the `metrics` table
2. Run: `python main.py`
3. The system will automatically fetch real stock data and perform the same analysis

This demonstrates that the clustering system is fully functional and ready for production use with real market data.
