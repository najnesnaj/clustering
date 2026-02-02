# Stock Clustering Analysis Project

A comprehensive Python project for clustering stocks based on their price behavior and characteristics using data from PostgreSQL and Yahoo Finance.

## Overview

This project fetches stock symbols from a PostgreSQL database, retrieves historical price data from Yahoo Finance, extracts meaningful features (including fluctuation patterns like 30-70% price ranges), performs clustering analysis, and generates detailed visualizations and reports.

## Key Features

- **Data Sources**: 
  - Stock symbols from PostgreSQL `metrics` table
  - Historical price data from Yahoo Finance API
  
- **Advanced Feature Extraction**:
  - Volatility analysis (multiple time windows)
  - Fluctuation cycle detection (e.g., 30-70% price ranges)
  - Technical indicators (RSI, MACD, Bollinger Bands)
  - Drawdown analysis and trend strength
  - Statistical return characteristics

- **Clustering Algorithms**:
  - K-Means clustering
  - Hierarchical clustering  
  - Time series clustering with DTW distance
  - Automatic optimal cluster determination

- **Comprehensive Visualization**:
  - Cluster distribution pie charts
  - 2D scatter plots (PCA/TSNE)
  - Feature importance heatmaps
  - Radar plots for cluster profiles
  - Sample time series per cluster

## Database Configuration

The project connects to a PostgreSQL database with these settings:
- **Host**: localhost:5432
- **Database**: mydatabase  
- **Username**: myuser
- **Password**: mypassword
- **Table**: metrics (contains stock symbols)

## Installation

### Prerequisites
- Python 3.8 or higher
- PostgreSQL database with stock symbols

### Setup Steps

1. **Clone the project**:
```bash
git clone <repository-url>
cd Clustering
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up PostgreSQL database**:
   - Ensure your PostgreSQL server is running
   - Create the `metrics` table if it doesn't exist
   - Populate with stock symbols (Yahoo Finance symbols)

4. **Install the project**:
```bash
pip install -e .
```

## Usage

### Basic Usage

Run the complete clustering analysis with default settings:

```bash
python main.py
```

### Advanced Usage

Customize the analysis with command-line arguments:

```bash
python main.py --max-clusters 30 --algorithm kmeans --period 10y --validate-symbols
```

### Command Line Arguments

- `--max-clusters`: Maximum number of clusters to create (default: 50)
- `--algorithm`: Clustering algorithm - kmeans, hierarchical, dbscan (default: kmeans)
- `--period`: Data period to fetch - max, 10y, 5y, 2y (default: max)
- `--validate-symbols`: Validate symbols before downloading (default: True)
- `--features-per-symbol`: Create one feature row per symbol vs per observation (default: True)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--output-dir`: Output directory for results (default: reports)
- `--cache-dir`: Cache directory for data (default: data/raw)
- `--use-cache`: Use cached data if available

## Project Structure

```
Clustering/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                    # Package setup script
├── main.py                     # Main execution script
├── .gitignore                  # Git ignore file
├── config/
│   ├── __init__.py
│   └── database.py             # Database connection configuration
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py         # Yahoo Finance data fetching
│   ├── feature_extractor.py    # Feature engineering module
│   ├── clustering.py           # Clustering algorithms
│   └── visualizer.py          # Visualization and reporting
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── clustering_results.ipynb
├── data/
│   └── raw/                   # Cached stock price data
├── tests/
│   ├── test_data_fetcher.py
│   ├── test_feature_extractor.py
│   └── test_clustering.py
├── reports/                    # Generated reports and visualizations
└── results/                    # CSV exports and analysis results
```

## Methodology

### 1. Data Collection
- Connects to PostgreSQL to fetch stock symbols from `metrics` table
- Validates each symbol with Yahoo Finance
- Downloads maximum available historical price data
- Implements rate limiting and caching for API efficiency

### 2. Feature Engineering
- **Returns**: Daily returns, log returns, cumulative returns
- **Volatility**: Rolling standard deviations (30, 90, 252 days)
- **Fluctuation Cycles**: Counts movements between percentage thresholds (e.g., 30-70%)
- **Technical Indicators**: RSI, MACD, Bollinger Bands, volume ratios
- **Drawdowns**: Maximum drawdown, recovery periods
- **Trends**: Moving averages, trend strength, momentum

### 3. Clustering Analysis
- Automatically determines optimal number of clusters (max 50)
- Uses multiple evaluation metrics (silhouette, Calinski-Harabasz, Davies-Bouldin)
- Supports both traditional and time series clustering
- Generates descriptive cluster labels based on characteristics

### 4. Visualization & Reporting
- Creates comprehensive static reports with embedded visualizations
- Includes cluster profiles, feature importance, and sample time series
- Exports data for further analysis (CSV format)
- Generates detailed markdown analysis report

## Output Files

The analysis generates several output files in the specified directories:

### Reports (`reports/`):
- `cluster_sizes_pie.png` - Distribution of stocks across clusters
- `clusters_scatter.png` - 2D visualization of clusters
- `feature_importance_heatmap.png` - Feature values across clusters
- `cluster_profiles_radar.png` - Radar plots of cluster characteristics
- `sample_time_series.png` - Sample price charts per cluster
- `clustering_metrics.png` - Quality metrics visualization
- `analysis_report.md` - Comprehensive analysis report

### Results (`results/`):
- `cluster_assignments.csv` - Symbol-to-cluster mapping
- `feature_matrix.csv` - Complete feature dataset
- `cluster_summary_table.csv` - Cluster statistics

### Data Cache (`data/raw/`):
- Cached stock price data (parquet format)
- Combined dataset files

## Example Cluster Labels

The system generates descriptive labels like:
- "Large - High Volatility - Growth Stocks - High Fluctuation"
- "Medium - Low Volatility - Stable Stocks - Low Fluctuation"  
- "Small - Moderate Volatility - Declining Stocks - Moderate Fluctuation"

## Customization

### Adding New Features
Extend the `FeatureExtractor` class in `src/feature_extractor.py` to add custom features:

```python
def custom_feature(self, df: pd.DataFrame) -> pd.DataFrame:
    # Your custom feature calculation
    return df
```

### New Clustering Algorithms
Add new clustering methods to the `StockClustering` class in `src/clustering.py`.

### Custom Visualizations
Extend the `ClusterVisualizer` class in `src/visualizer.py` for additional chart types.

## Performance Considerations

- **Memory Usage**: Large datasets may require significant RAM for feature extraction
- **API Limits**: Yahoo Finance has rate limits - the code implements delays
- **Caching**: Uses parquet format for efficient data storage and retrieval
- **Parallel Processing**: Downloads data for multiple symbols concurrently

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Verify PostgreSQL is running
   - Check connection parameters in `config/database.py`
   - Ensure database and table exist

2. **Yahoo Finance API Issues**:
   - Rate limiting may occur with many symbols
   - Some symbols may be delisted or invalid
   - Network connectivity issues

3. **Memory Issues**:
   - Reduce number of symbols or time period
   - Use `--features-per-symbol` flag to reduce data size

4. **Clustering Problems**:
   - Insufficient data for cluster count
   - Too many missing values in features
   - Inappropriate feature scaling

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the log files generated during execution
3. Check the analysis report for detailed information
4. Open an issue with detailed error information