# Stock Clustering Analysis Project

A comprehensive Python project for clustering stocks based on their price behavior and characteristics using data from PostgreSQL and Yahoo Finance database.

## 🎯 Project Overview

**Objective**: Create a comprehensive stock clustering analysis tool that groups stocks based on their price behavior and characteristics using data from PostgreSQL `metrics` table and Yahoo Finance API.

**Key Requirements**:
- Fetch stock symbols from PostgreSQL database (localhost:5432, database: mydatabase, user: myuser, password: mypassword)
- Download maximum available historical price data from Yahoo Finance for each symbol
- Extract meaningful features including fluctuation patterns (e.g., 30-70% price ranges)
- Perform clustering analysis with maximum 50 clusters
- Generate descriptive labels for clusters
- Create comprehensive static reports with visualizations

## 🏗️ Technical Architecture

### 1. Project Structure
```
Clustering/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── setup.py                    # Package installation
├── main.py                     # Main execution script
├── .gitignore                  # Git ignore rules
├── config/
│   ├── __init__.py
│   └── database.py             # Database connection management
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py         # Yahoo Finance API integration
│   ├── feature_extractor.py    # Advanced feature engineering
│   ├── clustering.py           # Clustering algorithms
│   └── visualizer.py          # Visualization and reporting
├── notebooks/
│   └── exploratory_analysis.ipynb # Interactive analysis
├── tests/
│   └── test_data_fetcher.py    # Unit tests
├── data/raw/                   # Cache directory for stock data
├── reports/                    # Output for visualizations
└── results/                    # CSV exports and analysis results
```

### 2. Core Components

#### 2.1 Database Connection (`config/database.py`)
- **Purpose**: Manage PostgreSQL connections
- **Features**:
  - Connection pooling for performance
  - Retry mechanism with exponential backoff
  - Environment variable configuration
  - Connection health checks

#### 2.2 Data Fetcher (`src/data_fetcher.py`)
- **Purpose**: Retrieve stock symbols from database and price data from Yahoo Finance
- **Features**:
  - Parallel downloading with configurable workers
  - Data caching with parquet format
  - Rate limiting and API error handling
  - Symbol validation before download
  - Data quality checks and cleaning

#### 2.3 Feature Extractor (`src/feature_extractor.py`)
- **Purpose**: Extract 236+ features for clustering analysis
- **Features**:
  - Returns: Daily, log, cumulative returns
  - Volatility: Multiple time windows (30, 90, 252 days)
  - Trend: Moving averages, trend strength
  - Drawdown: Maximum drawdown, recovery periods
  - Technical: RSI, MACD, Bollinger Bands
  - Statistical: Skewness, kurtosis, VaR
  - **Fluctuation Cycles**: Counts movements between percentage thresholds (30-70%)

#### 2.4 Clustering (`src/clustering.py`)
- **Purpose**: Perform various clustering algorithms
- **Features**:
  - K-Means with automatic optimal cluster detection
  - Hierarchical clustering with various linkages
  - Time series clustering with DTW distance
  - Cluster quality evaluation (silhouette, Calinski-Harabasz, Davies-Bouldin)
  - Dimensionality reduction (PCA, t-SNE)
  - Automatic cluster labeling based on characteristics

#### 2.5 Visualizer (`src/visualizer.py`)
- **Purpose**: Create comprehensive reports and visualizations
- **Features**:
  - Cluster distribution pie charts
  - 2D scatter plots with cluster overlays
  - Feature importance heatmaps
  - Radar plots for cluster profiles
  - Sample time series per cluster
  - Performance metrics visualization
  - Export capabilities (PNG, CSV)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- PostgreSQL database with stock symbols in `metrics` table
- Internet connection for Yahoo Finance API

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Clustering

# Install dependencies
pip install -r requirements.txt

# Install the project
pip install -e .
```

### Usage

#### Basic Usage
```bash
# Run complete clustering analysis with default settings
python main.py
```

#### Advanced Usage
```bash
# Customize analysis with command-line arguments
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

## 📊 Key Features

### Data Sources
- **PostgreSQL Database**: Stock symbols from `metrics` table
- **Yahoo Finance API**: Historical price data (OHLCV)
- **Data Validation**: Symbol checking and quality control

### Advanced Feature Extraction
- **Returns Analysis**: Daily returns, log returns, cumulative returns
- **Volatility Analysis**: Multiple time windows, regime changes
- **Fluctuation Cycles**: Unique feature counting price range movements
- **Technical Indicators**: RSI, MACD, Bollinger Bands, volume ratios
- **Drawdown Analysis**: Maximum drawdown, recovery periods, drawdown duration
- **Trend Analysis**: Moving averages, trend strength, momentum
- **Statistical Features**: Skewness, kurtosis, VaR, positive return ratios

### Clustering Capabilities
- **Multiple Algorithms**: K-Means, Hierarchical, DBSCAN
- **Time Series Clustering**: DTW distance for temporal patterns
- **Optimal Cluster Detection**: Automatic selection using multiple metrics
- **Cluster Analysis**: Size, characteristics, descriptive labels
- **Dimensionality Reduction**: PCA and t-SNE for visualization

### Visualization & Reporting
- **Static Reports**: Comprehensive HTML reports with embedded charts
- **Interactive Charts**: Cluster profiles, feature importance, time series
- **Export Options**: CSV data exports, PNG chart downloads
- **Cluster Profiles**: Detailed characteristics per cluster

## 🐳 Docker Demo (Quick Start)

For users who want an instant demonstration without database setup, we provide a **Docker container** with pre-built data:

### Single Command Deployment
```bash
# Build and run the container
docker build -t clustering-demo .
docker run -p 8501:8501 clustering-demo

# Access the interactive dashboard
# Visit: http://localhost:8501
```

**See `README_DOCKER.md` for detailed Docker demo information.**

## 📁 Output Files

### Reports (`reports/`)
- `cluster_sizes_pie.png` - Distribution of stocks across clusters
- `clusters_scatter.png` - 2D visualization of clusters
- `feature_importance_heatmap.png` - Feature values across clusters
- `cluster_profiles_radar.png` - Radar plots of cluster characteristics
- `sample_time_series.png` - Sample price charts per cluster
- `clustering_metrics.png` - Quality metrics visualization
- `analysis_report.md` - Comprehensive analysis report

### Results (`results/`)
- `cluster_assignments.csv` - Symbol-to-cluster mapping
- `feature_matrix.csv` - Complete feature dataset
- `cluster_summary_table.csv` - Cluster statistics

### Data Cache (`data/raw/`)
- Cached stock price data (parquet format)
- Combined dataset files

## 🔧 Development

### Adding New Features
Extend the `FeatureExtractor` class in `src/feature_extractor.py`:

```python
def custom_feature(self, df: pd.DataFrame) -> pd.DataFrame:
    # Your custom feature calculation
    return df
```

### New Clustering Algorithms
Add new clustering methods to the `StockClustering` class in `src/clustering.py`.

### Custom Visualizations
Extend the `ClusterVisualizer` class in `src/visualizer.py` for additional chart types.

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_data_fetcher.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 🎯 Use Cases

### Portfolio Management
- **Risk Assessment**: Group stocks by volatility and return characteristics
- **Diversification**: Identify stocks with different behavioral patterns
- **Performance Attribution**: Understand which factors drive returns

### Market Analysis
- **Market Segmentation**: Discover natural groupings in the market
- **Factor Analysis**: Identify common risk factors across groups
- **Sector Analysis**: Compare behavior across different industries

### Investment Research
- **Screening**: Find stocks with desired characteristics
- **Strategy Development**: Test clustering-based investment strategies
- **Backtesting**: Use clusters for portfolio construction

## ⚠️ Performance Considerations

- **Memory Usage**: Large datasets may require significant RAM for feature extraction
- **API Limits**: Yahoo Finance has rate limits - code implements delays
- **Caching**: Uses parquet format for efficient data storage and retrieval
- **Parallel Processing**: Downloads data for multiple symbols concurrently

## 🐛 Troubleshooting

### Database Issues
1. **Connection Errors**:
   - Verify PostgreSQL is running
   - Check connection parameters in `config/database.py`
   - Ensure database and table exist

2. **Performance Issues**:
   - Adjust connection pool size
   - Add appropriate indexes to database
   - Use data caching effectively

### Yahoo Finance API Issues
- **Rate Limiting**: May occur with many symbols
- **Symbol Errors**: Some symbols may be delisted or invalid
- **Network Issues**: Check internet connectivity and firewall settings

### Clustering Problems
- **Memory Issues**: Reduce number of symbols or use streaming approach
- **Poor Results**: Check feature scaling and parameter tuning
- **Too Many Clusters**: Use domain knowledge to set reasonable limits

## 📜 License

This project is open source and available under MIT License.

## 🤝 Contributing

1. Fork repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs generated during execution
3. Check the analysis reports for detailed information
4. Open an issue with detailed error information and system specifications

---

**For quick demo without database setup, see the Docker demo option above.**
