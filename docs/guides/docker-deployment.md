# Stock Clustering Docker Demo

A comprehensive stock clustering demonstration with 20 years of historical data (2006-2026) analyzing 100 carefully selected stocks across various market segments.

## 🚀 Quick Start

### Single Command Deployment
```bash
docker run -p 8501:8501 clustering-demo
```

Then visit: http://localhost:8501

That's it! The container comes pre-built with all data processed and analyzed.

## 📊 What's Included

### Data Coverage
- **Time Period**: 2006-2026 (20 years of historical data)
- **Stocks**: 100 curated symbols across market segments
- **Features**: 236+ technical and statistical features
- **Clusters**: Automatically determined optimal groupings

### Stock Categories
- **Large-Cap Established** (25): AAPL, MSFT, GOOGL, AMZN, META, NVDA, etc.
- **Mid-Cap Established** (25): TXN, CSCO, ADP, IBM, ORCL, etc.
- **ETF Representation** (15): SPY, QQQ, IWM, DIA, VTI, etc.
- **International ADRs** (10): ASML, SAP, TSM, BABA, etc.
- **Sector Specific** (25): BLK, SCHW, AXP, COF, etc.

### Features Analyzed
- Returns & volatility metrics
- Trend analysis & moving averages
- Drawdown characteristics
- Technical indicators (RSI, MACD, Bollinger Bands)
- Statistical features (skewness, kurtosis, VaR)
- Market cycle resilience

## 🎯 Interactive Dashboard

### Overview
- Cluster distribution pie charts
- Performance metrics
- Market statistics

### Cluster Analysis
- Detailed breakdown of each cluster
- Stock membership lists
- Performance comparisons
- Volatility and return characteristics

### Stock Explorer
- Individual stock analysis
- Price and volume charts
- Cluster identification
- Peer comparison

### Time Series Analysis
- Multi-stock performance comparison
- Normalized returns visualization
- Custom date range filtering
- Interactive charting

## 🏗️ Architecture

### Single-Container Design
- **Self-contained**: No external dependencies
- **Pre-computed**: All analysis done during build
- **Instant startup**: Zero loading time
- **Embedded SQLite**: Database included in container

### Build Process
1. Downloads 20 years of stock data
2. Extracts 236+ features per stock
3. Performs optimal clustering analysis
4. Stores results in SQLite database
5. Ready for immediate interactive exploration

## 🛠️ Technical Stack

- **Backend**: Python 3.11
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly
- **Data Source**: Yahoo Finance (yfinance)
- **Database**: SQLite

## 📁 Project Structure

```
clustering-demo/
├── Dockerfile              # Container definition
├── app.py                  # Streamlit application
├── build_data.py           # Data processing script
├── database_manager.py     # SQLite database operations
├── requirements.txt        # Python dependencies
├── src/                    # Core modules
│   ├── feature_extractor.py
│   ├── clustering.py
│   └── data_fetcher.py
└── data/                   # Generated SQLite database
```

## 🔧 Build from Source

If you want to build the container yourself:

```bash
# Clone the repository
git clone <repository-url>
cd clustering

# Build the Docker image
docker build -t clustering-demo .

# Run the container
docker run -p 8501:8501 clustering-demo
```

## 📱 Usage Tips

1. **First Load**: Initial data processing during build takes 5-10 minutes
2. **Navigation**: Use sidebar to switch between views
3. **Interactivity**: All charts are fully interactive
4. **Performance**: Optimized for instant responsiveness
5. **Export**: Use browser's save functionality to export charts

## 🎨 Key Features

### Advanced Analytics
- **Optimal Clustering**: Automatically determines best number of clusters
- **Feature Engineering**: 236+ technical and statistical metrics
- **Risk Metrics**: Value-at-Risk, drawdown analysis, volatility profiling
- **Performance Attribution**: Multi-decade performance analysis

### User Experience
- **Zero Configuration**: Works out-of-the-box
- **Responsive Design**: Adapts to different screen sizes
- **Intuitive Navigation**: Clear section organization
- **Rich Visualizations**: Interactive charts and graphs

### Data Quality
- **Survivorship Bias Free**: Includes delisted and failed stocks
- **Complete Coverage**: 20 years of daily price data
- **Data Validation**: Automatic quality checks
- **Error Handling**: Graceful degradation for missing data

## 📈 What You'll Discover

- **Market Segments**: How stocks naturally group by behavior
- **Risk Profiles**: Volatility patterns across different stocks
- **Performance Clusters**: Groups with similar return characteristics
- **Market Cycles**: How different sectors respond to market conditions
- **Hidden Relationships**: Discover unexpected stock similarities

## 🤝 Contributing

This is a demonstration project. For suggestions or improvements:

1. Check the existing issues
2. Create a new issue with detailed description
3. Submit pull requests with clear documentation

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 🔍 Troubleshooting

**Common Issues:**

- **Port 8501 in use**: Change with `-p 8502:8501`
- **Container fails to start**: Check Docker logs with `docker logs <container-id>`
- **No data displayed**: Ensure build completed successfully
- **Slow performance**: Check system resources, may need more RAM

**Getting Help:**
- Check the build logs for any data download errors
- Verify internet connection during initial build
- Ensure Docker has sufficient resources (4GB+ RAM recommended)

---

🚀 **Ready to explore 20 years of market intelligence? Run the command above and dive in!**