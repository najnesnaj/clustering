---
layout: default
title: Stock Clustering Analysis
---

# Stock Clustering Analysis Project

Welcome to the Stock Clustering Analysis project! This repository contains an interactive web application for performing stock market clustering analysis using machine learning techniques.

## 🚀 Features

- **Interactive Dashboard**: Streamlit-based web interface for real-time analysis
- **Machine Learning**: K-means clustering algorithm for stock categorization
- **Data Visualization**: Plotly charts for interactive data exploration
- **Technical Indicators**: 6 key technical features for each stock
- **Database Integration**: SQLite database for efficient data management

## 📊 What's Included

### Stock Data
- **15 diversified stocks** from various sectors
- **5 years of historical data** for comprehensive analysis
- **6 technical features** per stock:
  - Returns
  - Volatility
  - RSI (Relative Strength Index)
  - Moving Averages
  - Volume patterns
  - Price momentum

### Analysis Features
- **Optimal clustering** with automatic K detection
- **Professional visualizations** using Plotly
- **Real-time updates** and interactive controls
- **Export capabilities** for results

## 🐳 Quick Start with Docker

The easiest way to get started is with our pre-built Docker container:

```bash
docker run -p 8501:8501 najmussa/stock-clustering:latest
```

Then visit [http://localhost:8501](http://localhost:8501) to access the interactive dashboard.

## 📁 Repository Structure

```
├── app.py                  # Main Streamlit application
├── build_data_mock.py     # Data processing pipeline
├── database_manager.py    # Database operations
├── demo_stock_data.csv    # Sample stock data
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
└── README.md            # Detailed documentation
```

## 🛠️ Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/najmus-saqib/clustering.git
   cd clustering
   ```

2. **Set up the environment**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📈 Analysis Results

The clustering analysis groups stocks into distinct categories based on:

- **Risk profiles** (volatility levels)
- **Performance patterns** (returns and trends)
- **Technical characteristics** (indicators and momentum)

This helps investors:
- Identify similar stocks for portfolio diversification
- Recognize undervalued opportunities
- Understand risk-reward relationships

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Built with ❤️ using Python, Streamlit, and Scikit-learn**