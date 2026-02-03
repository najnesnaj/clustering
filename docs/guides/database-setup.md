# Database Setup for Stock Clustering

## Price Data Table Setup

The system now supports storing and retrieving stock price data from your local PostgreSQL database instead of always fetching from Yahoo Finance.

### Create the Database Table

```sql
-- Connect to your PostgreSQL database
CREATE TABLE IF NOT EXISTS price_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    close_price DECIMAL(10,4),
    volume BIGINT,
    dividends DECIMAL(10,4),
    stock_splits DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_price_data_symbol ON price_data(symbol);
CREATE INDEX IF NOT EXISTS idx_price_data_date ON price_data(date);
```

### Load Historical Data

Option 1: **Manual Data Loading**
```sql
-- Example: Insert AAPL data
INSERT INTO price_data (symbol, date, open_price, high_price, low_price, close_price, volume, dividends, stock_splits)
VALUES 
    ('AAPL', '2023-01-01', 130.28, 132.45, 129.11, 131.25, 100000000, 0, 0),
    ('AAPL', '2023-01-02', 131.25, 133.20, 130.50, 132.60, 105000000, 0, 0),
    -- ... more rows
;
```

Option 2: **Bulk Loading from CSV**
```sql
-- Load from CSV file
COPY price_data (symbol, date, open_price, high_price, low_price, close_price, volume, dividends, stock_splits)
FROM '/path/to/your/stock_data.csv' 
WITH (FORMAT csv, HEADER);
```

### Update Metrics Table

```sql
-- Ensure your symbols are in the metrics table
INSERT INTO metrics (symbol) VALUES 
('AAPL'), ('MSFT'), ('GOOGL'), ('TSLA'), ('AMZN'), 
-- Add all your desired stock symbols
ON CONFLICT (symbol) DO NOTHING;
```

## How the System Works

1. **Automatic Detection**: The system first checks if price_data exists and has data
2. **Smart Data Source Selection**: 
   - If database has data → Use local database (faster, no API limits)
   - If database empty → Fetch from Yahoo Finance
3. **Caching**: Data downloaded from Yahoo Finance is still cached locally
4. **Fallback Gracefully**: If all data sources fail, system reports error and exits

## Benefits

✅ **Performance**: Database queries are much faster than API calls  
✅ **Reliability**: Local data is always available (no API failures)  
✅ **Cost Efficiency**: Reduces Yahoo Finance API usage  
✅ **Offline Capability**: Can run analysis without internet connection  
✅ **Data Control**: You have full control over data quality and history

## Usage

Once your database is set up with historical price data:

```bash
# The system will automatically detect and use your database
python main.py --max-clusters 50 --period max

# Force fetch from Yahoo Finance even if database exists
python main.py --use-cache false
```

## Maintenance

### Update Data Periodically
```sql
-- Add new daily data (automated scripts can be created)
INSERT INTO price_data (symbol, date, open_price, high_price, low_price, close_price, volume, dividends, stock_splits)
SELECT symbol, date, open, high, low, close, volume, adj_close, dividends
FROM yahoo_finance_source
WHERE symbol IN (SELECT symbol FROM metrics)
AND date >= CURRENT_DATE - INTERVAL '1 day';
```

### Data Quality Checks
```sql
-- Check for missing data
SELECT symbol, COUNT(*) as total_records, 
       MIN(date) as earliest_date, 
       MAX(date) as latest_date,
       COUNT(DISTINCT date) as unique_dates
FROM price_data 
GROUP BY symbol;
```

---

**This approach gives you complete control over your stock data while maintaining all the clustering functionality!**