import sqlite3
import pandas as pd
from pathlib import Path
import logging

class DatabaseManager:
    """Manages SQLite database for stock clustering demo."""
    
    def __init__(self, db_path: str = "data/clustering.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
    def create_database_schema(self):
        """Create all necessary database tables."""
        
        # Create database directory if it doesn't exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create stock_data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    dividends REAL,
                    stock_splits REAL,
                    PRIMARY KEY (symbol, date)
                )
            ''')
            
            # Create features table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    symbol TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    PRIMARY KEY (symbol, feature_name)
                )
            ''')
            
            # Create clusters table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS clusters (
                    symbol TEXT PRIMARY KEY,
                    cluster_id INTEGER,
                    cluster_label TEXT,
                    confidence_score REAL
                )
            ''')
            
            # Create cluster_info table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cluster_info (
                    cluster_id INTEGER PRIMARY KEY,
                    cluster_label TEXT,
                    size INTEGER,
                    percentage REAL,
                    avg_volatility REAL,
                    avg_return REAL,
                    description TEXT
                )
            ''')
            
            # Create metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_data_symbol ON stock_data(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_data_date ON stock_data(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_clusters_cluster_id ON clusters(cluster_id)')
            
            conn.commit()
            self.logger.info("Database schema created successfully")
    
    def insert_stock_data(self, df: pd.DataFrame):
        """Insert stock price data into database."""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql('stock_data', conn, if_exists='replace', index=False)
            self.logger.info(f"Inserted {len(df)} rows of stock data")
    
    def insert_features(self, features_df: pd.DataFrame):
        """Insert feature data into database."""
        with sqlite3.connect(self.db_path) as conn:
            features_df.to_sql('features', conn, if_exists='replace', index=False)
            self.logger.info(f"Inserted {len(features_df)} feature records")
    
    def insert_cluster_assignments(self, clusters_df: pd.DataFrame):
        """Insert cluster assignments into database."""
        with sqlite3.connect(self.db_path) as conn:
            clusters_df.to_sql('clusters', conn, if_exists='replace', index=False)
            self.logger.info(f"Inserted cluster assignments for {len(clusters_df)} symbols")
    
    def insert_cluster_info(self, cluster_info_dict: dict):
        """Insert cluster information into database."""
        cluster_info_list = []
        for cluster_id, info in cluster_info_dict.items():
            cluster_info_list.append({
                'cluster_id': cluster_id,
                'cluster_label': info.get('label', f'Cluster {cluster_id}'),
                'size': info.get('size', 0),
                'percentage': info.get('percentage', 0),
                'avg_volatility': info.get('avg_volatility', 0),
                'avg_return': info.get('avg_return', 0),
                'description': info.get('description', '')
            })
        
        if cluster_info_list:
            df = pd.DataFrame(cluster_info_list)
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql('cluster_info', conn, if_exists='replace', index=False)
                self.logger.info(f"Inserted information for {len(cluster_info_list)} clusters")
    
    def insert_metadata(self, metadata_dict: dict):
        """Insert metadata into database."""
        with sqlite3.connect(self.db_path) as conn:
            for key, value in metadata_dict.items():
                conn.execute('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)', 
                           (key, str(value)))
            self.logger.info("Metadata updated successfully")
    
    def get_stock_data(self, symbols: list = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Retrieve stock data from database."""
        query = "SELECT * FROM stock_data WHERE 1=1"
        params = []
        
        if symbols:
            placeholders = ','.join(['?'] * len(symbols))
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY symbol, date"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(query, conn, params=params)
            df['date'] = pd.to_datetime(df['date'])
            return df
    
    def get_clusters(self) -> pd.DataFrame:
        """Retrieve cluster assignments."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql('SELECT * FROM clusters ORDER BY cluster_id', conn)
            return df
    
    def get_cluster_info(self) -> pd.DataFrame:
        """Retrieve cluster information."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql('SELECT * FROM cluster_info ORDER BY cluster_id', conn)
            return df
    
    def get_symbols_in_cluster(self, cluster_id: int) -> list:
        """Get all symbols in a specific cluster."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT symbol FROM clusters WHERE cluster_id = ?', (cluster_id,))
            return [row[0] for row in cursor.fetchall()]
    
    def get_metadata(self) -> dict:
        """Retrieve metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM metadata')
            return dict(cursor.fetchall())