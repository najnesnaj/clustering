"""
Database configuration and connection management for the clustering project.
"""

import os
import logging
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
import pandas as pd


class DatabaseConnection:
    """
    Manages database connections to PostgreSQL for the clustering project.
    """
    
    def __init__(self, host: str = "localhost", port: int = 5432, 
                 database: str = "mydatabase", username: str = "myuser", 
                 password: str = "mypassword"):
        """
        Initialize database connection parameters.
        
        Args:
            host: Database host address
            port: Database port
            database: Database name
            username: Database username
            password: Database password
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.engine = None
        self.SessionLocal = None
        self.logger = logging.getLogger(__name__)
        
    def create_connection_string(self) -> str:
        """Create PostgreSQL connection string."""
        return f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def connect(self):
        """Establish database connection with retry mechanism."""
        if self.engine is None:
            try:
                connection_string = self.create_connection_string()
                self.engine = create_engine(
                    connection_string,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False
                )
                self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
                self.logger.info("Database connection established successfully")
            except SQLAlchemyError as e:
                self.logger.error(f"Failed to connect to database: {e}")
                raise
        return self.engine
    
    def get_session(self):
        """Get database session."""
        if self.SessionLocal is None:
            self.connect()
        return self.SessionLocal()
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            if self.engine is None:
                self.connect()
            
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                return result.fetchone()[0] == 1
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    def fetch_symbols_from_metrics(self) -> pd.DataFrame:
        """
        Fetch all symbols from the metrics table.
        
        Returns:
            DataFrame containing symbols from the metrics table
        """
        try:
            if self.engine is None:
                self.connect()
                
            query = "SELECT DISTINCT symbol FROM metrics ORDER BY symbol"
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Successfully fetched {len(df)} symbols from metrics table")
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch symbols from metrics table: {e}")
            raise
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query.
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with query results
        """
        try:
            if self.engine is None:
                self.connect()
                
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            self.logger.error(f"Failed to execute query: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connection closed")


# Global database instance
db_connection = DatabaseConnection()


def get_db_connection() -> DatabaseConnection:
    """Get global database connection instance."""
    return db_connection