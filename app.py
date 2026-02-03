import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append('/app/src')

# Import our modules
from database_manager import DatabaseManager

# Set page config
st.set_page_config(
    page_title="Stock Clustering Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database manager
@st.cache_resource
def get_database():
    """Initialize database connection."""
    return DatabaseManager("data/clustering.db")

# Load data with caching
@st.cache_data
def load_stock_data():
    """Load stock data from database."""
    db = get_database()
    return db.get_stock_data()

@st.cache_data
def load_clusters():
    """Load cluster assignments from database."""
    db = get_database()
    return db.get_clusters()

@st.cache_data
def load_cluster_info():
    """Load cluster information from database."""
    db = get_database()
    return db.get_cluster_info()

@st.cache_data
def load_metadata():
    """Load metadata from database."""
    db = get_database()
    return db.get_metadata()

def main():
    """Main Streamlit application."""
    st.title("📊 Stock Clustering Demo")
    st.markdown("---")
    
    # Load data
    try:
        stock_data = load_stock_data()
        clusters = load_clusters()
        cluster_info = load_cluster_info()
        metadata = load_metadata()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please make sure the data has been built by running 'python build_data.py'")
        return
    
    # Display metadata
    if metadata:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Symbols", metadata.get('total_symbols', 'N/A'))
        with col2:
            st.metric("Total Records", metadata.get('total_records', 'N/A'))
        with col3:
            st.metric("Number of Clusters", metadata.get('num_clusters', 'N/A'))
        with col4:
            st.metric("Date Range", f"{metadata.get('date_range_start', 'N/A')} to {metadata.get('date_range_end', 'N/A')}")
    
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a view:",
        ["Overview", "Cluster Analysis", "Stock Explorer", "Time Series Analysis"]
    )
    
    if page == "Overview":
        show_overview(clusters, cluster_info)
    elif page == "Cluster Analysis":
        show_cluster_analysis(clusters, cluster_info, stock_data)
    elif page == "Stock Explorer":
        show_stockExplorer(clusters, stock_data)
    elif page == "Time Series Analysis":
        show_time_series_analysis(stock_data, clusters)

def show_overview(clusters, cluster_info):
    """Show overview dashboard."""
    st.header("📈 Overview Dashboard")
    
    # Cluster distribution pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Distribution")
        if not cluster_info.empty:
            fig = px.pie(
                cluster_info, 
                values='size', 
                names='cluster_label',
                title='Stock Distribution Across Clusters'
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No cluster information available")
    
    with col2:
        st.subheader("Cluster Sizes")
        if not cluster_info.empty:
            fig = px.bar(
                cluster_info,
                x='cluster_label',
                y='size',
                title='Number of Stocks per Cluster',
                labels={'size': 'Number of Stocks', 'cluster_label': 'Cluster'}
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No cluster information available")
    
    # Cluster characteristics table
    st.subheader("Cluster Characteristics")
    if not cluster_info.empty:
        # Format the data for better display
        display_info = cluster_info.copy()
        display_info['avg_volatility'] = display_info['avg_volatility'].round(4)
        display_info['avg_return'] = display_info['avg_return'].round(4)
        display_info['percentage'] = display_info['percentage'].round(1)
        
        st.dataframe(
            display_info[['cluster_label', 'size', 'percentage', 'avg_volatility', 'avg_return']],
            use_container_width=True
        )

def show_cluster_analysis(clusters, cluster_info, stock_data):
    """Show detailed cluster analysis."""
    st.header("🔍 Cluster Analysis")
    
    # Cluster selection
    if cluster_info.empty:
        st.warning("No cluster information available")
        return
    
    selected_cluster = st.selectbox(
        "Select a cluster to analyze:",
        options=cluster_info['cluster_label'].tolist(),
        format_func=lambda x: f"{x} ({cluster_info[cluster_info['cluster_label'] == x]['size'].iloc[0]} stocks)"
    )
    
    # Get cluster ID from label
    cluster_id = cluster_info[cluster_info['cluster_label'] == selected_cluster]['cluster_id'].iloc[0]
    
    # Show cluster details
    cluster_details = cluster_info[cluster_info['cluster_label'] == selected_cluster].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Stocks", cluster_details['size'])
    with col2:
        st.metric("Percentage", f"{cluster_details['percentage']:.1f}%")
    with col3:
        st.metric("Avg Volatility", f"{cluster_details['avg_volatility']:.4f}")
    
    # Show stocks in this cluster
    cluster_stocks = clusters[clusters['cluster_id'] == cluster_id]['symbol'].tolist()
    
    st.subheader(f"Stocks in {selected_cluster}")
    stock_list_df = pd.DataFrame({'Symbol': cluster_stocks})
    st.dataframe(stock_list_df, use_container_width=True)
    
    # Performance comparison (if we have price data)
    if not stock_data.empty:
        st.subheader("Performance Comparison")
        
        # Get price data for stocks in this cluster
        cluster_price_data = stock_data[stock_data['symbol'].isin(cluster_stocks)]
        
        if not cluster_price_data.empty:
            # Calculate normalized returns for each stock
            normalized_returns = []
            for symbol in cluster_stocks:
                symbol_data = cluster_price_data[cluster_price_data['symbol'] == symbol]
                if len(symbol_data) > 1:
                    first_price = symbol_data['close'].iloc[0]
                    if first_price > 0:
                        symbol_data = symbol_data.copy()
                        symbol_data['normalized_return'] = (symbol_data['close'] / first_price - 1) * 100
                        symbol_data['symbol'] = symbol
                        normalized_returns.append(symbol_data[['date', 'symbol', 'normalized_return']])
            
            if normalized_returns:
                normalized_df = pd.concat(normalized_returns, ignore_index=True)
                
                # Plot normalized returns
                fig = px.line(
                    normalized_df,
                    x='date',
                    y='normalized_return',
                    color='symbol',
                    title=f'Normalized Performance of Stocks in {selected_cluster} (%)',
                    labels={'normalized_return': 'Return (%)', 'date': 'Date'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

def show_stockExplorer(clusters, stock_data):
    """Show individual stock explorer."""
    st.header("🔎 Stock Explorer")
    
    # Stock selection
    if clusters.empty:
        st.warning("No cluster data available")
        return
    
    selected_stock = st.selectbox(
        "Select a stock to explore:",
        options=clusters['symbol'].tolist(),
        format_func=lambda x: f"{x} - Cluster {clusters[clusters['symbol'] == x]['cluster_label'].iloc[0]}"
    )
    
    # Get stock's cluster information
    stock_cluster_info = clusters[clusters['symbol'] == selected_stock].iloc[0]
    
    # Display stock information
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stock Information")
        st.write(f"**Symbol:** {selected_stock}")
        st.write(f"**Cluster:** {stock_cluster_info['cluster_label']}")
        st.write(f"**Confidence Score:** {stock_cluster_info['confidence_score']:.2f}")
    
    with col2:
        st.subheader("Cluster Mates")
        cluster_mates = clusters[clusters['cluster_id'] == stock_cluster_info['cluster_id']]['symbol'].tolist()
        cluster_mates.remove(selected_stock)  # Remove the selected stock
        st.write(", ".join(cluster_mates[:10]))  # Show first 10
        if len(cluster_mates) > 10:
            st.write(f"... and {len(cluster_mates) - 10} more")
    
    # Show stock price data if available
    if not stock_data.empty:
        stock_price_data = stock_data[stock_data['symbol'] == selected_stock]
        
        if not stock_price_data.empty:
            st.subheader("Price History")
            
            # Create subplots for price and volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=['Price', 'Volume'],
                row_width=[0.2, 0.7]
            )
            
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=stock_price_data['date'],
                    y=stock_price_data['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Volume chart
            if 'volume' in stock_price_data.columns:
                fig.add_trace(
                    go.Bar(
                        x=stock_price_data['date'],
                        y=stock_price_data['volume'],
                        name='Volume',
                        marker_color='lightblue'
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title=f'{selected_stock} Price History',
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_time_series_analysis(stock_data, clusters):
    """Show time series analysis."""
    st.header("📅 Time Series Analysis")
    
    if stock_data.empty:
        st.warning("No stock data available")
        return
    
    # Date range selection
    min_date = stock_data['date'].min().date()
    max_date = stock_data['date'].max().date()
    
    start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)
    
    if start_date >= end_date:
        st.error("End date must be after start date")
        return
    
    # Filter data by date range
    filtered_data = stock_data[
        (stock_data['date'].dt.date >= start_date) & 
        (stock_data['date'].dt.date <= end_date)
    ]
    
    if filtered_data.empty:
        st.warning("No data available for selected date range")
        return
    
    # Multi-stock selection
    available_symbols = sorted(filtered_data['symbol'].unique())
    selected_symbols = st.multiselect(
        "Select stocks to compare:",
        options=available_symbols,
        default=available_symbols[:5],  # Default to first 5 stocks
        max_selections=10
    )
    
    if not selected_symbols:
        st.warning("Please select at least one stock")
        return
    
    # Create comparison chart
    comparison_data = filtered_data[filtered_data['symbol'].isin(selected_symbols)]
    
    # Calculate normalized returns
    normalized_data = []
    for symbol in selected_symbols:
        symbol_data = comparison_data[comparison_data['symbol'] == symbol].copy()
        if len(symbol_data) > 1:
            first_price = symbol_data['close'].iloc[0]
            if first_price > 0:
                symbol_data['normalized_return'] = (symbol_data['close'] / first_price - 1) * 100
                normalized_data.append(symbol_data)
    
    if normalized_data:
        normalized_df = pd.concat(normalized_data, ignore_index=True)
        
        # Create the chart
        fig = px.line(
            normalized_df,
            x='date',
            y='normalized_return',
            color='symbol',
            title=f'Normalized Performance Comparison ({start_date} to {end_date})',
            labels={'normalized_return': 'Return (%)', 'date': 'Date'}
        )
        
        fig.update_layout(
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add cluster information if available
        if not clusters.empty:
            st.subheader("Cluster Information")
            cluster_info_table = []
            for symbol in selected_symbols:
                symbol_cluster = clusters[clusters['symbol'] == symbol]
                if not symbol_cluster.empty:
                    cluster_info_table.append({
                        'Symbol': symbol,
                        'Cluster': symbol_cluster['cluster_label'].iloc[0],
                        'Cluster ID': symbol_cluster['cluster_id'].iloc[0]
                    })
            
            if cluster_info_table:
                cluster_df = pd.DataFrame(cluster_info_table)
                st.dataframe(cluster_df, use_container_width=True)

if __name__ == "__main__":
    main()