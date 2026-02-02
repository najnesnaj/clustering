"""
Clustering module for stock price analysis and grouping.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from typing import Dict, List, Optional, Tuple, Any

import warnings
warnings.filterwarnings('ignore')


class StockClustering:
    """
    Handles clustering of stocks based on their price behavior and features.
    """
    
    def __init__(self, max_clusters: int = 50):
        """
        Initialize the clustering analyzer.
        
        Args:
            max_clusters: Maximum number of clusters to create
        """
        self.max_clusters = max_clusters
        self.logger = logging.getLogger(__name__)
        self.best_model = None
        self.best_n_clusters = None
        self.cluster_labels = None
        self.feature_names = None
        self.scaler = None
        
    def find_optimal_clusters(self, features: pd.DataFrame, 
                            cluster_range: Optional[range] = None,
                            algorithm: str = 'kmeans') -> Dict[int, Dict[str, float]]:
        """
        Find optimal number of clusters using various metrics.
        
        Args:
            features: Feature matrix (excluding non-numeric columns)
            cluster_range: Range of cluster numbers to test
            algorithm: Clustering algorithm to use
            
        Returns:
            Dictionary with evaluation metrics for each cluster number
        """
        if cluster_range is None:
            cluster_range = range(2, min(self.max_clusters + 1, 51))
        
        # Remove non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        if numeric_features.shape[0] < cluster_range.stop:
            cluster_range = range(2, min(numeric_features.shape[0], 51))
            self.logger.warning(f"Reducing cluster range due to insufficient samples: {cluster_range}")
        
        results = {}
        
        self.logger.info(f"Testing {len(cluster_range)} cluster numbers...")
        
        for n_clusters in cluster_range:
            try:
                if algorithm == 'kmeans':
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                elif algorithm == 'hierarchical':
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                else:
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                
                cluster_labels = model.fit_predict(numeric_features)
                
                # Calculate metrics
                silhouette = silhouette_score(numeric_features, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(numeric_features, cluster_labels)
                davies_bouldin = davies_bouldin_score(numeric_features, cluster_labels)
                
                # Inertia for kmeans
                inertia = None
                if hasattr(model, 'inertia_'):
                    inertia = model.inertia_
                
                results[n_clusters] = {
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski_harabasz,
                    'davies_bouldin_score': davies_bouldin,
                    'inertia': inertia
                }
                
                self.logger.debug(f"n_clusters={n_clusters}: silhouette={silhouette:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error testing {n_clusters} clusters: {e}")
                continue
        
        return results
    
    def select_best_n_clusters(self, cluster_results: Dict[int, Dict[str, float]]) -> int:
        """
        Select the best number of clusters based on evaluation metrics.
        
        Args:
            cluster_results: Results from find_optimal_clusters
            
        Returns:
            Best number of clusters
        """
        if not cluster_results:
            return 5  # Default fallback
        
        # Calculate composite score (weighted average of normalized metrics)
        best_score = -np.inf
        best_n_clusters = 5  # Default
        
        for n_clusters, metrics in cluster_results.items():
            # Normalize metrics (higher is better for all)
            silhouette_norm = metrics['silhouette_score']
            ch_norm = metrics['calinski_harabasz_score'] / max(m['calinski_harabasz_score'] for m in cluster_results.values())
            db_norm = 1 - (metrics['davies_bouldin_score'] / max(m['davies_bouldin_score'] for m in cluster_results.values()))
            
            # Composite score (adjust weights as needed)
            composite_score = 0.4 * silhouette_norm + 0.4 * ch_norm + 0.2 * db_norm
            
            if composite_score > best_score:
                best_score = composite_score
                best_n_clusters = n_clusters
        
        self.logger.info(f"Selected optimal number of clusters: {best_n_clusters}")
        return best_n_clusters
    
    def perform_clustering(self, features: pd.DataFrame, 
                          algorithm: str = 'kmeans',
                          n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Perform clustering on the feature matrix.
        
        Args:
            features: Feature matrix
            algorithm: Clustering algorithm to use
            n_clusters: Number of clusters (if None, will be auto-determined)
            
        Returns:
            Array of cluster labels
        """
        # Remove non-numeric columns and save feature names
        self.feature_names = features.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = features[self.feature_names]
        
        if n_clusters is None:
            # Find optimal number of clusters
            cluster_results = self.find_optimal_clusters(numeric_features, algorithm=algorithm)
            n_clusters = self.select_best_n_clusters(cluster_results)
        
        if n_clusters is None:
            n_clusters = 5  # Default
            
        self.logger.info(f"Performing clustering with {algorithm} and {n_clusters} clusters...")
        
        # Perform clustering
        if algorithm == 'kmeans':
            self.best_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == 'hierarchical':
            self.best_model = AgglomerativeClustering(n_clusters=n_clusters)
        elif algorithm == 'dbscan':
            self.best_model = DBSCAN(eps=0.5, min_samples=5)
        else:
            self.best_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        self.cluster_labels = self.best_model.fit_predict(numeric_features)
        self.best_n_clusters = n_clusters
        
        self.logger.info(f"Clustering completed. Cluster sizes: {pd.Series(self.cluster_labels).value_counts().to_dict()}")
        
        return self.cluster_labels
    
    def perform_time_series_clustering(self, features: pd.DataFrame, 
                                      n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Perform time series clustering using DTW distance.
        
        Args:
            features: Feature matrix with time series data
            n_clusters: Number of clusters
            
        Returns:
            Array of cluster labels
        """
        # Remove non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        if numeric_features.empty:
            raise ValueError("No numeric features found for time series clustering")
        
        # Prepare data for tslearn (3D format: n_samples, n_timestamps, n_features)
        if 'symbol' in features.columns:
            # Group by symbol and create time series
            symbols = features['symbol'].unique()
            ts_data = []
            
            for symbol in symbols:
                symbol_data = features[features['symbol'] == symbol][self.feature_names].values
                if len(symbol_data) > 0:
                    ts_data.append(symbol_data)
            
            if not ts_data:
                raise ValueError("No valid time series data found")
                
            # Convert to numpy array and ensure consistent length
            max_length = max(len(ts) for ts in ts_data)
            ts_data_padded = []
            
            for ts in ts_data:
                if len(ts) < max_length:
                    # Pad with zeros or last value
                    padding = np.zeros((max_length - len(ts), ts.shape[1]))
                    ts_padded = np.vstack([ts, padding])
                else:
                    ts_padded = ts[:max_length]
                ts_data_padded.append(ts_padded)
            
            X = np.array(ts_data_padded)
        else:
            # Reshape for single time series
            X = numeric_features.values.reshape(numeric_features.shape[0], -1, 1)
        
        # Scale time series
        scaler = TimeSeriesScalerMeanVariance()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal clusters if not specified
        if n_clusters is None:
            n_clusters = min(10, X_scaled.shape[0] // 2)
        
        self.logger.info(f"Performing time series clustering with DTW and {n_clusters} clusters...")
        
        # Perform time series k-means with DTW
        self.best_model = TimeSeriesKMeans(
            n_clusters=n_clusters, 
            metric="dtw", 
            random_state=42,
            max_iter=50
        )
        
        self.cluster_labels = self.best_model.fit_predict(X_scaled)
        self.best_n_clusters = n_clusters
        
        return self.cluster_labels
    
    def analyze_clusters(self, features: pd.DataFrame, 
                        cluster_labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Analyze cluster characteristics.
        
        Args:
            features: Feature matrix
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary with cluster analysis
        """
        # Add cluster labels to features
        analysis_data = features.copy()
        analysis_data['cluster'] = cluster_labels
        
        # Remove non-numeric columns for statistical analysis
        numeric_cols = analysis_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'cluster' in numeric_cols:
            numeric_cols.remove('cluster')
        
        cluster_analysis = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_data = analysis_data[analysis_data['cluster'] == cluster_id]
            
            if cluster_data.empty:
                continue
                
            # Basic statistics
            cluster_stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(analysis_data) * 100,
                'symbols': cluster_data['symbol'].unique().tolist() if 'symbol' in cluster_data.columns else []
            }
            
            # Feature statistics
            if numeric_cols:
                feature_means = cluster_data[numeric_cols].mean()
                feature_stds = cluster_data[numeric_cols].std()
                
                for col in numeric_cols:
                    cluster_stats[f'{col}_mean'] = feature_means[col]
                    cluster_stats[f'{col}_std'] = feature_stds[col]
            
            cluster_analysis[cluster_id] = cluster_stats
        
        return cluster_analysis
    
    def generate_cluster_labels(self, cluster_analysis: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
        """
        Generate descriptive labels for clusters based on their characteristics.
        
        Args:
            cluster_analysis: Cluster analysis results
            
        Returns:
            Dictionary mapping cluster IDs to descriptive labels
        """
        cluster_labels = {}
        
        for cluster_id, stats in cluster_analysis.items():
            label_parts = []
            
            # Size descriptor
            size = stats['size']
            if size > 50:
                label_parts.append("Large")
            elif size > 20:
                label_parts.append("Medium")
            else:
                label_parts.append("Small")
            
            # Volatility characteristics
            volatility_mean = stats.get('volatility_252d_mean', 0)
            if volatility_mean > 0.3:
                label_parts.append("Very High Volatility")
            elif volatility_mean > 0.2:
                label_parts.append("High Volatility")
            elif volatility_mean > 0.1:
                label_parts.append("Moderate Volatility")
            else:
                label_parts.append("Low Volatility")
            
            # Return characteristics
            return_mean = stats.get('return_mean', 0)
            if return_mean > 0.001:
                label_parts.append("Growth Stocks")
            elif return_mean < -0.001:
                label_parts.append("Declining Stocks")
            else:
                label_parts.append("Stable Stocks")
            
            # Fluctuation characteristics
            fluctuation_count = stats.get('fluctuation_count_mean', 0)
            if fluctuation_count > 5:
                label_parts.append("High Fluctuation")
            elif fluctuation_count > 2:
                label_parts.append("Moderate Fluctuation")
            else:
                label_parts.append("Low Fluctuation")
            
            # Special characteristics
            if stats.get('rsi_mean', 50) > 70:
                label_parts.append("Overbought")
            elif stats.get('rsi_mean', 50) < 30:
                label_parts.append("Oversold")
            
            # Combine parts
            if label_parts:
                label = " - ".join(label_parts)
            else:
                label = f"Cluster {cluster_id} ({size} stocks)"
            
            cluster_labels[cluster_id] = label
        
        return cluster_labels
    
    def reduce_dimensions(self, features: pd.DataFrame, method: str = 'pca', 
                         n_components: int = 2) -> Tuple[np.ndarray, Any]:
        """
        Reduce dimensionality for visualization.
        
        Args:
            features: Feature matrix
            method: Dimensionality reduction method ('pca', 'tsne')
            n_components: Number of components to reduce to
            
        Returns:
            Tuple of (reduced_features, fitted_transformer)
        """
        numeric_features = features.select_dtypes(include=[np.number])
        
        if method == 'pca':
            transformer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            transformer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(numeric_features)-1))
        else:
            transformer = PCA(n_components=n_components, random_state=42)
        
        reduced_features = transformer.fit_transform(numeric_features)
        
        self.logger.info(f"Dimensionality reduction ({method}): {numeric_features.shape} -> {reduced_features.shape}")
        
        return reduced_features, transformer
    
    def evaluate_clustering_quality(self, features: pd.DataFrame, 
                                  cluster_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics.
        
        Args:
            features: Feature matrix
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary with quality metrics
        """
        numeric_features = features.select_dtypes(include=[np.number])
        
        if len(np.unique(cluster_labels)) < 2:
            self.logger.warning("Cannot evaluate clustering quality with less than 2 clusters")
            return {}
        
        metrics = {
            'silhouette_score': silhouette_score(numeric_features, cluster_labels),
            'calinski_harabasz_score': calinski_harabasz_score(numeric_features, cluster_labels),
            'davies_bouldin_score': davies_bouldin_score(numeric_features, cluster_labels)
        }
        
        # Additional metrics
        n_clusters = len(np.unique(cluster_labels))
        n_samples = len(cluster_labels)
        
        metrics['clusters_to_samples_ratio'] = n_clusters / n_samples
        metrics['cluster_balance'] = 1 - np.std([np.sum(cluster_labels == i) for i in range(n_clusters)]) / np.mean([np.sum(cluster_labels == i) for i in range(n_clusters)])
        
        self.logger.info(f"Clustering quality: Silhouette={metrics['silhouette_score']:.3f}, "
                        f"Calinski-Harabasz={metrics['calinski_harabasz_score']:.1f}, "
                        f"Davies-Bouldin={metrics['davies_bouldin_score']:.3f}")
        
        return metrics
    
    def get_cluster_assignments(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Get cluster assignments for all symbols.
        
        Args:
            features: Original feature matrix
            
        Returns:
            DataFrame with symbols and their cluster assignments
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering has been performed yet")
        
        assignments = pd.DataFrame({
            'symbol': features['symbol'] if 'symbol' in features.columns else range(len(features)),
            'cluster': self.cluster_labels
        })
        
        return assignments