"""
Visualization module for stock clustering analysis and reporting.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")


class ClusterVisualizer:
    """
    Creates visualizations for stock clustering analysis.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Configure matplotlib for better output
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
    def plot_cluster_sizes(self, cluster_assignments: pd.DataFrame, 
                          save_path: Optional[str] = None) -> str:
        """
        Create a pie chart showing cluster sizes.
        
        Args:
            cluster_assignments: DataFrame with symbols and cluster assignments
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        cluster_counts = cluster_assignments['cluster'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
        
        plt.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
                autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title('Distribution of Stocks Across Clusters', fontsize=16, fontweight='bold')
        
        if save_path is None:
            save_path = self.output_dir / "cluster_sizes_pie.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Cluster sizes plot saved to {save_path}")
        return str(save_path)
    
    def plot_cluster_scatter(self, features_2d: np.ndarray, 
                           cluster_labels: np.ndarray,
                           cluster_assignments: pd.DataFrame,
                           method: str = 'PCA',
                           save_path: Optional[str] = None) -> str:
        """
        Create scatter plot of clusters in 2D space.
        
        Args:
            features_2d: 2D feature array
            cluster_labels: Cluster assignments
            cluster_assignments: DataFrame with symbols and clusters
            method: Dimensionality reduction method used
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(14, 10))
        
        # Create scatter plot
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=cluster_labels, cmap='tab10', s=50, alpha=0.7)
        
        # Add cluster centers if available
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            cluster_points = features_2d[cluster_labels == label]
            center = np.mean(cluster_points, axis=0)
            plt.scatter(center[0], center[1], c='black', marker='x', s=200, linewidths=3)
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'{method} Component 1', fontsize=12)
        plt.ylabel(f'{method} Component 2', fontsize=12)
        plt.title(f'Stock Clusters Visualization ({method} Reduction)', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add legend with cluster sizes
        legend_elements = []
        cluster_sizes = cluster_assignments['cluster'].value_counts().sort_index()
        for label in unique_labels:
            size = cluster_sizes.get(label, 0)
            legend_elements.append(plt.scatter([], [], c=scatter.cmap(scatter.norm(label)), 
                                              label=f'Cluster {label} ({size} stocks)'))
        
        plt.legend(handles=legend_elements, loc='best', fontsize=10)
        
        if save_path is None:
            save_path = self.output_dir / "clusters_scatter.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Cluster scatter plot saved to {save_path}")
        return str(save_path)
    
    def plot_feature_importance(self, cluster_analysis: Dict[int, Dict[str, Any]],
                              save_path: Optional[str] = None) -> str:
        """
        Create heatmap showing feature importance across clusters.
        
        Args:
            cluster_analysis: Cluster analysis results
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        # Extract feature means for each cluster
        feature_data = {}
        
        for cluster_id, stats in cluster_analysis.items():
            feature_means = {k.replace('_mean', ''): v for k, v in stats.items() 
                          if k.endswith('_mean') and not k.startswith('fluctuation')}
            feature_data[f'Cluster {cluster_id}'] = feature_means
        
        # Create DataFrame
        feature_df = pd.DataFrame(feature_data).T
        
        if feature_df.empty:
            self.logger.warning("No feature data available for importance plot")
            return ""
        
        # Select top features by variance
        feature_variance = feature_df.var(axis=1)
        top_features = feature_variance.nlargest(min(15, len(feature_variance))).index
        feature_df_top = feature_df.loc[top_features]
        
        # Create heatmap
        plt.figure(figsize=(16, 12))
        sns.heatmap(feature_df_top, annot=True, cmap='RdYlBu_r', center=0, 
                   fmt='.3f', cbar_kws={'label': 'Feature Value'})
        
        plt.title('Feature Values Across Clusters (Top Features by Variance)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Clusters', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "feature_importance_heatmap.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Feature importance heatmap saved to {save_path}")
        return str(save_path)
    
    def plot_cluster_profiles(self, cluster_analysis: Dict[int, Dict[str, Any]],
                            save_path: Optional[str] = None) -> str:
        """
        Create radar plots for cluster profiles.
        
        Args:
            cluster_analysis: Cluster analysis results
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        # Select key features for radar plot
        key_features = ['volatility_252d_mean', 'return_mean', 'fluctuation_count_mean', 
                       'drawdown_mean', 'trend_strength_50d_mean', 'sharpe_ratio_mean']
        
        # Prepare data
        cluster_ids = []
        feature_values = []
        
        for cluster_id, stats in cluster_analysis.items():
            values = []
            cluster_ids.append(f'Cluster {cluster_id}')
            
            for feature in key_features:
                value = stats.get(feature, 0)
                # Normalize to 0-1 scale for radar plot
                if feature == 'return_mean':
                    value = max(0, min(1, (value + 0.01) * 50))  # Normalize returns
                elif feature == 'volatility_252d_mean':
                    value = min(1, value * 5)  # Normalize volatility
                elif feature == 'fluctuation_count_mean':
                    value = min(1, value / 10)  # Normalize fluctuation count
                else:
                    value = max(0, min(1, abs(value)))  # Other features
                    
                values.append(value)
            
            feature_values.append(values)
        
        if not feature_values:
            self.logger.warning("No data available for cluster profiles")
            return ""
        
        # Create radar plot
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(key_features), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_ids)))
        
        for i, (cluster_id, values) in enumerate(zip(cluster_ids, feature_values)):
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=cluster_id, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.replace('_mean', '').replace('_', ' ').title() 
                          for f in key_features])
        ax.set_ylim(0, 1)
        ax.set_title('Cluster Profiles Radar Plot', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        if save_path is None:
            save_path = self.output_dir / "cluster_profiles_radar.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Cluster profiles radar plot saved to {save_path}")
        return str(save_path)
    
    def plot_sample_time_series(self, stock_data: pd.DataFrame, 
                              cluster_assignments: pd.DataFrame,
                              cluster_analysis: Dict[int, Dict[str, Any]],
                              n_samples_per_cluster: int = 3,
                              save_path: Optional[str] = None) -> str:
        """
        Plot sample time series for each cluster.
        
        Args:
            stock_data: Original stock price data
            cluster_assignments: Cluster assignments
            cluster_analysis: Cluster analysis results
            n_samples_per_cluster: Number of sample stocks to plot per cluster
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        # Merge cluster assignments with stock data
        data_with_clusters = stock_data.merge(cluster_assignments, on='symbol')
        
        # Get unique clusters
        clusters = sorted(data_with_clusters['cluster'].unique())
        
        # Calculate subplot grid
        n_clusters = len(clusters)
        n_cols = min(3, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
        if n_clusters == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, cluster_id in enumerate(clusters):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            cluster_symbols = cluster_data['symbol'].unique()
            
            # Select random samples
            if len(cluster_symbols) > n_samples_per_cluster:
                sample_symbols = np.random.choice(cluster_symbols, n_samples_per_cluster, replace=False)
            else:
                sample_symbols = cluster_symbols
            
            # Plot sample time series
            for symbol in sample_symbols:
                symbol_data = cluster_data[cluster_data['symbol'] == symbol]
                ax.plot(symbol_data['date'], symbol_data['close'], 
                       label=symbol, alpha=0.7, linewidth=1)
            
            ax.set_title(f'Cluster {cluster_id} Samples\n{len(cluster_symbols)} stocks total', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Hide empty subplots
        for i in range(n_clusters, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows == 1:
                fig.delaxes(axes[col])
            else:
                fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "sample_time_series.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Sample time series plot saved to {save_path}")
        return str(save_path)
    
    def plot_clustering_metrics(self, clustering_metrics: Dict[str, float],
                              save_path: Optional[str] = None) -> str:
        """
        Create bar chart of clustering quality metrics.
        
        Args:
            clustering_metrics: Dictionary of clustering quality metrics
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        metrics = clustering_metrics.copy()
        
        # Remove metrics that don't make sense in bar chart
        bar_metrics = {k: v for k, v in metrics.items() 
                      if k not in ['clusters_to_samples_ratio', 'cluster_balance']}
        
        if not bar_metrics:
            self.logger.warning("No suitable metrics for bar chart")
            return ""
        
        plt.figure(figsize=(12, 8))
        
        metric_names = list(bar_metrics.keys())
        metric_values = list(bar_metrics.values())
        
        bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(metric_values),
                    f'{value:.3f}', ha='center', va='bottom', fontsize=12)
        
        plt.title('Clustering Quality Metrics', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Adjust y-axis to make bars more visible
        if max(metric_values) < 1:
            plt.ylim(0, 1.2)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "clustering_metrics.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Clustering metrics plot saved to {save_path}")
        return str(save_path)
    
    def create_comprehensive_report(self, features_2d: np.ndarray,
                                  cluster_labels: np.ndarray,
                                  cluster_assignments: pd.DataFrame,
                                  cluster_analysis: Dict[int, Dict[str, Any]],
                                  clustering_metrics: Dict[str, float],
                                  stock_data: Optional[pd.DataFrame] = None,
                                  method: str = 'PCA') -> Dict[str, str]:
        """
        Create a comprehensive set of visualizations.
        
        Args:
            features_2d: 2D feature array
            cluster_labels: Cluster assignments
            cluster_assignments: DataFrame with symbols and clusters
            cluster_analysis: Cluster analysis results
            clustering_metrics: Clustering quality metrics
            stock_data: Original stock price data (optional)
            method: Dimensionality reduction method
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plots_created = {}
        
        try:
            plots_created['cluster_sizes'] = self.plot_cluster_sizes(cluster_assignments)
            self.logger.info("✓ Cluster sizes plot created")
        except Exception as e:
            self.logger.error(f"✗ Failed to create cluster sizes plot: {e}")
        
        try:
            plots_created['cluster_scatter'] = self.plot_cluster_scatter(
                features_2d, cluster_labels, cluster_assignments, method
            )
            self.logger.info("✓ Cluster scatter plot created")
        except Exception as e:
            self.logger.error(f"✗ Failed to create cluster scatter plot: {e}")
        
        try:
            plots_created['feature_importance'] = self.plot_feature_importance(cluster_analysis)
            self.logger.info("✓ Feature importance heatmap created")
        except Exception as e:
            self.logger.error(f"✗ Failed to create feature importance heatmap: {e}")
        
        try:
            plots_created['cluster_profiles'] = self.plot_cluster_profiles(cluster_analysis)
            self.logger.info("✓ Cluster profiles radar plot created")
        except Exception as e:
            self.logger.error(f"✗ Failed to create cluster profiles radar plot: {e}")
        
        try:
            plots_created['clustering_metrics'] = self.plot_clustering_metrics(clustering_metrics)
            self.logger.info("✓ Clustering metrics plot created")
        except Exception as e:
            self.logger.error(f"✗ Failed to create clustering metrics plot: {e}")
        
        if stock_data is not None:
            try:
                plots_created['sample_time_series'] = self.plot_sample_time_series(
                    stock_data, cluster_assignments, cluster_analysis
                )
                self.logger.info("✓ Sample time series plot created")
            except Exception as e:
                self.logger.error(f"✗ Failed to create sample time series plot: {e}")
        
        self.logger.info(f"Comprehensive visualization report completed. {len(plots_created)} plots created.")
        return plots_created
    
    def save_cluster_summary_table(self, cluster_analysis: Dict[int, Dict[str, Any]],
                                 cluster_labels: Dict[int, str],
                                 save_path: Optional[str] = None) -> str:
        """
        Create and save a summary table of cluster characteristics.
        
        Args:
            cluster_analysis: Cluster analysis results
            cluster_labels: Descriptive cluster labels
            save_path: Optional path to save the table
            
        Returns:
            Path to saved file
        """
        # Create summary DataFrame
        summary_data = []
        
        for cluster_id, stats in cluster_analysis.items():
            summary_row = {
                'Cluster ID': cluster_id,
                'Label': cluster_labels.get(cluster_id, f'Cluster {cluster_id}'),
                'Size': stats['size'],
                'Percentage': f"{stats['percentage']:.1f}%",
                'Avg Volatility': f"{stats.get('volatility_252d_mean', 0):.3f}",
                'Avg Return': f"{stats.get('return_mean', 0):.4f}",
                'Fluctuation Count': f"{stats.get('fluctuation_count_mean', 0):.1f}",
                'Sharpe Ratio': f"{stats.get('sharpe_ratio_mean', 0):.3f}",
                'Max Drawdown': f"{stats.get('drawdown_mean', 0):.3f}",
                'Sample Symbols': ', '.join(stats.get('symbols', [])[:5])  # First 5 symbols
            }
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        
        if save_path is None:
            save_path = self.output_dir / "cluster_summary_table.csv"
        
        summary_df.to_csv(save_path, index=False)
        
        self.logger.info(f"Cluster summary table saved to {save_path}")
        return str(save_path)