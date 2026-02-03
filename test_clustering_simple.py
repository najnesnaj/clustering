#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Test clustering function
def test_clustering():
    """Test simple clustering without tslearn."""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    # Generate sample features
    volatility = np.random.normal(0.15, 0.1, n_samples)
    returns = np.random.normal(0.001, 0.02, n_samples)
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    
    # Create feature matrix
    features = pd.DataFrame({
        'volatility': volatility,
        'returns': returns,
        'feature1': feature1,
        'feature2': feature2
    })
    
    # Simple clustering analysis
    silhouette_scores = []
    cluster_range = range(3, min(15, len(features)))
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        score = silhouette_score(features, cluster_labels)
        silhouette_scores.append(score)
        print(f"Clusters: {n_clusters}, Silhouette: {score:.3f}")
    
    # Choose best number of clusters
    best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"Best number of clusters: {best_n_clusters}")
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    print(f"Clustering successful! Created {len(np.unique(cluster_labels))} clusters")
    return True

if __name__ == "__main__":
    test_clustering()