"""
K-means Clustering Implementation
Performs centroid-based clustering using K-means algorithm
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from ....utils.nlp_utils import nlp_utils
from ....config.settings import settings

try:
    from sklearn.cluster import KMeans
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False


class KMeansClustering:
    """K-means clustering implementation following detailed specification"""
    
    async def cluster_signals(self, signals: List) -> List[Dict[str, Any]]:
        """Perform K-means clustering"""
        if not CLUSTERING_AVAILABLE or not signals:
            return []
        
        try:
            texts = [signal.content for signal in signals]
            embeddings = nlp_utils.encode_sentences(texts)
            
            n_clusters = min(len(signals) // settings.MIN_CLUSTER_SIZE, 10)
            n_clusters = max(2, n_clusters)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Create clusters
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(signals[i])
            
            cluster_objects = []
            for cluster_id, cluster_signals in clusters.items():
                if len(cluster_signals) >= settings.MIN_CLUSTER_SIZE:
                    cluster_objects.append({
                        'cluster_id': f"kmeans_cluster_{cluster_id}",
                        'cluster_type': 'kmeans',
                        'size': len(cluster_signals),
                        'signals': [signal.id for signal in cluster_signals],
                        'avg_confidence': np.mean([signal.confidence_score for signal in cluster_signals]),
                        'source_distribution': self._calculate_source_distribution(cluster_signals),
                        'created_at': datetime.now().isoformat()
                    })
            
            return cluster_objects
            
        except Exception as e:
            print(f"K-means clustering failed: {e}")
            return []
    
    def _calculate_source_distribution(self, signals: List) -> Dict[str, float]:
        """Calculate distribution of source types"""
        source_counts = {}
        for signal in signals:
            source_type = signal.source_type
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        total = len(signals)
        return {source: count/total for source, count in source_counts.items()}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for K-means clustering"""
        return {
            'status': 'working' if CLUSTERING_AVAILABLE else 'not_available',
            'sklearn_available': CLUSTERING_AVAILABLE
        }
