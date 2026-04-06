"""
DBSCAN Clustering Implementation
Performs density-based clustering using DBSCAN algorithm
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from ....utils.nlp_utils import nlp_utils
from ....config.settings import settings

try:
    from sklearn.cluster import DBSCAN
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False


class DBSCANClustering:
    """DBSCAN clustering implementation following detailed specification"""
    
    async def cluster_signals(self, signals: List) -> List[Dict[str, Any]]:
        """Perform DBSCAN clustering"""
        if not CLUSTERING_AVAILABLE or not signals:
            return []
        
        try:
            texts = [signal.content for signal in signals]
            embeddings = nlp_utils.encode_sentences(texts)
            
            eps = 1 - settings.SIMILARITY_THRESHOLD
            dbscan = DBSCAN(eps=eps, min_samples=settings.MIN_CLUSTER_SIZE, metric='cosine')
            cluster_labels = dbscan.fit_predict(embeddings)
            
            # Create clusters
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label != -1:
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(signals[i])
            
            cluster_objects = []
            for cluster_id, cluster_signals in clusters.items():
                if len(cluster_signals) >= settings.MIN_CLUSTER_SIZE:
                    cluster_objects.append({
                        'cluster_id': f"dbscan_cluster_{cluster_id}",
                        'cluster_type': 'dbscan',
                        'size': len(cluster_signals),
                        'signals': [signal.id for signal in cluster_signals],
                        'avg_confidence': np.mean([signal.confidence_score for signal in cluster_signals]),
                        'source_distribution': self._calculate_source_distribution(cluster_signals),
                        'created_at': datetime.now().isoformat()
                    })
            
            return cluster_objects
            
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")
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
        """Health check for DBSCAN clustering"""
        return {
            'status': 'working' if CLUSTERING_AVAILABLE else 'not_available',
            'sklearn_available': CLUSTERING_AVAILABLE
        }
