"""
Semantic Clustering Implementation
Performs semantic clustering using embeddings and similarity metrics
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


class SemanticClustering:
    """Semantic clustering implementation following detailed specification"""
    
    async def cluster_signals(self, signals: List) -> List[Dict[str, Any]]:
        """Perform semantic clustering on signals"""
        if not signals:
            return []
        
        try:
            # Generate embeddings
            texts = [signal.content for signal in signals]
            embeddings = nlp_utils.encode_sentences(texts)
            
            # Perform clustering
            eps = 1 - settings.SIMILARITY_THRESHOLD
            dbscan = DBSCAN(eps=eps, min_samples=settings.MIN_CLUSTER_SIZE, metric='cosine')
            cluster_labels = dbscan.fit_predict(embeddings)
            
            # Group signals by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Not noise
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(signals[i])
            
            # Create cluster objects
            cluster_objects = []
            for cluster_id, cluster_signals in clusters.items():
                if len(cluster_signals) >= settings.MIN_CLUSTER_SIZE:
                    cluster_obj = await self._create_cluster_object(
                        cluster_id, cluster_signals, embeddings[[i for i, s in enumerate(signals) if s in cluster_signals]]
                    )
                    cluster_objects.append(cluster_obj)
            
            return cluster_objects
            
        except Exception as e:
            print(f"Semantic clustering failed: {e}")
            return []
    
    async def _create_cluster_object(self, cluster_id: int, cluster_signals: List, 
                                  cluster_embeddings: np.ndarray) -> Dict[str, Any]:
        """Create cluster object with metadata"""
        # Calculate cluster centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Find most representative signal (closest to centroid)
        similarities = [
            nlp_utils.calculate_similarity(cluster_embeddings[i], centroid)
            for i in range(len(cluster_signals))
        ]
        representative_idx = np.argmax(similarities)
        representative_signal = cluster_signals[representative_idx]
        
        # Extract common keywords
        all_text = " ".join([signal.content for signal in cluster_signals])
        keywords = nlp_utils.extract_keywords(all_text, max_keywords=5)
        
        return {
            'cluster_id': f"semantic_cluster_{cluster_id}",
            'cluster_type': 'semantic',
            'size': len(cluster_signals),
            'signals': [signal.id for signal in cluster_signals],
            'representative_signal': {
                'id': representative_signal.id,
                'content': representative_signal.content[:200] + "..."
            },
            'keywords': keywords,
            'centroid': centroid.tolist(),
            'avg_confidence': np.mean([signal.confidence_score for signal in cluster_signals]),
            'source_distribution': self._calculate_source_distribution(cluster_signals),
            'created_at': datetime.now().isoformat()
        }
    
    def _calculate_source_distribution(self, signals: List) -> Dict[str, float]:
        """Calculate distribution of source types"""
        source_counts = {}
        for signal in signals:
            source_type = signal.source_type
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        total = len(signals)
        return {source: count/total for source, count in source_counts.items()}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for semantic clustering"""
        return {
            'status': 'working',
            'nlp_utils_available': hasattr(nlp_utils, 'encode_sentences'),
            'clustering_available': CLUSTERING_AVAILABLE
        }
