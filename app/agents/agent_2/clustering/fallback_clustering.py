"""
Fallback Clustering Implementation
Simple keyword-based clustering for when ML libraries are not available
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from ....utils.nlp_utils import nlp_utils
from ....config.settings import settings


class FallbackClustering:
    """Fallback clustering for when ML libraries are not available"""
    
    async def cluster_signals(self, signals: List) -> List[Dict[str, Any]]:
        """Simple keyword-based clustering"""
        clusters = {}
        
        for signal in signals:
            keywords = nlp_utils.extract_keywords(signal.content, max_keywords=3)
            if keywords:
                cluster_key = keywords[0]  # Use first keyword as cluster key
                if cluster_key not in clusters:
                    clusters[cluster_key] = []
                clusters[cluster_key].append(signal)
        
        # Create cluster objects
        cluster_objects = []
        for cluster_id, cluster_signals in clusters.items():
            if len(cluster_signals) >= settings.MIN_CLUSTER_SIZE:
                cluster_objects.append({
                    'cluster_id': f"keyword_cluster_{cluster_id}",
                    'cluster_type': 'keyword_based',
                    'size': len(cluster_signals),
                    'signals': [signal.id for signal in cluster_signals],
                    'keywords': [cluster_id],
                    'avg_confidence': np.mean([signal.confidence_score for signal in cluster_signals]),
                    'source_distribution': self._calculate_source_distribution(cluster_signals),
                    'created_at': datetime.now().isoformat()
                })
        
        return cluster_objects
    
    def _calculate_source_distribution(self, signals: List) -> Dict[str, float]:
        """Calculate distribution of source types"""
        source_counts = {}
        for signal in signals:
            source_type = signal.source_type
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        total = len(signals)
        return {source: count/total for source, count in source_counts.items()}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for fallback clustering"""
        return {
            'status': 'working',
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords'),
            'fallback_mode': True
        }
