"""
Correlation Finder Implementation
Finds correlations between signals including source types and content
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from ....utils.nlp_utils import nlp_utils


class CorrelationFinder:
    """Correlation finder implementation following detailed specification"""
    
    async def find_correlations(self, signals: List, 
                            clustering_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find correlations between signals"""
        correlations = []
        
        # Source type correlations
        source_correlations = self._analyze_source_correlations(signals)
        correlations.extend(source_correlations)
        
        # Content correlations
        content_correlations = self._analyze_content_correlations(signals)
        correlations.extend(content_correlations)
        
        # Cluster correlations
        if clustering_results.get('clusters'):
            cluster_correlations = self._analyze_cluster_correlations(
                signals, clustering_results['clusters']
            )
            correlations.extend(cluster_correlations)
        
        return correlations
    
    def _analyze_source_correlations(self, signals: List) -> List[Dict[str, Any]]:
        """Analyze correlations between source types"""
        source_types = {}
        for signal in signals:
            source_type = signal.source_type
            source_types[source_type] = source_types.get(source_type, 0) + 1
        
        # Find correlations
        correlations = []
        for source_type, count in source_types.items():
            if count > 1:
                correlations.append({
                    'correlation_type': 'source_type',
                    'source': source_type,
                    'frequency': count,
                    'correlation_strength': count / len(signals)
                })
        
        return correlations
    
    def _analyze_content_correlations(self, signals: List) -> List[Dict[str, Any]]:
        """Analyze content correlations"""
        # Simple keyword co-occurrence analysis
        all_keywords = []
        for signal in signals:
            keywords = nlp_utils.extract_keywords(signal.content, max_keywords=3)
            all_keywords.append(keywords)
        
        # Find co-occurring keywords
        co_occurrences = {}
        for keyword_list in all_keywords:
            for i, keyword in enumerate(keyword_list):
                for other_keyword in keyword_list[i+1:]:
                    pair = tuple(sorted([keyword, other_keyword]))
                    co_occurrences[pair] = co_occurrences.get(pair, 0) + 1
        
        correlations = []
        for (keyword1, keyword2), count in co_occurrences.items():
            if count > 1:
                correlations.append({
                    'correlation_type': 'content_co_occurrence',
                    'keywords': [keyword1, keyword2],
                    'co_occurrence_count': count,
                    'correlation_strength': count / len(all_keywords)
                })
        
        return correlations
    
    def _analyze_cluster_correlations(self, signals: List, 
                                 clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze correlations between clusters"""
        correlations = []
        
        # Size correlations
        sizes = [cluster.get('size', 0) for cluster in clusters]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        
        for cluster in clusters:
            cluster_size = cluster.get('size', 0)
            if cluster_size > avg_size * 1.5:  # Significantly larger than average
                correlations.append({
                    'correlation_type': 'cluster_size',
                    'cluster_id': cluster.get('cluster_id'),
                    'size': cluster_size,
                    'size_deviation': (cluster_size - avg_size) / avg_size if avg_size > 0 else 0
                })
        
        return correlations
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for correlation finder"""
        return {
            'status': 'working',
            'correlation_types': ['source_type', 'content_co_occurrence', 'cluster_size'],
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
