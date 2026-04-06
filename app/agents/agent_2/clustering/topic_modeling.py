"""
Topic Modeling Implementation
Performs topic modeling using LDA (Latent Dirichlet Allocation)
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from ....utils.nlp_utils import nlp_utils
from ....config.settings import settings

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False


class TopicModeling:
    """Topic modeling implementation following detailed specification"""
    
    async def cluster_signals(self, signals: List) -> List[Dict[str, Any]]:
        """Perform topic modeling"""
        if not CLUSTERING_AVAILABLE or not signals:
            return []
        
        try:
            texts = [signal.content for signal in signals]
            
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Perform LDA
            n_topics = min(len(signals) // settings.MIN_CLUSTER_SIZE, 5)
            n_topics = max(2, n_topics)
            
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(tfidf_matrix)
            
            # Get topic distributions
            topic_distributions = lda.transform(tfidf_matrix)
            
            # Assign to topics
            clusters = {}
            for i, topic_dist in enumerate(topic_distributions):
                dominant_topic = np.argmax(topic_dist)
                if dominant_topic not in clusters:
                    clusters[dominant_topic] = []
                clusters[dominant_topic].append(signals[i])
            
            cluster_objects = []
            for topic_id, cluster_signals in clusters.items():
                if len(cluster_signals) >= settings.MIN_CLUSTER_SIZE:
                    # Get top words for this topic
                    feature_names = vectorizer.get_feature_names_out()
                    top_words_idx = lda.components_[topic_id].argsort()[-5:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    
                    cluster_objects.append({
                        'cluster_id': f"topic_{topic_id}",
                        'cluster_type': 'topic_model',
                        'size': len(cluster_signals),
                        'signals': [signal.id for signal in cluster_signals],
                        'top_words': top_words,
                        'avg_confidence': np.mean([signal.confidence_score for signal in cluster_signals]),
                        'source_distribution': self._calculate_source_distribution(cluster_signals),
                        'created_at': datetime.now().isoformat()
                    })
            
            return cluster_objects
            
        except Exception as e:
            print(f"Topic modeling failed: {e}")
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
        """Health check for topic modeling"""
        return {
            'status': 'working' if CLUSTERING_AVAILABLE else 'not_available',
            'sklearn_available': CLUSTERING_AVAILABLE
        }
