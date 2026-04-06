"""
Emergent Themes Implementation
Identifies emergent themes from problem patterns
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from .....utils.nlp_utils import nlp_utils
from .....config.settings import settings

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    THEMES_AVAILABLE = True
except ImportError:
    THEMES_AVAILABLE = False


class EmergentThemes:
    """Emergent themes implementation following detailed specification"""
    
    async def identify_emergent_themes(self, problems: List, 
                                       graph_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify emergent themes from problem patterns"""
        if not problems:
            return []
        
        try:
            # Extract all problem statements
            problem_texts = [problem.problem_statement for problem in problems]
            
            # Perform topic modeling to find themes
            themes = await self._perform_topic_modeling(problem_texts)
            
            # Analyze theme evolution
            evolution_patterns = await self._analyze_theme_evolution(problems)
            
            # Identify theme clusters
            theme_clusters = await self._identify_theme_clusters(themes, problems)
            
            # Combine all theme analyses
            emergent_themes = []
            for theme in themes:
                theme['emergence_type'] = 'topic_modeling'
                theme['evidence'] = self._collect_theme_evidence(theme, problems)
                theme['emergence_strength'] = self._calculate_emergence_strength(theme, problems)
                emergent_themes.append(theme)
            
            # Add evolution patterns
            for pattern in evolution_patterns:
                pattern['emergence_type'] = 'temporal_evolution'
                emergent_themes.append(pattern)
            
            # Add theme clusters
            for cluster in theme_clusters:
                cluster['emergence_type'] = 'thematic_clustering'
                emergent_themes.append(cluster)
            
            return emergent_themes
            
        except Exception as e:
            print(f"Emergent themes identification failed: {e}")
            return []
    
    async def _perform_topic_modeling(self, problem_texts: List[str]) -> List[Dict[str, Any]]:
        """Perform topic modeling to identify themes"""
        if not THEMES_AVAILABLE:
            return await self._fallback_topic_identification(problem_texts)
        
        try:
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)  # Include bigrams
            )
            tfidf_matrix = vectorizer.fit_transform(problem_texts)
            
            # Perform LDA
            n_topics = min(len(problem_texts) // 3, 8)  # One topic per 3 problems, max 8
            n_topics = max(2, n_topics)
            
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )
            lda.fit(tfidf_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            themes = []
            
            for topic_idx, topic in enumerate(lda.components_):
                # Get top words for this topic
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                
                # Calculate topic coherence
                coherence = self._calculate_topic_coherence(top_words, problem_texts)
                
                themes.append({
                    'theme_id': f"theme_{topic_idx}",
                    'top_words': top_words,
                    'coherence_score': coherence,
                    'topic_weight': np.mean(topic),
                    'emergence_method': 'lda_topic_modeling'
                })
            
            return themes
            
        except Exception as e:
            print(f"Topic modeling failed: {e}")
            return []
    
    async def _fallback_topic_identification(self, problem_texts: List[str]) -> List[Dict[str, Any]]:
        """Fallback topic identification using keyword frequency"""
        # Simple keyword frequency analysis
        all_words = []
        for text in problem_texts:
            words = nlp_utils.extract_keywords(text, max_keywords=10)
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Identify high-frequency themes
        themes = []
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            if count >= 2:  # Appears in at least 2 problems
                themes.append({
                    'theme_id': f"theme_{word}",
                    'top_words': [word],
                    'coherence_score': count / len(problem_texts),
                    'topic_weight': count,
                    'emergence_method': 'frequency_analysis'
                })
        
        return themes
    
    async def _analyze_theme_evolution(self, problems: List) -> List[Dict[str, Any]]:
        """Analyze how themes evolve over time"""
        evolution_patterns = []
        
        # Sort problems by creation time
        time_sorted_problems = sorted(
            [p for p in problems if hasattr(p, 'created_at')],
            key=lambda x: x.created_at
        )
        
        if len(time_sorted_problems) < 3:
            return evolution_patterns
        
        # Analyze temporal windows
        window_size = max(3, len(time_sorted_problems) // 3)
        
        for i in range(len(time_sorted_problems) - window_size + 1):
            window = time_sorted_problems[i:i+window_size]
            
            # Extract themes from window
            window_texts = [p.problem_statement for p in window]
            window_keywords = self._extract_window_keywords(window_texts)
            
            # Calculate theme drift
            if i > 0:
                prev_keywords = self._extract_window_keywords(
                    [p.problem_statement for p in time_sorted_problems[i-window_size:i]]
                )
                drift = self._calculate_theme_drift(prev_keywords, window_keywords)
                
                evolution_patterns.append({
                    'pattern_type': 'theme_drift',
                    'time_window': f"{i}-{i+window_size}",
                    'dominant_themes': list(window_keywords)[:5],
                    'theme_drift': drift,
                    'drift_direction': self._calculate_drift_direction(prev_keywords, window_keywords)
                })
        
        return evolution_patterns
    
    def _extract_window_keywords(self, problem_texts: List[str]) -> List[str]:
        """Extract keywords from a time window"""
        all_keywords = []
        for text in problem_texts:
            keywords = nlp_utils.extract_keywords(text, max_keywords=5)
            all_keywords.extend(keywords)
        
        # Count and return top keywords
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        return [kw for kw, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    def _calculate_theme_drift(self, prev_keywords: List[str], curr_keywords: List[str]) -> float:
        """Calculate theme drift between time windows"""
        prev_set = set(prev_keywords)
        curr_set = set(curr_keywords)
        
        if not prev_set or not curr_set:
            return 0.0
        
        # Calculate Jaccard distance
        intersection = len(prev_set.intersection(curr_set))
        union = len(prev_set.union(curr_set))
        
        jaccard_similarity = intersection / union if union else 0
        drift = 1 - jaccard_similarity
        
        return drift
    
    def _calculate_drift_direction(self, prev_keywords: List[str], curr_keywords: List[str]) -> str:
        """Calculate direction of theme drift"""
        prev_set = set(prev_keywords)
        curr_set = set(curr_keywords)
        
        # New themes appearing
        new_themes = curr_set - prev_set
        if new_themes:
            return 'emerging'
        
        # Old themes disappearing
        disappearing_themes = prev_set - curr_set
        if disappearing_themes:
            return 'declining'
        
        return 'stable'
    
    async def _identify_theme_clusters(self, themes: List[Dict[str, Any]], problems: List) -> List[Dict[str, Any]]:
        """Identify clusters of related themes"""
        if not themes:
            return []
        
        # Create theme similarity matrix
        theme_similarities = self._calculate_theme_similarities(themes)
        
        # Cluster themes using similarity
        clusters = self._cluster_themes(theme_similarities)
        
        # Enhance clusters with problem evidence
        for cluster in clusters:
            cluster['evidence'] = self._collect_cluster_evidence(cluster, themes, problems)
            cluster['cluster_strength'] = self._calculate_cluster_strength(cluster, problems)
        
        return clusters
    
    def _calculate_theme_similarities(self, themes: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate similarities between themes"""
        similarities = {}
        
        for i, theme1 in enumerate(themes):
            similarities[theme1['theme_id']] = {}
            for theme2 in themes[i+1:]:
                similarity = self._calculate_theme_similarity(theme1, theme2)
                similarities[theme1['theme_id']][theme2['theme_id']] = similarity
        
        return similarities
    
    def _calculate_theme_similarity(self, theme1: Dict[str, Any], theme2: Dict[str, Any]) -> float:
        """Calculate similarity between two themes"""
        words1 = set(theme1.get('top_words', []))
        words2 = set(theme2.get('top_words', []))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union else 0.0
    
    def _cluster_themes(self, theme_similarities: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Cluster themes based on similarities"""
        # Simple threshold-based clustering
        threshold = 0.3  # Similarity threshold for clustering
        clusters = []
        processed_themes = set()
        
        for theme_id, similarities in theme_similarities.items():
            if theme_id in processed_themes:
                continue
            
            # Find similar themes
            similar_themes = [
                other_id for other_id, similarity in similarities.items()
                if similarity > threshold
            ]
            
            if similar_themes:
                cluster = {
                    'pattern_type': 'theme_cluster',
                    'theme_ids': [theme_id] + similar_themes,
                    'cluster_size': len(similar_themes) + 1,
                    'avg_similarity': np.mean([similarities[tid] for tid in similar_themes])
                }
                clusters.append(cluster)
                processed_themes.update([theme_id] + similar_themes)
        
        return clusters
    
    def _calculate_topic_coherence(self, top_words: List[str], problem_texts: List[str]) -> float:
        """Calculate coherence score for a topic"""
        if not top_words or not problem_texts:
            return 0.0
        
        # Count co-occurrences of top words in problem texts
        co_occurrence_counts = []
        for word in top_words:
            count = sum(1 for text in problem_texts if word in text.lower())
            co_occurrence_counts.append(count)
        
        # Coherence is average co-occurrence
        coherence = np.mean(co_occurrence_counts) / len(problem_texts)
        return min(coherence, 1.0)
    
    def _collect_theme_evidence(self, theme: Dict[str, Any], problems: List) -> Dict[str, Any]:
        """Collect evidence for a theme"""
        theme_words = set(theme.get('top_words', []))
        evidence_problems = []
        
        for problem in problems:
            problem_words = set(nlp_utils.extract_keywords(problem.problem_statement, max_keywords=10))
            if theme_words.intersection(problem_words):
                evidence_problems.append({
                    'problem_id': problem.id,
                    'relevance_score': len(theme_words.intersection(problem_words)) / len(theme_words),
                    'problem_statement': problem.problem_statement
                })
        
        return {
            'supporting_problems': evidence_problems,
            'evidence_count': len(evidence_problems),
            'avg_relevance': np.mean([p['relevance_score'] for p in evidence_problems]) if evidence_problems else 0
        }
    
    def _collect_cluster_evidence(self, cluster: Dict[str, Any], themes: List[Dict[str, Any]], problems: List) -> Dict[str, Any]:
        """Collect evidence for a theme cluster"""
        theme_ids = set(cluster.get('theme_ids', []))
        evidence_themes = []
        
        for theme in themes:
            if theme['theme_id'] in theme_ids:
                theme_evidence = self._collect_theme_evidence(theme, problems)
                evidence_themes.append(theme_evidence)
        
        # Combine evidence from all themes in cluster
        all_evidence_problems = []
        for theme_evidence in evidence_themes:
            all_evidence_problems.extend(theme_evidence.get('supporting_problems', []))
        
        return {
            'supporting_themes': evidence_themes,
            'total_evidence_problems': len(set(p['problem_id'] for p in all_evidence_problems)),
            'cluster_coherence': np.mean([t.get('coherence_score', 0) for t in evidence_themes])
        }
    
    def _calculate_cluster_strength(self, cluster: Dict[str, Any], problems: List) -> float:
        """Calculate strength of a theme cluster"""
        cluster_size = cluster.get('cluster_size', 0)
        avg_similarity = cluster.get('avg_similarity', 0.0)
        
        # Strength based on size and internal similarity
        size_score = min(cluster_size / len(problems), 0.5)
        similarity_score = avg_similarity * 0.5
        
        return size_score + similarity_score
    
    def _calculate_emergence_strength(self, theme: Dict[str, Any], problems: List) -> float:
        """Calculate emergence strength of a theme"""
        coherence = theme.get('coherence_score', 0.0)
        topic_weight = theme.get('topic_weight', 0.0)
        
        # Normalize and combine
        normalized_coherence = min(coherence, 1.0)
        normalized_weight = min(topic_weight / len(problems), 1.0)
        
        return (normalized_coherence * 0.6) + (normalized_weight * 0.4)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for emergent themes"""
        return {
            'status': 'working',
            'sklearn_available': THEMES_AVAILABLE,
            'identification_methods': [
                'lda_topic_modeling',
                'frequency_analysis',
                'temporal_evolution',
                'thematic_clustering'
            ],
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
