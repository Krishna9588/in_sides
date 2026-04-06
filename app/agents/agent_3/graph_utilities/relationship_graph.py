"""
Relationship Graph Implementation
Builds and manages signal-problem relationship graphs
"""
import asyncio
from typing import Dict, Any, List, Tuple
from datetime import datetime

from .....utils.nlp_utils import nlp_utils
from .....config.settings import settings

try:
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


class RelationshipGraph:
    """Relationship graph implementation following detailed specification"""
    
    def __init__(self):
        self.graph = None
        self.node_types = ['signal', 'problem', 'insight']
        self.edge_types = ['similarity', 'causality', 'correlation']
    
    async def build_relationship_graph(self, problems: List) -> Dict[str, Any]:
        """Build signal-problem relationship graph"""
        if not GRAPH_AVAILABLE:
            return await self._build_fallback_graph(problems)
        
        try:
            # Initialize graph
            self.graph = nx.DiGraph()
            
            # Add problem nodes
            for problem in problems:
                self.graph.add_node(
                    f"problem_{problem.id}",
                    type='problem',
                    data={
                        'id': problem.id,
                        'statement': problem.problem_statement,
                        'category': problem.problem_category,
                        'severity': problem.severity,
                        'confidence': problem.confidence_score
                    }
                )
            
            # Retrieve related signals
            related_signals = await self._get_related_signals(problems)
            
            # Add signal nodes and edges
            for signal in related_signals:
                self.graph.add_node(
                    f"signal_{signal.id}",
                    type='signal',
                    data={
                        'id': signal.id,
                        'content': signal.content,
                        'source_type': signal.source_type,
                        'confidence': signal.confidence_score
                    }
                )
                
                # Connect to related problems
                connected_problems = await self._find_connected_problems(signal, problems)
                for problem in connected_problems:
                    similarity = self._calculate_similarity(signal, problem)
                    if similarity > settings.SIMILARITY_THRESHOLD:
                        self.graph.add_edge(
                            f"signal_{signal.id}",
                            f"problem_{problem.id}",
                            type='similarity',
                            weight=similarity,
                            data={
                                'similarity_score': similarity,
                                'relationship_type': 'content_similarity'
                            }
                        )
            
            return {
                'graph': self.graph,
                'nodes': list(self.graph.nodes(data=True)),
                'edges': list(self.graph.edges(data=True)),
                'construction_metadata': {
                    'total_problems': len(problems),
                    'total_signals': len(related_signals),
                    'total_edges': self.graph.number_of_edges(),
                    'created_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"Graph construction failed: {e}")
            return {'graph': None, 'error': str(e)}
    
    async def _build_fallback_graph(self, problems: List) -> Dict[str, Any]:
        """Build fallback graph when NetworkX is not available"""
        # Simple adjacency list representation
        nodes = []
        edges = []
        
        # Add problem nodes
        for problem in problems:
            nodes.append({
                'id': f"problem_{problem.id}",
                'type': 'problem',
                'data': {
                    'id': problem.id,
                    'statement': problem.problem_statement,
                    'category': problem.problem_category,
                    'severity': problem.severity,
                    'confidence': problem.confidence_score
                }
            })
        
        # Simple keyword-based connections
        for i, problem1 in enumerate(problems):
            for problem2 in problems[i+1:]:
                similarity = self._calculate_keyword_similarity(problem1, problem2)
                if similarity > settings.SIMILARITY_THRESHOLD:
                    edges.append({
                        'source': f"problem_{problem1.id}",
                        'target': f"problem_{problem2.id}",
                        'type': 'similarity',
                        'weight': similarity,
                        'data': {
                            'similarity_score': similarity,
                            'relationship_type': 'keyword_similarity'
                        }
                    })
        
        return {
            'graph': {'nodes': nodes, 'edges': edges},
            'nodes': nodes,
            'edges': edges,
            'construction_metadata': {
                'total_problems': len(problems),
                'total_edges': len(edges),
                'method': 'fallback_keyword_based',
                'created_at': datetime.now().isoformat()
            }
        }
    
    async def _get_related_signals(self, problems: List) -> List:
        """Get signals related to problems"""
        # This would typically query the database for signals
        # For now, return empty list as placeholder
        return []
    
    async def _find_connected_problems(self, signal, problems: List) -> List:
        """Find problems connected to a signal"""
        connected = []
        for problem in problems:
            # Simple keyword matching for connection
            signal_keywords = set(nlp_utils.extract_keywords(signal.content, max_keywords=5))
            problem_keywords = set(nlp_utils.extract_keywords(problem.problem_statement, max_keywords=5))
            
            if signal_keywords.intersection(problem_keywords):
                connected.append(problem)
        
        return connected
    
    def _calculate_similarity(self, signal, problem) -> float:
        """Calculate similarity between signal and problem"""
        if not GRAPH_AVAILABLE:
            return self._calculate_keyword_similarity_light(signal, problem)
        
        try:
            # Use TF-IDF and cosine similarity
            texts = [signal.content, problem.problem_statement]
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return similarities[0][0] if len(similarities) > 0 else 0.0
            
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_keyword_similarity(self, problem1, problem2) -> float:
        """Calculate keyword similarity between problems"""
        keywords1 = set(nlp_utils.extract_keywords(problem1.problem_statement, max_keywords=5))
        keywords2 = set(nlp_utils.extract_keywords(problem2.problem_statement, max_keywords=5))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_keyword_similarity_light(self, signal, problem) -> float:
        """Lightweight keyword similarity calculation"""
        signal_keywords = set(nlp_utils.extract_keywords(signal.content, max_keywords=5))
        problem_keywords = set(nlp_utils.extract_keywords(problem.problem_statement, max_keywords=5))
        
        if not signal_keywords or not problem_keywords:
            return 0.0
        
        intersection = signal_keywords.intersection(problem_keywords)
        union = signal_keywords.union(problem_keywords)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for relationship graph"""
        return {
            'status': 'working',
            'networkx_available': GRAPH_AVAILABLE,
            'graph_constructed': self.graph is not None,
            'node_types_supported': self.node_types,
            'edge_types_supported': self.edge_types
        }
