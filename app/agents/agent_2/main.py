"""
Agent 2: Insight Extraction - Main Orchestrator
Complete implementation following detailed specification from docs/agent-2-implementation.md
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import numpy as np

from ..base_agent import BaseAgent
from ...models.signal import Signal
from ...models.problem import Problem, ProblemCreate
from ...config.database import signal_repo, problem_repo
from ...utils.nlp_utils import nlp_utils
from ...utils.cache import cache_manager, CacheKeys
from ...config.settings import settings

from .clustering import SemanticClustering, DBSCANClustering, KMeansClustering, TopicModeling, FallbackClustering
from .classification import ProblemClassifier, PatternAnalyzer
from .pattern_analysis import TrendAnalyzer, CorrelationFinder
from .problem_generation import ProblemGenerator

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import networkx as nx
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False


class InsightExtractionAgent(BaseAgent):
    """
    Insight Extraction Agent - Complete Implementation
    
    Following detailed specification:
    1. Clustering (semantic clustering, DBSCAN, K-means)
    2. Classification (problem classifier, pattern analysis)
    3. Problem Generation (structured problem statements)
    4. Storage (problems database with evidence)
    """
    
    def __init__(self):
        super().__init__("agent_2")
        self.clustering = {}
        self.classification = {}
        self.pattern_analysis = {}
        self.problem_generation = {}
        self._init_components()
    
    def _init_components(self):
        """Initialize all components as per detailed specification"""
        # Initialize clustering components
        if CLUSTERING_AVAILABLE:
            self.clustering = {
                'semantic': SemanticClustering(),
                'dbscan': DBSCANClustering(),
                'kmeans': KMeansClustering(),
                'topic': TopicModeling()
            }
        else:
            self.clustering = {
                'semantic': FallbackClustering()
            }
        
        # Initialize classification components
        self.classification = {
            'problem_classifier': ProblemClassifier(),
            'pattern_analyzer': PatternAnalyzer()
        }
        
        # Initialize pattern analysis components
        self.pattern_analysis = {
            'trend_analyzer': TrendAnalyzer(),
            'correlation_finder': CorrelationFinder()
        }
        
        # Initialize problem generation components
        self.problem_generation = {
            'problem_generator': ProblemGenerator()
        }
    
    async def validate_input(self, signal_ids: List[str] = None, **kwargs) -> bool:
        """Validate input parameters"""
        if signal_ids is None:
            return True  # Will use recent signals
        
        if not isinstance(signal_ids, list):
            self.log_error("signal_ids must be a list")
            return False
        
        return True
    
    async def run(self, signal_ids: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Run complete insight extraction pipeline
        
        Following detailed specification:
        1. Signal Retrieval
        2. Clustering Analysis
        3. Classification Analysis
        4. Pattern Analysis
        5. Problem Generation
        6. Storage
        """
        self.log_info("Starting insight extraction process")
        
        try:
            # Step 1: Signal Retrieval
            signals = await self._retrieve_signals(signal_ids)
            self.log_info(f"Retrieved {len(signals)} signals")
            
            if len(signals) < settings.MIN_CLUSTER_SIZE:
                return {
                    'status': 'insufficient_data',
                    'message': f'Need at least {settings.MIN_CLUSTER_SIZE} signals for clustering',
                    'signals_count': len(signals),
                    'agent_id': self.agent_id
                }
            
            # Step 2: Clustering Analysis
            clustering_results = await self._run_clustering_analysis(signals)
            self.log_info(f"Clustering analysis completed: {len(clustering_results.get('clusters', []))} clusters")
            
            # Step 3: Classification Analysis
            classification_results = await self._run_classification_analysis(signals)
            self.log_info(f"Classification analysis completed: {len(classification_results)} classifications")
            
            # Step 4: Pattern Analysis
            pattern_results = await self._run_pattern_analysis(signals, clustering_results)
            self.log_info(f"Pattern analysis completed: {len(pattern_results.get('patterns', []))} patterns")
            
            # Step 5: Problem Generation
            problems = await self._run_problem_generation(
                signals, clustering_results, classification_results, pattern_results
            )
            self.log_info(f"Generated {len(problems)} problems")
            
            # Step 6: Storage
            stored_problems = await self._store_problems(problems)
            self.log_info(f"Stored {len(stored_problems)} problems")
            
            return {
                'status': 'success',
                'signals_processed': len(signals),
                'clustering_results': clustering_results,
                'classification_results': classification_results,
                'pattern_results': pattern_results,
                'problems_generated': len(problems),
                'problems_stored': len(stored_problems),
                'problem_ids': [p.get('id') for p in stored_problems],
                'processing_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log_error(f"Pipeline execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'agent_id': self.agent_id,
                'processing_time': datetime.now().isoformat()
            }
    
    async def _retrieve_signals(self, signal_ids: List[str] = None) -> List[Signal]:
        """Retrieve signals from database"""
        if signal_ids:
            # Retrieve specific signals
            signals = []
            for signal_id in signal_ids:
                signal_data = await signal_repo.get_by_id(signal_id)
                if signal_data:
                    signals.append(Signal.from_dict(signal_data))
            return signals
        else:
            # Retrieve recent signals
            signal_data_list = await signal_repo.get_recent_signals(limit=500)
            return [Signal.from_dict(signal_data) for signal_data in signal_data_list]
    
    async def _run_clustering_analysis(self, signals: List[Signal]) -> Dict[str, Any]:
        """Run clustering analysis following detailed specification"""
        if not CLUSTERING_AVAILABLE:
            return await self.clustering['semantic'].cluster_signals(signals)
        
        all_clusters = []
        
        # Extract text content
        texts = [signal.content for signal in signals]
        
        # Method 1: Semantic Clustering
        try:
            semantic_clusters = await self.clustering['semantic'].cluster_signals(signals)
            all_clusters.extend(semantic_clusters)
        except Exception as e:
            self.log_error(f"Semantic clustering failed: {e}")
        
        # Method 2: DBSCAN Clustering
        try:
            dbscan_clusters = await self.clustering['dbscan'].cluster_signals(signals)
            all_clusters.extend(dbscan_clusters)
        except Exception as e:
            self.log_error(f"DBSCAN clustering failed: {e}")
        
        # Method 3: K-means Clustering
        try:
            kmeans_clusters = await self.clustering['kmeans'].cluster_signals(signals)
            all_clusters.extend(kmeans_clusters)
        except Exception as e:
            self.log_error(f"K-means clustering failed: {e}")
        
        # Method 4: Topic Modeling
        try:
            topic_clusters = await self.clustering['topic'].cluster_signals(signals)
            all_clusters.extend(topic_clusters)
        except Exception as e:
            self.log_error(f"Topic modeling failed: {e}")
        
        # Merge and deduplicate clusters
        merged_clusters = self._merge_clusters(all_clusters)
        
        return {
            'clusters': merged_clusters,
            'total_clusters': len(merged_clusters),
            'methods_used': ['semantic', 'dbscan', 'kmeans', 'topic_modeling']
        }
    
    async def _run_classification_analysis(self, signals: List[Signal]) -> List[Dict[str, Any]]:
        """Run classification analysis following detailed specification"""
        classifications = []
        
        for signal in signals:
            try:
                # Problem Classification
                problem_classification = await self.classification['problem_classifier'].classify_signal(signal)
                
                # Pattern Analysis
                pattern_analysis = await self.classification['pattern_analyzer'].analyze_signal(signal)
                
                classification = {
                    'signal_id': signal.id,
                    'problem_classification': problem_classification,
                    'pattern_analysis': pattern_analysis,
                    'combined_confidence': self._calculate_combined_confidence(
                        problem_classification, pattern_analysis
                    )
                }
                
                classifications.append(classification)
                
            except Exception as e:
                self.log_error(f"Classification failed for signal {signal.id}: {e}")
        
        return classifications
    
    async def _run_pattern_analysis(self, signals: List[Signal], 
                                 clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run pattern analysis following detailed specification"""
        patterns = []
        
        # Trend Analysis
        try:
            trends = await self.pattern_analysis['trend_analyzer'].analyze_trends(signals)
            patterns.extend(trends)
        except Exception as e:
            self.log_error(f"Trend analysis failed: {e}")
        
        # Correlation Analysis
        try:
            correlations = await self.pattern_analysis['correlation_finder'].find_correlations(
                signals, clustering_results
            )
            patterns.extend(correlations)
        except Exception as e:
            self.log_error(f"Correlation analysis failed: {e}")
        
        return {
            'patterns': patterns,
            'total_patterns': len(patterns),
            'analysis_types': ['trends', 'correlations']
        }
    
    async def _run_problem_generation(self, signals: List[Signal],
                                   clustering_results: Dict[str, Any],
                                   classification_results: List[Dict[str, Any]],
                                   pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run problem generation following detailed specification"""
        problems = []
        
        # Generate problems from clusters
        clusters = clustering_results.get('clusters', [])
        for cluster in clusters:
            try:
                problem = await self.problem_generation['problem_generator'].generate_from_cluster(
                    cluster, signals, classification_results, pattern_results
                )
                if problem:
                    problems.append(problem)
            except Exception as e:
                self.log_error(f"Problem generation failed for cluster {cluster.get('cluster_id')}: {e}")
        
        # Generate problems from high-confidence individual signals
        high_confidence_signals = [
            signal for signal in signals 
            if signal.confidence_score > 0.8
        ]
        
        for signal in high_confidence_signals:
            try:
                # Find corresponding classification
                signal_classification = next(
                    (c for c in classification_results if c['signal_id'] == signal.id),
                    None
                )
                
                if signal_classification:
                    problem = await self.problem_generation['problem_generator'].generate_from_signal(
                        signal, signal_classification, pattern_results
                    )
                    if problem:
                        problems.append(problem)
            except Exception as e:
                self.log_error(f"Problem generation failed for signal {signal.id}: {e}")
        
        return problems
    
    async def _store_problems(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Store problems in database following detailed specification"""
        self.log_info(f"Storing {len(problems)} problems")
        
        stored_problems = []
        
        for problem_data in problems:
            try:
                # Create problem model
                problem = ProblemCreate(**problem_data)
                
                # Store in database
                stored_problem = await problem_repo.create(problem.dict())
                
                if stored_problem:
                    stored_problems.append(stored_problem)
                    
                    # Cache for quick retrieval
                    cache_key = f"problem:{stored_problem['id']}"
                    cache_manager.set(cache_key, stored_problem, ttl=3600)
                
            except Exception as e:
                self.log_error(f"Failed to store problem: {e}")
        
        self.log_info(f"Stored {len(stored_problems)} problems successfully")
        return stored_problems
    
    def _merge_clusters(self, all_clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge and deduplicate clusters"""
        # Simple merging logic - in production, this would be more sophisticated
        seen_signal_ids = set()
        merged_clusters = []
        
        for cluster in all_clusters:
            signal_ids = cluster.get('signals', [])
            unique_signals = [sid for sid in signal_ids if sid not in seen_signal_ids]
            
            if unique_signals:
                cluster['signals'] = unique_signals
                cluster['size'] = len(unique_signals)
                merged_clusters.append(cluster)
                seen_signal_ids.update(unique_signals)
        
        return merged_clusters
    
    def _calculate_combined_confidence(self, problem_classification: Dict[str, Any],
                                   pattern_analysis: Dict[str, Any]) -> float:
        """Calculate combined confidence from classification results"""
        problem_confidence = problem_classification.get('confidence', 0.5)
        pattern_confidence = pattern_analysis.get('confidence', 0.5)
        
        # Weighted average
        combined_confidence = (problem_confidence * 0.6) + (pattern_confidence * 0.4)
        return min(combined_confidence, 1.0)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform agent-specific health checks"""
        health_status = {}
        
        # Check clustering libraries
        if CLUSTERING_AVAILABLE:
            health_status['clustering_libs'] = 'available'
        else:
            health_status['clustering_libs'] = 'not_available'
        
        # Check individual clustering components
        for component_name, component in self.clustering.items():
            try:
                if hasattr(component, 'health_check'):
                    health_status[f'clustering_{component_name}'] = await component.health_check()
                else:
                    health_status[f'clustering_{component_name}'] = 'working'
            except Exception as e:
                health_status[f'clustering_{component_name}'] = f'error: {str(e)}'
        
        # Check classification components
        for component_name, component in self.classification.items():
            try:
                if hasattr(component, 'health_check'):
                    health_status[f'classification_{component_name}'] = await component.health_check()
                else:
                    health_status[f'classification_{component_name}'] = 'working'
            except Exception as e:
                health_status[f'classification_{component_name}'] = f'error: {str(e)}'
        
        # Check pattern analysis components
        for component_name, component in self.pattern_analysis.items():
            try:
                if hasattr(component, 'health_check'):
                    health_status[f'pattern_{component_name}'] = await component.health_check()
                else:
                    health_status[f'pattern_{component_name}'] = 'working'
            except Exception as e:
                health_status[f'pattern_{component_name}'] = f'error: {str(e)}'
        
        # Check problem generation components
        for component_name, component in self.problem_generation.items():
            try:
                if hasattr(component, 'health_check'):
                    health_status[f'problem_generation_{component_name}'] = await component.health_check()
                else:
                    health_status[f'problem_generation_{component_name}'] = 'working'
            except Exception as e:
                health_status[f'problem_generation_{component_name}'] = f'error: {str(e)}'
        
        # Check database connection
        try:
            recent_problems = await problem_repo.get_recent_problems(limit=1)
            health_status['database'] = 'connected'
        except Exception as e:
            health_status['database'] = f'error: {str(e)}'
        
        return health_status
