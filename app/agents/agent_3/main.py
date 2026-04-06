"""
Agent 3: Research Synthesis - Main Orchestrator
Complete implementation following detailed specification from docs/agent-3-implementation.md
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import numpy as np

from ..base_agent import BaseAgent
from ...models.signal import Signal
from ...models.problem import Problem
from ...models.insight import Insight, InsightCreate
from ...config.database import signal_repo, problem_repo, insight_repo
from ...utils.nlp_utils import nlp_utils
from ...utils.cache import cache_manager, CacheKeys
from ...config.settings import settings

from .graph_utilities import GraphUtilities
from .pattern_synthesis import PatternSynthesis
from .root_cause_analysis import RootCauseAnalysis
from .insight_generation import InsightGenerator

try:
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


class ResearchSynthesisAgent(BaseAgent):
    """
    Research Synthesis Agent - Complete Implementation
    
    Following detailed specification:
    1. Graph Construction (signal relationships, problem connections)
    2. Pattern Synthesis (cross-domain patterns, emergent themes)
    3. Root Cause Analysis (causal chains, underlying factors)
    4. Insight Generation (strategic insights, actionable recommendations)
    5. Storage (insights database with evidence)
    """
    
    def __init__(self):
        super().__init__("agent_3")
        self.graph_utilities = {}
        self.pattern_synthesis = {}
        self.root_cause_analysis = {}
        self.insight_generation = {}
        self._init_components()
    
    def _init_components(self):
        """Initialize all components as per detailed specification"""
        # Initialize graph utilities
        self.graph_utilities = GraphUtilities()
        
        # Initialize pattern synthesis
        self.pattern_synthesis = PatternSynthesis()
        
        # Initialize root cause analysis
        self.root_cause_analysis = RootCauseAnalysis()
        
        # Initialize insight generation
        self.insight_generation = InsightGenerator()
    
    async def validate_input(self, problem_ids: List[str] = None, **kwargs) -> bool:
        """Validate input parameters"""
        if problem_ids is None:
            return True  # Will use recent problems
        
        if not isinstance(problem_ids, list):
            self.log_error("problem_ids must be a list")
            return False
        
        return True
    
    async def run(self, problem_ids: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Run complete research synthesis pipeline
        
        Following detailed specification:
        1. Problem Retrieval
        2. Graph Construction
        3. Pattern Synthesis
        4. Root Cause Analysis
        5. Insight Generation
        6. Storage
        """
        self.log_info("Starting research synthesis process")
        
        try:
            # Step 1: Problem Retrieval
            problems = await self._retrieve_problems(problem_ids)
            self.log_info(f"Retrieved {len(problems)} problems")
            
            if len(problems) < settings.MIN_INSIGHT_THRESHOLD:
                return {
                    'status': 'insufficient_data',
                    'message': f'Need at least {settings.MIN_INSIGHT_THRESHOLD} problems for synthesis',
                    'problems_count': len(problems),
                    'agent_id': self.agent_id
                }
            
            # Step 2: Graph Construction
            graph_results = await self._run_graph_construction(problems)
            self.log_info(f"Graph construction completed: {len(graph_results.get('nodes', []))} nodes, {len(graph_results.get('edges', []))} edges")
            
            # Step 3: Pattern Synthesis
            pattern_results = await self._run_pattern_synthesis(problems, graph_results)
            self.log_info(f"Pattern synthesis completed: {len(pattern_results.get('patterns', []))} patterns")
            
            # Step 4: Root Cause Analysis
            root_cause_results = await self._run_root_cause_analysis(problems, graph_results, pattern_results)
            self.log_info(f"Root cause analysis completed: {len(root_cause_results.get('root_causes', []))} root causes")
            
            # Step 5: Insight Generation
            insights = await self._run_insight_generation(
                problems, graph_results, pattern_results, root_cause_results
            )
            self.log_info(f"Generated {len(insights)} insights")
            
            # Step 6: Storage
            stored_insights = await self._store_insights(insights)
            self.log_info(f"Stored {len(stored_insights)} insights")
            
            return {
                'status': 'success',
                'problems_processed': len(problems),
                'graph_results': graph_results,
                'pattern_results': pattern_results,
                'root_cause_results': root_cause_results,
                'insights_generated': len(insights),
                'insights_stored': len(stored_insights),
                'insight_ids': [i.get('id') for i in stored_insights],
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
    
    async def _retrieve_problems(self, problem_ids: List[str] = None) -> List[Problem]:
        """Retrieve problems from database"""
        if problem_ids:
            # Retrieve specific problems
            problems = []
            for problem_id in problem_ids:
                problem_data = await problem_repo.get_by_id(problem_id)
                if problem_data:
                    problems.append(Problem.from_dict(problem_data))
            return problems
        else:
            # Retrieve recent problems
            problem_data_list = await problem_repo.get_recent_problems(limit=100)
            return [Problem.from_dict(problem_data) for problem_data in problem_data_list]
    
    async def _run_graph_construction(self, problems: List[Problem]) -> Dict[str, Any]:
        """Run graph construction following detailed specification"""
        try:
            # Build signal-problem graph
            graph = await self.graph_utilities.build_relationship_graph(problems)
            
            # Analyze graph properties
            graph_analysis = await self.graph_utilities.analyze_graph_properties(graph)
            
            # Identify key nodes and connections
            key_nodes = await self.graph_utilities.identify_key_nodes(graph)
            
            return {
                'graph': graph,
                'nodes': graph_analysis.get('nodes', []),
                'edges': graph_analysis.get('edges', []),
                'graph_metrics': graph_analysis.get('metrics', {}),
                'key_nodes': key_nodes,
                'construction_method': 'signal_problem_relationships'
            }
            
        except Exception as e:
            self.log_error(f"Graph construction failed: {e}")
            return {'graph': None, 'nodes': [], 'edges': [], 'error': str(e)}
    
    async def _run_pattern_synthesis(self, problems: List[Problem], 
                                  graph_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run pattern synthesis following detailed specification"""
        try:
            # Synthesize cross-domain patterns
            cross_domain_patterns = await self.pattern_synthesis.synthesize_cross_domain_patterns(
                problems, graph_results
            )
            
            # Identify emergent themes
            emergent_themes = await self.pattern_synthesis.identify_emergent_themes(
                problems, graph_results
            )
            
            # Detect temporal patterns
            temporal_patterns = await self.pattern_synthesis.detect_temporal_patterns(problems)
            
            all_patterns = cross_domain_patterns + emergent_themes + temporal_patterns
            
            return {
                'patterns': all_patterns,
                'cross_domain_patterns': cross_domain_patterns,
                'emergent_themes': emergent_themes,
                'temporal_patterns': temporal_patterns,
                'total_patterns': len(all_patterns)
            }
            
        except Exception as e:
            self.log_error(f"Pattern synthesis failed: {e}")
            return {'patterns': [], 'error': str(e)}
    
    async def _run_root_cause_analysis(self, problems: List[Problem],
                                     graph_results: Dict[str, Any],
                                     pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run root cause analysis following detailed specification"""
        try:
            # Analyze causal chains
            causal_chains = await self.root_cause_analysis.analyze_causal_chains(
                problems, graph_results
            )
            
            # Identify underlying factors
            underlying_factors = await self.root_cause_analysis.identify_underlying_factors(
                problems, pattern_results
            )
            
            # Determine root causes
            root_causes = await self.root_cause_analysis.determine_root_causes(
                causal_chains, underlying_factors
            )
            
            return {
                'root_causes': root_causes,
                'causal_chains': causal_chains,
                'underlying_factors': underlying_factors,
                'analysis_confidence': self._calculate_analysis_confidence(root_causes)
            }
            
        except Exception as e:
            self.log_error(f"Root cause analysis failed: {e}")
            return {'root_causes': [], 'error': str(e)}
    
    async def _run_insight_generation(self, problems: List[Problem],
                                   graph_results: Dict[str, Any],
                                   pattern_results: Dict[str, Any],
                                   root_cause_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run insight generation following detailed specification"""
        insights = []
        
        try:
            # Generate strategic insights
            strategic_insights = await self.insight_generation.generate_strategic_insights(
                problems, graph_results, pattern_results, root_cause_results
            )
            
            # Generate actionable recommendations
            actionable_recommendations = await self.insight_generation.generate_actionable_recommendations(
                problems, root_cause_results
            )
            
            # Generate opportunity insights
            opportunity_insights = await self.insight_generation.generate_opportunity_insights(
                problems, pattern_results
            )
            
            insights = strategic_insights + actionable_recommendations + opportunity_insights
            
            # Rank insights by importance
            ranked_insights = await self.insight_generation.rank_insights(insights)
            
            return ranked_insights
            
        except Exception as e:
            self.log_error(f"Insight generation failed: {e}")
            return []
    
    async def _store_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Store insights in database following detailed specification"""
        self.log_info(f"Storing {len(insights)} insights")
        
        stored_insights = []
        
        for insight_data in insights:
            try:
                # Create insight model
                insight = InsightCreate(**insight_data)
                
                # Store in database
                stored_insight = await insight_repo.create(insight.dict())
                
                if stored_insight:
                    stored_insights.append(stored_insight)
                    
                    # Cache for quick retrieval
                    cache_key = f"insight:{stored_insight['id']}"
                    cache_manager.set(cache_key, stored_insight, ttl=3600)
                
            except Exception as e:
                self.log_error(f"Failed to store insight: {e}")
        
        self.log_info(f"Stored {len(stored_insights)} insights successfully")
        return stored_insights
    
    def _calculate_analysis_confidence(self, root_causes: List[Dict[str, Any]]) -> float:
        """Calculate confidence in analysis results"""
        if not root_causes:
            return 0.0
        
        # Average confidence of root causes
        confidences = [rc.get('confidence', 0.5) for rc in root_causes]
        return np.mean(confidences)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform agent-specific health checks"""
        health_status = {}
        
        # Check graph libraries
        if GRAPH_AVAILABLE:
            health_status['graph_libs'] = 'available'
        else:
            health_status['graph_libs'] = 'not_available'
        
        # Check individual components
        components = [
            ('graph_utilities', self.graph_utilities),
            ('pattern_synthesis', self.pattern_synthesis),
            ('root_cause_analysis', self.root_cause_analysis),
            ('insight_generation', self.insight_generation)
        ]
        
        for component_name, component in components:
            try:
                if hasattr(component, 'health_check'):
                    health_status[component_name] = await component.health_check()
                else:
                    health_status[component_name] = 'working'
            except Exception as e:
                health_status[component_name] = f'error: {str(e)}'
        
        # Check database connections
        try:
            recent_insights = await insight_repo.get_recent_insights(limit=1)
            health_status['insight_database'] = 'connected'
        except Exception as e:
            health_status['insight_database'] = f'error: {str(e)}'
        
        try:
            recent_problems = await problem_repo.get_recent_problems(limit=1)
            health_status['problem_database'] = 'connected'
        except Exception as e:
            health_status['problem_database'] = f'error: {str(e)}'
        
        return health_status
