"""
Causal Chains Implementation
Analyzes causal relationships and chains between problems
"""
import asyncio
from typing import Dict, Any, List, Tuple
from datetime import datetime
import numpy as np

from .....utils.nlp_utils import nlp_utils
from .....config.settings import settings

try:
    import networkx as nx
    CAUSAL_AVAILABLE = True
except ImportError:
    CAUSAL_AVAILABLE = False


class CausalChains:
    """Causal chains implementation following detailed specification"""
    
    def __init__(self):
        self.causal_graph = None
        self.causal_relationships = []
    
    async def analyze_causal_chains(self, problems: List, 
                                   graph_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze causal chains between problems"""
        if not problems:
            return []
        
        try:
            # Build causal relationship graph
            await self._build_causal_graph(problems)
            
            # Identify causal chains
            chains = await self._identify_causal_chains()
            
            # Analyze chain properties
            analyzed_chains = []
            for chain in chains:
                analyzed_chain = await self._analyze_chain_properties(chain, problems)
                analyzed_chains.append(analyzed_chain)
            
            return analyzed_chains
            
        except Exception as e:
            print(f"Causal chain analysis failed: {e}")
            return []
    
    async def _build_causal_graph(self, problems: List):
        """Build causal relationship graph"""
        if not CAUSAL_AVAILABLE:
            self.causal_graph = await self._build_fallback_causal_graph(problems)
            return
        
        try:
            # Initialize directed graph for causal relationships
            self.causal_graph = nx.DiGraph()
            
            # Add problem nodes
            for problem in problems:
                self.causal_graph.add_node(
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
            
            # Identify causal relationships
            causal_relationships = await self._identify_causal_relationships(problems)
            self.causal_relationships = causal_relationships
            
            # Add causal edges
            for relationship in causal_relationships:
                if relationship['confidence'] > settings.CAUSAL_THRESHOLD:
                    self.causal_graph.add_edge(
                        relationship['cause_problem'],
                        relationship['effect_problem'],
                        type='causal',
                        weight=relationship['confidence'],
                        data={
                            'causal_strength': relationship['confidence'],
                            'evidence': relationship['evidence'],
                            'relationship_type': relationship['type']
                        }
                    )
            
        except Exception as e:
            print(f"Causal graph construction failed: {e}")
    
    async def _build_fallback_causal_graph(self, problems: List):
        """Build fallback causal graph when NetworkX is not available"""
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
        
        # Simple keyword-based causal relationships
        causal_relationships = await self._identify_causal_relationships(problems)
        
        for relationship in causal_relationships:
            if relationship['confidence'] > settings.CAUSAL_THRESHOLD:
                edges.append({
                    'source': relationship['cause_problem'],
                    'target': relationship['effect_problem'],
                    'type': 'causal',
                    'weight': relationship['confidence'],
                    'data': {
                        'causal_strength': relationship['confidence'],
                        'evidence': relationship['evidence'],
                        'relationship_type': relationship['type']
                    }
                })
        
        self.causal_graph = {'nodes': nodes, 'edges': edges}
    
    async def _identify_causal_relationships(self, problems: List) -> List[Dict[str, Any]]:
        """Identify potential causal relationships between problems"""
        relationships = []
        
        # Extract causal indicators from problem statements
        causal_indicators = [
            'leads to', 'causes', 'results in', 'due to', 'because of',
            'consequence', 'effect', 'impact', 'trigger', 'precipitates'
        ]
        
        for i, problem1 in enumerate(problems):
            for problem2 in problems[i+1:]:
                relationship = await self._analyze_causal_relationship(
                    problem1, problem2, causal_indicators
                )
                
                if relationship['confidence'] > 0:
                    relationships.append(relationship)
        
        return relationships
    
    async def _analyze_causal_relationship(self, problem1, problem2, causal_indicators: List[str]) -> Dict[str, Any]:
        """Analyze potential causal relationship between two problems"""
        text1 = problem1.problem_statement.lower()
        text2 = problem2.problem_statement.lower()
        
        # Look for causal indicators
        causal_score = 0.0
        evidence = []
        
        for indicator in causal_indicators:
            if indicator in text1 and any(word in text2 for word in ['problem', 'issue', 'symptom']):
                causal_score += 0.3
                evidence.append(f"'{indicator}' found in {problem1.id} referencing {problem2.id}")
            elif indicator in text2 and any(word in text1 for word in ['problem', 'issue', 'symptom']):
                causal_score += 0.3
                evidence.append(f"'{indicator}' found in {problem2.id} referencing {problem1.id}")
        
        # Temporal relationship (if one problem consistently precedes another)
        temporal_score = await self._analyze_temporal_causality(problem1, problem2)
        causal_score += temporal_score['score'] * 0.4
        if temporal_score['evidence']:
            evidence.extend(temporal_score['evidence'])
        
        # Category-based causality (e.g., UI problems causing feature requests)
        category_score = self._analyze_category_causality(problem1, problem2)
        causal_score += category_score['score'] * 0.3
        if category_score['evidence']:
            evidence.extend(category_score['evidence'])
        
        return {
            'cause_problem': f"problem_{problem1.id}",
            'effect_problem': f"problem_{problem2.id}",
            'confidence': min(causal_score, 1.0),
            'type': 'potential_causal',
            'evidence': evidence,
            'temporal_relationship': temporal_score,
            'category_relationship': category_score
        }
    
    async def _analyze_temporal_causality(self, problem1, problem2) -> Dict[str, Any]:
        """Analyze temporal relationship between problems"""
        if not (hasattr(problem1, 'created_at') and hasattr(problem2, 'created_at')):
            return {'score': 0.0, 'evidence': []}
        
        time1 = datetime.fromisoformat(problem1.created_at.replace('Z', '+00:00'))
        time2 = datetime.fromisoformat(problem2.created_at.replace('Z', '+00:00'))
        
        time_diff = (time2 - time1).total_seconds() / 3600  # Hours
        
        if time_diff > 1 and time_diff < 72:  # 1-72 hours apart
            return {
                'score': 0.7,
                'evidence': [f"{problem2.id} reported {time_diff:.1f} hours after {problem1.id}"]
            }
        elif time_diff > 72:  # More than 3 days apart
            return {
                'score': 0.3,
                'evidence': [f"{problem2.id} reported {time_diff/24:.1f} days after {problem1.id}"]
            }
        else:
            return {'score': 0.0, 'evidence': []}
    
    def _analyze_category_causality(self, problem1, problem2) -> Dict[str, Any]:
        """Analyze category-based causality"""
        category1 = problem1.problem_category
        category2 = problem2.problem_category
        
        # Define potential causal relationships between categories
        causal_mappings = {
            ('ui', 'feature'): 0.6,  # UI problems cause feature requests
            ('performance', 'feature'): 0.5,  # Performance issues cause feature requests
            ('ui', 'support'): 0.4,  # UI problems cause support requests
            ('feature', 'pricing'): 0.3,  # Feature issues affect pricing perception
            ('performance', 'support'): 0.5,  # Performance issues cause support requests
        }
        
        score = causal_mappings.get((category1, category2), 0.0)
        if score > 0:
            return {
                'score': score,
                'evidence': [f"Category relationship: {category1} -> {category2}"]
            }
        
        return {'score': 0.0, 'evidence': []}
    
    async def _identify_causal_chains(self) -> List[List[str]]:
        """Identify causal chains in the graph"""
        if not CAUSAL_AVAILABLE or not self.causal_graph:
            return []
        
        try:
            chains = []
            
            # Find all simple paths (causal chains)
            for source in self.causal_graph.nodes():
                for target in self.causal_graph.nodes():
                    if source != target:
                        try:
                            # Find paths from source to target
                            paths = list(nx.all_simple_paths(self.causal_graph, source, target, cutoff=5))
                            
                            for path in paths:
                                if len(path) > 1:  # At least one edge
                                    chains.append(path)
                        except nx.NetworkXNoPath:
                            continue
            
            # Remove duplicate chains
            unique_chains = []
            seen_chains = set()
            
            for chain in chains:
                chain_tuple = tuple(chain)
                if chain_tuple not in seen_chains:
                    unique_chains.append(chain)
                    seen_chains.add(chain_tuple)
            
            return unique_chains
            
        except Exception as e:
            print(f"Causal chain identification failed: {e}")
            return []
    
    async def _analyze_chain_properties(self, chain: List[str], problems: List) -> Dict[str, Any]:
        """Analyze properties of a causal chain"""
        if not chain:
            return {}
        
        try:
            # Get problem objects for chain nodes
            chain_problems = []
            for node_id in chain:
                problem_id = node_id.replace('problem_', '')
                problem = next((p for p in problems if str(p.id) == problem_id), None)
                if problem:
                    chain_problems.append(problem)
            
            # Calculate chain metrics
            chain_length = len(chain) - 1  # Number of causal relationships
            
            # Calculate chain strength (average confidence)
            if CAUSAL_AVAILABLE and hasattr(self.causal_graph, 'get_edge_data'):
                edge_strengths = []
                for i in range(len(chain) - 1):
                    edge_data = self.causal_graph.get_edge_data(chain[i], chain[i+1])
                    if edge_data:
                        edge_strengths.append(edge_data.get('causal_strength', 0.5))
                
                avg_strength = np.mean(edge_strengths) if edge_strengths else 0.5
            else:
                avg_strength = 0.5  # Default for fallback
            
            # Calculate chain severity progression
            severities = [p.severity for p in chain_problems if hasattr(p, 'severity')]
            severity_progression = self._analyze_severity_progression(severities)
            
            # Calculate chain confidence
            confidences = [p.confidence_score for p in chain_problems if hasattr(p, 'confidence_score')]
            avg_confidence = np.mean(confidences) if confidences else 0.5
            
            return {
                'chain_nodes': chain,
                'chain_length': chain_length,
                'chain_problems': chain_problems,
                'avg_strength': avg_strength,
                'severity_progression': severity_progression,
                'avg_confidence': avg_confidence,
                'chain_type': self._classify_chain_type(chain_problems),
                'critical_path': self._identify_critical_path(chain, problems)
            }
            
        except Exception as e:
            print(f"Chain analysis failed: {e}")
            return {}
    
    def _analyze_severity_progression(self, severities: List[str]) -> Dict[str, Any]:
        """Analyze how severity progresses through the chain"""
        if not severities:
            return {'progression': 'stable', 'trend': 'none'}
        
        severity_order = ['low', 'medium', 'high', 'critical']
        numeric_severities = []
        
        for severity in severities:
            try:
                idx = severity_order.index(severity)
                numeric_severities.append(idx)
            except ValueError:
                numeric_severities.append(1)  # Default to medium
        
        if len(numeric_severities) < 2:
            return {'progression': 'stable', 'trend': 'none'}
        
        # Calculate trend
        if numeric_severities[-1] > numeric_severities[0]:
            trend = 'escalating'
        elif numeric_severities[-1] < numeric_severities[0]:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'progression': trend,
            'trend': trend,
            'severity_range': f"{severity_order[min(numeric_severities)]} to {severity_order[max(numeric_severities)]}"
        }
    
    def _classify_chain_type(self, chain_problems: List) -> str:
        """Classify the type of causal chain"""
        if not chain_problems:
            return 'unknown'
        
        categories = [p.problem_category for p in chain_problems if hasattr(p, 'problem_category')]
        unique_categories = list(set(categories))
        
        if len(unique_categories) == 1:
            return 'single_domain'
        elif len(unique_categories) == 2:
            return 'cross_domain'
        else:
            return 'multi_domain'
    
    def _identify_critical_path(self, chain: List[str], problems: List) -> Dict[str, Any]:
        """Identify if this is a critical path in the overall system"""
        # Simple heuristic: chains involving high-severity or high-confidence problems
        chain_problems = []
        for node_id in chain:
            problem_id = node_id.replace('problem_', '')
            problem = next((p for p in problems if str(p.id) == problem_id), None)
            if problem:
                chain_problems.append(problem)
        
        if not chain_problems:
            return {'is_critical': False, 'reason': 'no_problems'}
        
        # Check for high severity or confidence
        max_severity = max([
            self._severity_to_numeric(p.severity) for p in chain_problems 
            if hasattr(p, 'severity')
        ], default=0)
        
        avg_confidence = np.mean([
            p.confidence_score for p in chain_problems 
            if hasattr(p, 'confidence_score')
        ], default=0.5)
        
        is_critical = (max_severity >= 3 or avg_confidence > 0.8)  # High severity or confidence
        
        return {
            'is_critical': is_critical,
            'max_severity': max_severity,
            'avg_confidence': avg_confidence,
            'reason': 'high_severity_or_confidence' if is_critical else 'normal'
        }
    
    def _severity_to_numeric(self, severity: str) -> int:
        """Convert severity string to numeric"""
        severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        return severity_map.get(severity, 2)  # Default to medium
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for causal chains"""
        return {
            'status': 'working',
            'networkx_available': CAUSAL_AVAILABLE,
            'graph_constructed': self.causal_graph is not None,
            'causal_relationships_count': len(self.causal_relationships),
            'analysis_methods': [
                'temporal_analysis',
                'category_analysis',
                'chain_identification',
                'critical_path_analysis'
            ],
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
