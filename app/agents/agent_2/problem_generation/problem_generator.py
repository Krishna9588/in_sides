"""
Problem Generator Implementation
Generates structured problem statements from clusters and signals
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from ....utils.nlp_utils import nlp_utils


class ProblemGenerator:
    """Problem generator implementation following detailed specification"""
    
    async def generate_from_cluster(self, cluster: Dict[str, Any], signals: List,
                               classification_results: List[Dict[str, Any]],
                               pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate problem from cluster"""
        if cluster.get('size', 0) < 3:  # Minimum cluster size threshold
            return None
        
        # Get representative content
        representative_signal = cluster.get('representative_signal', {})
        content = representative_signal.get('content', '')
        
        # Generate problem statement
        problem_statement = await self._generate_problem_statement(
            content, cluster.get('keywords', [])
        )
        
        # Calculate problem metrics
        problem = {
            'problem_statement': problem_statement,
            'problem_category': self._determine_category_from_cluster(cluster),
            'severity': self._assess_severity_from_cluster(cluster),
            'frequency': {
                'count': cluster.get('size', 0),
                'time_period': '30d',
                'trend': 'stable',
                'growth_rate': 0.0
            },
            'user_segments': self._identify_user_segments(cluster),
            'source_mix': cluster.get('source_distribution', {}),
            'potential_impact': self._assess_impact(cluster),
            'confidence_score': cluster.get('avg_confidence', 0.5),
            'evidence': await self._collect_evidence_from_cluster(cluster, signals),
            'metadata': {
                'cluster_id': cluster.get('cluster_id'),
                'cluster_type': cluster.get('cluster_type'),
                'generated_at': datetime.now().isoformat()
            }
        }
        
        return problem
    
    async def generate_from_signal(self, signal,
                              signal_classification: Dict[str, Any],
                              pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate problem from individual signal"""
        content = signal.content
        
        # Generate problem statement
        problem_statement = await self._generate_problem_statement(
            content, signal_classification.get('keywords', [])
        )
        
        problem = {
            'problem_statement': problem_statement,
            'problem_category': signal_classification.get('primary_category', 'general'),
            'severity': signal_classification.get('severity', 'medium'),
            'frequency': {
                'count': 1,
                'time_period': '30d'
            },
            'user_segments': ['general_users'],
            'source_mix': {signal.source_type: 1.0},
            'potential_impact': self._assess_signal_impact(signal, signal_classification),
            'confidence_score': signal.confidence_score,
            'evidence': [{
                'signal_id': signal.id,
                'content': content,
                'source': 'individual_signal',
                'confidence': signal.confidence_score
            }],
            'metadata': {
                'source': 'individual_signal',
                'generated_at': datetime.now().isoformat()
            }
        }
        
        return problem
    
    async def _generate_problem_statement(self, content: str, keywords: List[str]) -> str:
        """Generate clear problem statement"""
        # Simple template-based generation
        if any(word in content.lower() for word in ['difficult', 'confusing']):
            return f"Users are experiencing difficulty with {', '.join(keywords[:3])}, indicating usability issues that need to be addressed."
        elif any(word in content.lower() for word in ['missing', 'need']):
            return f"Users are requesting {', '.join(keywords[:3])} functionality that is currently missing from the product."
        elif any(word in content.lower() for word in ['slow', 'performance']):
            return f"Users are reporting performance issues related to {', '.join(keywords[:3])}, affecting user experience."
        else:
            return f"Users are expressing concerns about {', '.join(keywords[:3])}, which requires attention and improvement."
    
    def _determine_category_from_cluster(self, cluster: Dict[str, Any]) -> str:
        """Determine problem category from cluster"""
        keywords = cluster.get('keywords', [])
        keyword_str = ' '.join(keywords).lower()
        
        if any(word in keyword_str for word in ['interface', 'design', 'navigation']):
            return 'ui'
        elif any(word in keyword_str for word in ['feature', 'functionality', 'capability']):
            return 'feature'
        elif any(word in keyword_str for word in ['price', 'cost', 'subscription']):
            return 'pricing'
        elif any(word in keyword_str for word in ['support', 'help', 'assistance']):
            return 'support'
        else:
            return 'general'
    
    def _assess_severity_from_cluster(self, cluster: Dict[str, Any]) -> str:
        """Assess problem severity from cluster"""
        size = cluster.get('size', 0)
        avg_confidence = cluster.get('avg_confidence', 0.5)
        
        # High severity if many signals with high confidence
        if size > 10 and avg_confidence > 0.8:
            return 'critical'
        elif size > 5 and avg_confidence > 0.7:
            return 'high'
        elif size > 3:
            return 'medium'
        else:
            return 'low'
    
    def _identify_user_segments(self, cluster: Dict[str, Any]) -> List[str]:
        """Identify user segments from cluster"""
        # This would require more sophisticated analysis
        return ['general_users']
    
    def _assess_impact(self, cluster: Dict[str, Any]) -> Dict[str, float]:
        """Assess potential impact"""
        size = cluster.get('size', 0)
        avg_confidence = cluster.get('avg_confidence', 0.5)
        
        return {
            'user_satisfaction': -min(size * 0.1, 1.0),
            'retention_risk': min(size * 0.15, 1.0),
            'competitive_disadvantage': min(size * 0.2, 1.0)
        }
    
    def _assess_signal_impact(self, signal, signal_classification: Dict[str, Any]) -> Dict[str, float]:
        """Assess impact from individual signal"""
        confidence = signal.confidence_score
        is_problem = signal_classification.get('is_problem', False)
        
        return {
            'user_satisfaction': -0.3 if is_problem else 0.1,
            'retention_risk': 0.2 if is_problem else 0.05,
            'competitive_disadvantage': 0.1 if is_problem else 0.0
        }
    
    async def _collect_evidence_from_cluster(self, cluster: Dict[str, Any], signals: List) -> List[Dict[str, Any]]:
        """Collect evidence from cluster"""
        evidence = []
        
        # Add representative signal as evidence
        representative_signal = cluster.get('representative_signal', {})
        if representative_signal:
            evidence.append({
                'signal_id': representative_signal.get('id'),
                'content': representative_signal.get('content'),
                'source': 'representative_signal',
                'confidence': cluster.get('avg_confidence', 0.5)
            })
        
        return evidence
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for problem generator"""
        return {
            'status': 'working',
            'generation_methods': ['cluster_based', 'signal_based'],
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
