"""
Cross-Domain Synthesis Implementation
Synthesizes patterns across different domains and categories
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from .....utils.nlp_utils import nlp_utils
from .....config.settings import settings

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    SYNTHESIS_AVAILABLE = True
except ImportError:
    SYNTHESIS_AVAILABLE = False


class CrossDomainSynthesis:
    """Cross-domain synthesis implementation following detailed specification"""
    
    async def synthesize_cross_domain_patterns(self, problems: List, 
                                            graph_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synthesize cross-domain patterns"""
        if not problems:
            return []
        
        try:
            # Group problems by domain
            domain_groups = self._group_by_domain(problems)
            
            # Analyze inter-domain relationships
            inter_domain_patterns = await self._analyze_inter_domain_relationships(
                domain_groups, graph_results
            )
            
            # Identify cross-domain themes
            cross_domain_themes = await self._identify_cross_domain_themes(domain_groups)
            
            # Calculate pattern strength
            patterns = []
            for pattern in inter_domain_patterns + cross_domain_themes:
                pattern['strength'] = self._calculate_pattern_strength(pattern, problems)
                pattern['confidence'] = self._calculate_pattern_confidence(pattern)
                pattern['synthesis_method'] = 'cross_domain_analysis'
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            print(f"Cross-domain synthesis failed: {e}")
            return []
    
    def _group_by_domain(self, problems: List) -> Dict[str, List]:
        """Group problems by domain/category"""
        domain_groups = {}
        
        for problem in problems:
            domain = problem.problem_category or 'general'
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(problem)
        
        return domain_groups
    
    async def _analyze_inter_domain_relationships(self, domain_groups: Dict[str, List], 
                                               graph_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze relationships between different domains"""
        patterns = []
        domains = list(domain_groups.keys())
        
        # Analyze pairwise domain relationships
        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                relationship = await self._calculate_domain_relationship(
                    domain_groups[domain1], domain_groups[domain2], graph_results
                )
                
                if relationship['strength'] > settings.CROSS_DOMAIN_THRESHOLD:
                    patterns.append({
                        'pattern_type': 'inter_domain_relationship',
                        'domains': [domain1, domain2],
                        'relationship': relationship,
                        'strength': relationship['strength'],
                        'evidence': relationship['evidence']
                    })
        
        return patterns
    
    async def _calculate_domain_relationship(self, domain1_problems: List, 
                                         domain2_problems: List,
                                         graph_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate relationship strength between two domains"""
        # Extract keywords from both domains
        domain1_keywords = self._extract_domain_keywords(domain1_problems)
        domain2_keywords = self._extract_domain_keywords(domain2_problems)
        
        # Calculate keyword overlap
        common_keywords = domain1_keywords.intersection(domain2_keywords)
        total_keywords = domain1_keywords.union(domain2_keywords)
        
        # Calculate relationship metrics
        overlap_ratio = len(common_keywords) / len(total_keywords) if total_keywords else 0
        frequency_score = (len(domain1_problems) + len(domain2_problems)) / 2
        
        # Combined relationship strength
        strength = (overlap_ratio * 0.6) + (frequency_score * 0.4)
        
        return {
            'strength': strength,
            'overlap_ratio': overlap_ratio,
            'frequency_score': frequency_score,
            'common_keywords': list(common_keywords),
            'evidence': {
                'domain1_count': len(domain1_problems),
                'domain2_count': len(domain2_problems),
                'common_keyword_count': len(common_keywords)
            }
        }
    
    def _extract_domain_keywords(self, problems: List) -> set:
        """Extract keywords from problems in a domain"""
        all_keywords = set()
        for problem in problems:
            keywords = nlp_utils.extract_keywords(problem.problem_statement, max_keywords=10)
            all_keywords.update(keywords)
        return all_keywords
    
    async def _identify_cross_domain_themes(self, domain_groups: Dict[str, List]) -> List[Dict[str, Any]]:
        """Identify themes that span across domains"""
        themes = []
        
        # Collect all keywords across all domains
        all_keywords = []
        for domain, problems in domain_groups.items():
            for problem in problems:
                keywords = nlp_utils.extract_keywords(problem.problem_statement, max_keywords=5)
                for keyword in keywords:
                    all_keywords.append({
                        'keyword': keyword,
                        'domain': domain,
                        'problem_id': problem.id
                    })
        
        # Find keywords that appear in multiple domains
        keyword_domains = {}
        for keyword_info in all_keywords:
            keyword = keyword_info['keyword']
            domain = keyword_info['domain']
            
            if keyword not in keyword_domains:
                keyword_domains[keyword] = set()
            keyword_domains[keyword].add(domain)
        
        # Identify cross-domain themes
        for keyword, domains in keyword_domains.items():
            if len(domains) >= 2:  # Appears in at least 2 domains
                theme_strength = len(domains) / len(domain_groups)
                
                themes.append({
                    'pattern_type': 'cross_domain_theme',
                    'theme_keyword': keyword,
                    'spanning_domains': list(domains),
                    'domain_count': len(domains),
                    'strength': theme_strength,
                    'evidence': {
                        'total_domains': len(domain_groups),
                        'spanning_domain_ratio': len(domains) / len(domain_groups)
                    }
                })
        
        return themes
    
    def _calculate_pattern_strength(self, pattern: Dict[str, Any], all_problems: List) -> float:
        """Calculate strength of a pattern"""
        base_strength = pattern.get('strength', 0.0)
        
        # Boost strength based on evidence
        evidence = pattern.get('evidence', {})
        if evidence:
            evidence_boost = min(evidence.get('common_keyword_count', 0) / 10, 0.3)
            base_strength += evidence_boost
        
        return min(base_strength, 1.0)
    
    def _calculate_pattern_confidence(self, pattern: Dict[str, Any]) -> float:
        """Calculate confidence in pattern"""
        base_confidence = 0.7  # Base confidence for cross-domain patterns
        
        # Adjust based on pattern type
        pattern_type = pattern.get('pattern_type', '')
        if pattern_type == 'inter_domain_relationship':
            base_confidence += 0.2
        elif pattern_type == 'cross_domain_theme':
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for cross-domain synthesis"""
        return {
            'status': 'working',
            'sklearn_available': SYNTHESIS_AVAILABLE,
            'synthesis_methods': [
                'domain_grouping',
                'inter_domain_relationships',
                'cross_domain_themes'
            ],
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
