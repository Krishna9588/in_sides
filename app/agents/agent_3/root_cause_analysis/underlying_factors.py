"""
Underlying Factors Implementation
Identifies underlying factors contributing to problems
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
    FACTORS_AVAILABLE = True
except ImportError:
    FACTORS_AVAILABLE = False


class UnderlyingFactors:
    """Underlying factors implementation following detailed specification"""
    
    async def identify_underlying_factors(self, problems: List, 
                                        pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify underlying factors contributing to problems"""
        if not problems:
            return []
        
        try:
            # Analyze different types of underlying factors
            technical_factors = await self._identify_technical_factors(problems)
            user_factors = await self._identify_user_factors(problems)
            business_factors = await self._identify_business_factors(problems)
            environmental_factors = await self._identify_environmental_factors(problems)
            
            # Combine all factors
            all_factors = technical_factors + user_factors + business_factors + environmental_factors
            
            # Rank factors by importance
            ranked_factors = await self._rank_factors_by_importance(all_factors, problems)
            
            return ranked_factors
            
        except Exception as e:
            print(f"Underlying factors identification failed: {e}")
            return []
    
    async def _identify_technical_factors(self, problems: List) -> List[Dict[str, Any]]:
        """Identify technical underlying factors"""
        technical_factors = []
        
        # Technical factor indicators
        technical_keywords = {
            'infrastructure': ['server', 'database', 'network', 'cloud', 'hosting', 'architecture'],
            'code_quality': ['bug', 'error', 'crash', 'performance', 'memory', 'cpu', 'optimization'],
            'integration': ['api', 'third_party', 'compatibility', 'interface', 'connection'],
            'scalability': ['scale', 'load', 'capacity', 'throughput', 'concurrency'],
            'security': ['vulnerability', 'authentication', 'authorization', 'encryption', 'security']
        }
        
        for problem in problems:
            problem_text = problem.problem_statement.lower()
            
            for factor_type, keywords in technical_keywords.items():
                relevance_score = self._calculate_factor_relevance(problem_text, keywords)
                
                if relevance_score > settings.TECHNICAL_FACTOR_THRESHOLD:
                    technical_factors.append({
                        'factor_type': 'technical',
                        'factor_category': factor_type,
                        'relevance_score': relevance_score,
                        'evidence': self._collect_factor_evidence(problem_text, keywords),
                        'contributing_problems': [problem.id],
                        'confidence': min(relevance_score * 0.8 + 0.2, 1.0)  # Boost for technical factors
                    })
        
        return technical_factors
    
    async def _identify_user_factors(self, problems: List) -> List[Dict[str, Any]]:
        """Identify user-related underlying factors"""
        user_factors = []
        
        # User factor indicators
        user_keywords = {
            'experience': ['user_experience', 'ux', 'usability', 'interface', 'navigation', 'workflow'],
            'behavior': ['behavior', 'usage_pattern', 'habit', 'preference', 'expectation'],
            'skill_level': ['beginner', 'expert', 'skilled', 'training', 'learning_curve', 'documentation'],
            'demographics': ['age_group', 'technical_skill', 'role', 'department', 'industry'],
            'feedback': ['complaint', 'frustration', 'confusion', 'satisfaction', 'rating', 'review']
        }
        
        for problem in problems:
            problem_text = problem.problem_statement.lower()
            
            for factor_type, keywords in user_keywords.items():
                relevance_score = self._calculate_factor_relevance(problem_text, keywords)
                
                if relevance_score > settings.USER_FACTOR_THRESHOLD:
                    user_factors.append({
                        'factor_type': 'user',
                        'factor_category': factor_type,
                        'relevance_score': relevance_score,
                        'evidence': self._collect_factor_evidence(problem_text, keywords),
                        'contributing_problems': [problem.id],
                        'confidence': min(relevance_score * 0.7 + 0.3, 1.0)  # Boost for user factors
                    })
        
        return user_factors
    
    async def _identify_business_factors(self, problems: List) -> List[Dict[str, Any]]:
        """Identify business-related underlying factors"""
        business_factors = []
        
        # Business factor indicators
        business_keywords = {
            'market': ['market', 'competition', 'trend', 'demand', 'customer', 'industry'],
            'operational': ['process', 'workflow', 'efficiency', 'cost', 'resource', 'productivity'],
            'strategic': ['strategy', 'roadmap', 'vision', 'goal', 'objective', 'priority'],
            'financial': ['revenue', 'profit', 'budget', 'investment', 'pricing', 'cost'],
            'organizational': ['team', 'communication', 'collaboration', 'management', 'structure']
        }
        
        for problem in problems:
            problem_text = problem.problem_statement.lower()
            
            for factor_type, keywords in business_keywords.items():
                relevance_score = self._calculate_factor_relevance(problem_text, keywords)
                
                if relevance_score > settings.BUSINESS_FACTOR_THRESHOLD:
                    business_factors.append({
                        'factor_type': 'business',
                        'factor_category': factor_type,
                        'relevance_score': relevance_score,
                        'evidence': self._collect_factor_evidence(problem_text, keywords),
                        'contributing_problems': [problem.id],
                        'confidence': min(relevance_score * 0.6 + 0.4, 1.0)
                    })
        
        return business_factors
    
    async def _identify_environmental_factors(self, problems: List) -> List[Dict[str, Any]]:
        """Identify environmental underlying factors"""
        environmental_factors = []
        
        # Environmental factor indicators
        environmental_keywords = {
            'technological': ['technology', 'platform', 'browser', 'device', 'os', 'version'],
            'temporal': ['time', 'season', 'period', 'schedule', 'deadline', 'milestone'],
            'geographic': ['location', 'region', 'country', 'language', 'timezone'],
            'regulatory': ['compliance', 'regulation', 'legal', 'policy', 'standard', 'requirement'],
            'economic': ['economy', 'market_conditions', 'recession', 'growth', 'inflation']
        }
        
        for problem in problems:
            problem_text = problem.problem_statement.lower()
            
            for factor_type, keywords in environmental_keywords.items():
                relevance_score = self._calculate_factor_relevance(problem_text, keywords)
                
                if relevance_score > settings.ENVIRONMENTAL_FACTOR_THRESHOLD:
                    environmental_factors.append({
                        'factor_type': 'environmental',
                        'factor_category': factor_type,
                        'relevance_score': relevance_score,
                        'evidence': self._collect_factor_evidence(problem_text, keywords),
                        'contributing_problems': [problem.id],
                        'confidence': min(relevance_score * 0.5 + 0.5, 1.0)
                    })
        
        return environmental_factors
    
    def _calculate_factor_relevance(self, problem_text: str, keywords: List[str]) -> float:
        """Calculate relevance score for a factor"""
        if not keywords:
            return 0.0
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in problem_text)
        
        # Calculate relevance based on match ratio and keyword importance
        match_ratio = matches / len(keywords)
        
        # Boost for exact phrase matches
        phrase_boost = 0.0
        for keyword in keywords:
            if ' ' in keyword:  # Multi-word keyword
                if keyword in problem_text:
                    phrase_boost += 0.2
        
        relevance = (match_ratio * 0.7) + phrase_boost
        return min(relevance, 1.0)
    
    def _collect_factor_evidence(self, problem_text: str, keywords: List[str]) -> List[str]:
        """Collect evidence for factor identification"""
        evidence = []
        
        for keyword in keywords:
            if keyword in problem_text:
                # Find context around keyword
                words = problem_text.split()
                for i, word in enumerate(words):
                    if keyword in word:
                        # Get surrounding words for context
                        start_idx = max(0, i - 3)
                        end_idx = min(len(words), i + 4)
                        context = ' '.join(words[start_idx:end_idx])
                        evidence.append(f"Keyword '{keyword}' found in context: '{context}'")
                        break
        
        return evidence
    
    async def _rank_factors_by_importance(self, factors: List[Dict[str, Any]], problems: List) -> List[Dict[str, Any]]:
        """Rank factors by importance"""
        if not factors:
            return []
        
        # Calculate importance scores
        for factor in factors:
            # Base importance from relevance and confidence
            base_importance = factor.get('relevance_score', 0.5) * factor.get('confidence', 0.5)
            
            # Boost based on number of contributing problems
            problem_count = len(factor.get('contributing_problems', []))
            breadth_boost = min(problem_count / len(problems), 0.3)
            
            # Boost based on factor type
            type_boost = self._get_type_importance_boost(factor.get('factor_type', ''))
            
            # Calculate final importance score
            importance_score = base_importance + breadth_boost + type_boost
            factor['importance_score'] = min(importance_score, 1.0)
        
        # Sort factors by importance
        ranked_factors = sorted(factors, key=lambda x: x.get('importance_score', 0), reverse=True)
        
        return ranked_factors
    
    def _get_type_importance_boost(self, factor_type: str) -> float:
        """Get importance boost based on factor type"""
        type_boosts = {
            'technical': 0.3,    # Technical factors often critical
            'user': 0.2,        # User factors important for UX
            'business': 0.25,      # Business factors strategic importance
            'environmental': 0.15   # Environmental factors contextual
        }
        
        return type_boosts.get(factor_type, 0.1)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for underlying factors"""
        return {
            'status': 'working',
            'sklearn_available': FACTORS_AVAILABLE,
            'factor_types': [
                'technical',
                'user', 
                'business',
                'environmental'
            ],
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
