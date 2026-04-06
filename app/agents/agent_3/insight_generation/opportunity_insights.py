"""
Opportunity Insights Implementation
Generates opportunity insights from analysis results
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from .....utils.nlp_utils import nlp_utils
from .....config.settings import settings


class OpportunityInsights:
    """Opportunity insights implementation following detailed specification"""
    
    async def generate_opportunity_insights(self, problems: List, 
                                          pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate opportunity insights from analysis results"""
        if not problems:
            return []
        
        try:
            insights = []
            
            # 1. Unmet need opportunities
            unmet_opportunities = await self._identify_unmet_need_opportunities(problems, pattern_results)
            insights.extend(unmet_opportunities)
            
            # 2. Market gap opportunities
            gap_opportunities = await self._identify_market_gap_opportunities(problems, pattern_results)
            insights.extend(gap_opportunities)
            
            # 3. Innovation opportunities
            innovation_opportunities = await self._identify_innovation_opportunities(problems, pattern_results)
            insights.extend(innovation_opportunities)
            
            # 4. Competitive advantage opportunities
            competitive_opportunities = await self._identify_competitive_opportunities(problems, pattern_results)
            insights.extend(competitive_opportunities)
            
            return insights
            
        except Exception as e:
            print(f"Opportunity insights generation failed: {e}")
            return []
    
    async def _identify_unmet_need_opportunities(self, problems: List, pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities from unmet needs"""
        opportunities = []
        
        # Extract user needs from problems
        user_problems = [
            p for p in problems 
            if any(term in p.problem_statement.lower() for term in ['need', 'want', 'require', 'missing', 'lack'])
        ]
        
        if not user_problems:
            return opportunities
        
        # Analyze need patterns
        need_patterns = await self._analyze_need_patterns(user_problems)
        
        for pattern in need_patterns:
            opportunity = {
                'insight_type': 'unmet_need_opportunity',
                'opportunity_category': 'user_need',
                'title': f"Opportunity: {pattern['dominant_need']}",
                'description': f"Address unmet need for {pattern['dominant_need']}",
                'findings': pattern,
                'market_potential': self._estimate_market_potential(pattern),
                'implementation_complexity': self._estimate_implementation_complexity(pattern),
                'confidence': pattern['confidence'],
                'evidence': pattern['evidence_problems'],
                'generated_at': datetime.now().isoformat()
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _analyze_need_patterns(self, user_problems: List) -> List[Dict[str, Any]]:
        """Analyze patterns in user needs"""
        patterns = []
        
        # Extract keywords from user need problems
        all_needs = []
        for problem in user_problems:
            needs = nlp_utils.extract_keywords(problem.problem_statement, max_keywords=5)
            all_needs.extend(needs)
        
        # Group and analyze needs
        need_frequency = {}
        for need in all_needs:
            need_frequency[need] = need_frequency.get(need, 0) + 1
        
        # Identify significant needs
        significant_needs = [
            (need, count) for need, count in need_frequency.items() 
            if count >= 2  # Appears in at least 2 problems
        ]
        
        for need, count in significant_needs:
            # Find related problems
            related_problems = [
                p for p in user_problems 
                if need in p.problem_statement.lower()
            ]
            
            # Analyze need context
            need_context = self._analyze_need_context(need, related_problems)
            
            pattern = {
                'dominant_need': need,
                'frequency': count,
                'related_problems': [p.id for p in related_problems],
                'need_context': need_context,
                'confidence': min(count / len(user_problems), 1.0),
                'evidence_problems': related_problems
            }
            patterns.append(pattern)
        
        return patterns
    
    def _analyze_need_context(self, need: str, related_problems: List) -> Dict[str, Any]:
        """Analyze context around a specific need"""
        if not related_problems:
            return {'context_type': 'unknown', 'severity': 'unknown'}
        
        # Extract context from problem statements
        context_texts = [p.problem_statement for p in related_problems]
        
        # Analyze sentiment and urgency
        sentiments = []
        urgencies = []
        
        for text in context_texts:
            # Simple sentiment analysis
            positive_words = ['love', 'great', 'excellent', 'perfect']
            negative_words = ['frustrated', 'difficult', 'confusing', 'hate']
            
            text_lower = text.lower()
            if any(word in text_lower for word in negative_words):
                sentiments.append('negative')
            elif any(word in text_lower for word in positive_words):
                sentiments.append('positive')
            else:
                sentiments.append('neutral')
            
            # Urgency analysis
            urgent_words = ['urgent', 'immediately', 'asap', 'critical', 'emergency']
            if any(word in text_lower for word in urgent_words):
                urgencies.append('high')
            else:
                urgencies.append('medium')
        
        dominant_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else 'neutral'
        dominant_urgency = max(set(urgencies), key=urgencies.count) if urgencies else 'medium'
        
        return {
            'context_type': 'user_feedback',
            'dominant_sentiment': dominant_sentiment,
            'dominant_urgency': dominant_urgency,
            'sentiment_distribution': {s: sentiments.count(s) for s in set(sentiments)},
            'urgency_distribution': {u: urgencies.count(u) for u in set(urgencies)}
        }
    
    def _estimate_market_potential(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate market potential for an opportunity"""
        frequency = pattern.get('frequency', 1)
        confidence = pattern.get('confidence', 0.5)
        urgency = pattern.get('need_context', {}).get('dominant_urgency', 'medium')
        
        # Calculate market potential score
        base_potential = frequency * confidence
        
        # Boost for high urgency
        if urgency == 'high':
            base_potential *= 1.5
        
        # Categorize potential
        if base_potential > 5:
            potential_level = 'high'
        elif base_potential > 2:
            potential_level = 'medium'
        else:
            potential_level = 'low'
        
        return {
            'potential_score': min(base_potential, 10),
            'potential_level': potential_level,
            'market_size_estimate': self._estimate_market_size(frequency, confidence)
        }
    
    def _estimate_market_size(self, frequency: int, confidence: float) -> str:
        """Estimate market size based on frequency and confidence"""
        adjusted_frequency = frequency * confidence
        
        if adjusted_frequency > 8:
            return 'large'
        elif adjusted_frequency > 4:
            return 'medium'
        else:
            return 'small'
    
    def _estimate_implementation_complexity(self, pattern: Dict[str, Any]) -> str:
        """Estimate implementation complexity"""
        need = pattern.get('dominant_need', '')
        context = pattern.get('need_context', {})
        
        # Complexity indicators
        technical_needs = ['integration', 'api', 'infrastructure', 'system']
        design_needs = ['interface', 'design', 'ux', 'user experience']
        business_needs = ['pricing', 'business', 'process', 'workflow']
        
        complexity_score = 0
        if any(indicator in need for indicator in technical_needs):
            complexity_score += 3
        if any(indicator in need for indicator in design_needs):
            complexity_score += 2
        if any(indicator in need for indicator in business_needs):
            complexity_score += 1
        
        # Adjust based on sentiment (negative needs may be more complex)
        if context.get('dominant_sentiment') == 'negative':
            complexity_score += 1
        
        if complexity_score >= 5:
            return 'high'
        elif complexity_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    async def _identify_market_gap_opportunities(self, problems: List, pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify market gap opportunities"""
        opportunities = []
        
        # Look for underserved areas
        underserved_patterns = await self._identify_underserved_patterns(problems, pattern_results)
        
        for pattern in underserved_patterns:
            opportunity = {
                'insight_type': 'market_gap_opportunity',
                'opportunity_category': 'market_gap',
                'title': f"Market Gap: {pattern['gap_area']}",
                'description': f"Address underserved market area: {pattern['gap_area']}",
                'findings': pattern,
                'competitive_advantage': self._assess_competitive_advantage(pattern),
                'first_mover_advantage': pattern.get('first_mover_potential', False),
                'confidence': pattern['confidence'],
                'evidence': pattern['evidence_problems'],
                'generated_at': datetime.now().isoformat()
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _identify_underserved_patterns(self, problems: List, pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify underserved patterns"""
        patterns = []
        
        # Analyze problem categories that appear less frequently
        category_counts = {}
        for problem in problems:
            category = getattr(problem, 'problem_category', 'general')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        total_problems = len(problems)
        underserved_categories = [
            cat for cat, count in category_counts.items() 
            if count / total_problems < 0.1  # Less than 10% of problems
        ]
        
        for category in underserved_categories:
            category_problems = [
                p for p in problems 
                if getattr(p, 'problem_category', 'general') == category
            ]
            
            pattern = {
                'gap_area': category,
                'problem_count': len(category_problems),
                'market_share': len(category_problems) / total_problems,
                'growth_potential': self._assess_growth_potential(category),
                'first_mover_potential': len(category_problems) < 3,
                'confidence': 0.6,  # Lower confidence for gaps
                'evidence_problems': [p.id for p in category_problems]
            }
            patterns.append(pattern)
        
        return patterns
    
    def _assess_growth_potential(self, category: str) -> str:
        """Assess growth potential for a category"""
        growth_categories = {
            'feature': 'high',      # Features often have high growth potential
            'ui': 'medium',        # UI improvements moderate growth
            'performance': 'high',   # Performance improvements high value
            'support': 'medium',     # Support improvements moderate value
            'pricing': 'low'        # Pricing changes lower growth
        }
        
        return growth_categories.get(category, 'medium')
    
    def _assess_competitive_advantage(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Assess competitive advantage for addressing a gap"""
        gap_area = pattern.get('gap_area', '')
        first_mover = pattern.get('first_mover_potential', False)
        
        # Competitive advantage factors
        advantage_scores = {}
        
        # First mover advantage
        if first_mover:
            advantage_scores['first_mover'] = 0.8
        else:
            advantage_scores['first_mover'] = 0.2
        
        # Market size advantage
        market_share = pattern.get('market_share', 0)
        if market_share < 0.05:  # Less than 5% market share
            advantage_scores['market_size'] = 0.9
        elif market_share < 0.15:  # Less than 15% market share
            advantage_scores['market_size'] = 0.6
        else:
            advantage_scores['market_size'] = 0.3
        
        # Growth potential advantage
        growth_potential = self._assess_growth_potential(gap_area)
        if growth_potential == 'high':
            advantage_scores['growth'] = 0.8
        elif growth_potential == 'medium':
            advantage_scores['growth'] = 0.5
        else:
            advantage_scores['growth'] = 0.2
        
        # Calculate overall advantage
        overall_advantage = np.mean(list(advantage_scores.values()))
        
        return {
            'advantage_score': overall_advantage,
            'advantage_factors': advantage_scores,
            'competitive_position': 'strong' if overall_advantage > 0.7 else 'moderate' if overall_advantage > 0.4 else 'weak'
        }
    
    async def _identify_innovation_opportunities(self, problems: List, pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify innovation opportunities"""
        opportunities = []
        
        # Look for emerging patterns and cross-domain combinations
        emerging_patterns = pattern_results.get('patterns', [])
        cross_domain_patterns = [p for p in emerging_patterns if 'cross_domain' in str(p)]
        
        for pattern in cross_domain_patterns:
            innovation_opportunity = await self._analyze_innovation_potential(pattern, problems)
            
            if innovation_opportunity['innovation_score'] > settings.INNOVATION_THRESHOLD:
                opportunity = {
                    'insight_type': 'innovation_opportunity',
                    'opportunity_category': 'innovation',
                    'title': f"Innovation Opportunity: {pattern.get('theme_keyword', 'emerging_pattern')}",
                    'description': f"Leverage cross-domain pattern for innovation: {pattern.get('theme_keyword', 'emerging_pattern')}",
                    'findings': innovation_opportunity,
                    'innovation_type': innovation_opportunity['innovation_type'],
                    'disruptive_potential': innovation_opportunity['disruptive_potential'],
                    'confidence': innovation_opportunity['confidence'],
                    'evidence': pattern.get('evidence', []),
                    'generated_at': datetime.now().isoformat()
                }
                opportunities.append(opportunity)
        
        return opportunities
    
    async def _analyze_innovation_potential(self, pattern: Dict[str, Any], problems: List) -> List[Dict[str, Any]]:
        """Analyze innovation potential of a pattern"""
        pattern_type = pattern.get('pattern_type', '')
        spanning_domains = pattern.get('spanning_domains', [])
        
        innovation_score = 0.5  # Base score
        
        # Boost for cross-domain patterns
        if 'cross_domain' in pattern_type or len(spanning_domains) > 1:
            innovation_score += 0.3
        
        # Boost for emergent themes
        if 'emergent_theme' in pattern_type:
            innovation_score += 0.2
        
        # Determine innovation type
        if len(spanning_domains) >= 3:
            innovation_type = 'convergence_innovation'
            disruptive_potential = 'high'
        elif len(spanning_domains) == 2:
            innovation_type = 'hybrid_innovation'
            disruptive_potential = 'medium'
        else:
            innovation_type = 'incremental_innovation'
            disruptive_potential = 'low'
        
        return {
            'innovation_score': min(innovation_score, 1.0),
            'innovation_type': innovation_type,
            'disruptive_potential': disruptive_potential,
            'confidence': pattern.get('confidence', 0.5)
        }
    
    async def _identify_competitive_opportunities(self, problems: List, pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify competitive advantage opportunities"""
        opportunities = []
        
        # Analyze competitor weaknesses and market positioning
        competitor_problems = [
            p for p in problems 
            if 'competitor' in p.problem_statement.lower() or 'competition' in p.problem_statement.lower()
        ]
        
        if not competitor_problems:
            return opportunities
        
        # Look for competitive differentiation opportunities
        differentiation_opportunities = await self._analyze_differentiation_opportunities(competitor_problems)
        
        for opportunity in differentiation_opportunities:
            opportunity = {
                'insight_type': 'competitive_opportunity',
                'opportunity_category': 'competitive_advantage',
                'title': f"Competitive Advantage: {opportunity['advantage_type']}",
                'description': f"Leverage competitive advantage: {opportunity['advantage_type']}",
                'findings': opportunity,
                'sustainability': opportunity['sustainability'],
                'market_impact': opportunity['market_impact'],
                'confidence': opportunity['confidence'],
                'evidence': opportunity['evidence'],
                'generated_at': datetime.now().isoformat()
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _analyze_differentiation_opportunities(self, competitor_problems: List) -> List[Dict[str, Any]]:
        """Analyze differentiation opportunities from competitor analysis"""
        opportunities = []
        
        # Extract competitor weakness patterns
        weakness_patterns = []
        for problem in competitor_problems:
            weaknesses = nlp_utils.extract_keywords(problem.problem_statement, max_keywords=5)
            weakness_patterns.extend(weaknesses)
        
        # Identify common weaknesses
        weakness_frequency = {}
        for weakness in weakness_patterns:
            weakness_frequency[weakness] = weakness_frequency.get(weakness, 0) + 1
        
        # Generate differentiation opportunities
        for weakness, frequency in weakness_frequency.items():
            if frequency >= 2:  # Appears in multiple competitor problems
                opportunity = {
                    'advantage_type': f"address_{weakness}",
                    'competitor_weakness': weakness,
                    'frequency': frequency,
                    'differentiation_strategy': self._generate_differentiation_strategy(weakness),
                    'sustainability': self._assess_sustainability(weakness),
                    'market_impact': self._assess_market_impact(weakness, frequency),
                    'confidence': min(frequency / len(competitor_problems), 1.0),
                    'evidence': []
                }
                opportunities.append(opportunity)
        
        return opportunities
    
    def _generate_differentiation_strategy(self, weakness: str) -> str:
        """Generate differentiation strategy for a weakness"""
        strategy_mappings = {
            'performance': 'optimize performance and reliability',
            'feature': 'develop superior features and functionality',
            'design': 'create better user experience and interface',
            'support': 'provide exceptional customer service',
            'price': 'offer better value proposition',
            'integration': 'improve third-party integrations'
        }
        
        return strategy_mappings.get(weakness, 'develop competitive advantages')
    
    def _assess_sustainability(self, weakness: str) -> str:
        """Assess sustainability of addressing a weakness"""
        sustainable_weaknesses = ['performance', 'design', 'support']
        temporary_weaknesses = ['price', 'feature_gap']
        
        if weakness in sustainable_weaknesses:
            return 'high'
        elif weakness in temporary_weaknesses:
            return 'medium'
        else:
            return 'low'
    
    def _assess_market_impact(self, weakness: str, frequency: int) -> str:
        """Assess market impact of addressing a weakness"""
        high_impact_weaknesses = ['performance', 'design', 'support']
        medium_impact_weaknesses = ['feature', 'integration']
        
        if weakness in high_impact_weaknesses:
            return 'high'
        elif weakness in medium_impact_weaknesses:
            return 'medium'
        else:
            return 'low'
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for opportunity insights"""
        return {
            'status': 'working',
            'opportunity_types': [
                'unmet_need_opportunities',
                'market_gap_opportunities',
                'innovation_opportunities',
                'competitive_opportunities'
            ],
            'analysis_methods': [
                'need_pattern_analysis',
                'market_gap_analysis',
                'innovation_potential_analysis',
                'competitive_differentiation'
            ],
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
