"""
Market Opportunity Analysis Implementation
Analyzes market opportunities from insights
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
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False


class MarketOpportunity:
    """Market opportunity analysis implementation following detailed specification"""
    
    async def analyze_market_opportunities(self, insights: List) -> List[Dict[str, Any]]:
        """Analyze market opportunities from insights"""
        if not insights:
            return []
        
        try:
            opportunities = []
            
            # Extract opportunity signals from insights
            opportunity_signals = await self._extract_opportunity_signals(insights)
            
            # Analyze market size and growth
            market_analysis = await self._analyze_market_characteristics(opportunity_signals)
            
            # Identify unmet needs
            unmet_needs = await self._identify_unmet_needs(opportunity_signals)
            
            # Assess market readiness
            market_readiness = await self._assess_market_readiness(opportunity_signals)
            
            # Generate opportunity scores
            for signal in opportunity_signals:
                opportunity = await self._generate_opportunity_score(
                    signal, market_analysis, unmet_needs, market_readiness
                )
                
                if opportunity['opportunity_score'] > settings.OPPORTUNITY_THRESHOLD:
                    opportunities.append(opportunity)
            
            # Rank opportunities by score
            ranked_opportunities = sorted(
                opportunities, 
                key=lambda x: x.get('opportunity_score', 0), 
                reverse=True
            )
            
            return ranked_opportunities
            
        except Exception as e:
            print(f"Market opportunity analysis failed: {e}")
            return []
    
    async def _extract_opportunity_signals(self, insights: List) -> List[Dict[str, Any]]:
        """Extract opportunity signals from insights"""
        signals = []
        
        for insight in insights:
            insight_text = insight.insight_statement.lower()
            
            # Opportunity indicators
            opportunity_keywords = [
                'opportunity', 'potential', 'market', 'demand', 'need', 'gap',
                'growth', 'expansion', 'emerging', 'trend', 'advantage'
            ]
            
            # Count opportunity indicators
            opportunity_count = sum(1 for keyword in opportunity_keywords if keyword in insight_text)
            
            if opportunity_count > 0:
                # Extract relevant keywords
                keywords = nlp_utils.extract_keywords(insight.insight_statement, max_keywords=5)
                
                signal = {
                    'insight_id': insight.id,
                    'insight_text': insight.insight_statement,
                    'opportunity_indicators': opportunity_count,
                    'keywords': keywords,
                    'confidence': getattr(insight, 'confidence_score', 0.5),
                    'category': getattr(insight, 'insight_category', 'general')
                }
                signals.append(signal)
        
        return signals
    
    async def _analyze_market_characteristics(self, opportunity_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze market characteristics"""
        if not opportunity_signals:
            return {}
        
        # Estimate market size
        market_size = self._estimate_market_size(opportunity_signals)
        
        # Analyze growth potential
        growth_potential = self._analyze_growth_potential(opportunity_signals)
        
        # Assess market maturity
        market_maturity = self._assess_market_maturity(opportunity_signals)
        
        return {
            'market_size': market_size,
            'growth_potential': growth_potential,
            'market_maturity': market_maturity,
            'analysis_confidence': self._calculate_market_analysis_confidence(opportunity_signals)
        }
    
    def _estimate_market_size(self, opportunity_signals: List[Dict[str, Any]]) -> str:
        """Estimate market size from signals"""
        if not opportunity_signals:
            return 'unknown'
        
        # Count market-related keywords
        market_keywords = ['market', 'industry', 'sector', 'space', 'ecosystem']
        market_mentions = sum(
            1 for signal in opportunity_signals 
            for keyword in market_keywords 
            if keyword in signal['insight_text']
        )
        
        # Estimate based on mentions and signal strength
        total_signals = len(opportunity_signals)
        market_ratio = market_mentions / total_signals if total_signals > 0 else 0
        
        if market_ratio > 0.7:
            return 'large'
        elif market_ratio > 0.4:
            return 'medium'
        elif market_ratio > 0.2:
            return 'small'
        else:
            return 'niche'
    
    def _analyze_growth_potential(self, opportunity_signals: List[Dict[str, Any]]) -> str:
        """Analyze growth potential from signals"""
        if not opportunity_signals:
            return 'unknown'
        
        # Growth indicators
        growth_keywords = ['growth', 'expanding', 'emerging', 'trend', 'increasing', 'rising']
        
        growth_score = 0
        for signal in opportunity_signals:
            signal_growth = sum(1 for keyword in growth_keywords if keyword in signal['insight_text'])
            signal_growth += signal.get('confidence', 0)  # Boost by confidence
            growth_score += signal_growth
        
        avg_growth = growth_score / len(opportunity_signals) if opportunity_signals else 0
        
        if avg_growth > 0.8:
            return 'high'
        elif avg_growth > 0.5:
            return 'medium'
        elif avg_growth > 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _assess_market_maturity(self, opportunity_signals: List[Dict[str, Any]]) -> str:
        """Assess market maturity from signals"""
        if not opportunity_signals:
            return 'unknown'
        
        # Maturity indicators
        mature_keywords = ['established', 'mature', 'saturated', 'stable', 'traditional']
        emerging_keywords = ['new', 'emerging', 'developing', 'early', 'nascent']
        
        mature_score = 0
        emerging_score = 0
        
        for signal in opportunity_signals:
            signal_text = signal['insight_text']
            
            mature_mentions = sum(1 for keyword in mature_keywords if keyword in signal_text)
            emerging_mentions = sum(1 for keyword in emerging_keywords if keyword in signal_text)
            
            mature_score += mature_mentions
            emerging_score += emerging_mentions
        
        # Determine maturity based on ratio
        total_signals = len(opportunity_signals)
        if total_signals == 0:
            return 'unknown'
        
        mature_ratio = mature_score / total_signals
        emerging_ratio = emerging_score / total_signals
        
        if emerging_ratio > mature_ratio:
            return 'emerging'
        elif mature_ratio > 0.6:
            return 'mature'
        elif mature_ratio > 0.3:
            return 'developing'
        else:
            return 'transitional'
    
    async def _identify_unmet_needs(self, opportunity_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify unmet needs from opportunity signals"""
        unmet_needs = []
        
        # Need indicators
        need_keywords = ['need', 'want', 'require', 'lack', 'missing', 'problem', 'issue', 'challenge']
        
        for signal in opportunity_signals:
            signal_text = signal['insight_text']
            
            need_mentions = [keyword for keyword in need_keywords if keyword in signal_text]
            
            if need_mentions:
                # Extract specific needs
                needs = nlp_utils.extract_keywords(signal_text, max_keywords=3)
                
                unmet_need = {
                    'need_statement': f"Users {signal_text}",
                    'specific_needs': needs,
                    'need_category': self._categorize_need(needs),
                    'urgency': self._assess_need_urgency(signal_text),
                    'frequency': self._assess_need_frequency(signal_text, opportunity_signals),
                    'confidence': signal.get('confidence', 0.5)
                }
                unmet_needs.append(unmet_need)
        
        return unmet_needs
    
    def _categorize_need(self, needs: List[str]) -> str:
        """Categorize user needs"""
        if not needs:
            return 'general'
        
        need_text = ' '.join(needs).lower()
        
        categories = {
            'functional': ['function', 'feature', 'capability', 'tool'],
            'usability': ['easy', 'simple', 'intuitive', 'usable'],
            'performance': ['fast', 'quick', 'responsive', 'reliable'],
            'integration': ['connect', 'integrate', 'work with', 'compatible'],
            'cost': ['price', 'cost', 'affordable', 'value']
        }
        
        for category, keywords in categories.items():
            if any(keyword in need_text for keyword in keywords):
                return category
        
        return 'general'
    
    def _assess_need_urgency(self, signal_text: str) -> str:
        """Assess urgency of need"""
        urgent_keywords = ['urgent', 'critical', 'immediate', 'asap', 'emergency']
        high_keywords = ['important', 'significant', 'major', 'high']
        
        if any(keyword in signal_text for keyword in urgent_keywords):
            return 'high'
        elif any(keyword in signal_text for keyword in high_keywords):
            return 'medium'
        else:
            return 'low'
    
    def _assess_need_frequency(self, signal_text: str, all_signals: List[Dict[str, Any]]) -> str:
        """Assess frequency of need across signals"""
        # Count similar needs across all signals
        similar_needs = 0
        signal_keywords = set(nlp_utils.extract_keywords(signal_text, max_keywords=3))
        
        for other_signal in all_signals:
            other_text = other_signal['insight_text']
            other_keywords = set(nlp_utils.extract_keywords(other_text, max_keywords=3))
            
            # Calculate keyword overlap
            overlap = len(signal_keywords.intersection(other_keywords))
            if overlap > 0:
                similar_needs += 1
        
        if similar_needs > 5:
            return 'high'
        elif similar_needs > 2:
            return 'medium'
        else:
            return 'low'
    
    async def _assess_market_readiness(self, opportunity_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess market readiness"""
        if not opportunity_signals:
            return {}
        
        # Technology readiness
        tech_readiness = self._assess_technology_readiness(opportunity_signals)
        
        # Customer readiness
        customer_readiness = self._assess_customer_readiness(opportunity_signals)
        
        # Regulatory readiness
        regulatory_readiness = self._assess_regulatory_readiness(opportunity_signals)
        
        return {
            'technology_readiness': tech_readiness,
            'customer_readiness': customer_readiness,
            'regulatory_readiness': regulatory_readiness,
            'overall_readiness': self._calculate_overall_readiness(
                tech_readiness, customer_readiness, regulatory_readiness
            )
        }
    
    def _assess_technology_readiness(self, opportunity_signals: List[Dict[str, Any]]) -> str:
        """Assess technology readiness"""
        tech_keywords = ['technology', 'infrastructure', 'platform', 'tools', 'systems']
        
        tech_mentions = 0
        for signal in opportunity_signals:
            tech_mentions += sum(1 for keyword in tech_keywords if keyword in signal['insight_text'])
        
        tech_ratio = tech_mentions / len(opportunity_signals) if opportunity_signals else 0
        
        if tech_ratio > 0.6:
            return 'ready'
        elif tech_ratio > 0.3:
            return 'developing'
        else:
            return 'emerging'
    
    def _assess_customer_readiness(self, opportunity_signals: List[Dict[str, Any]]) -> str:
        """Assess customer readiness"""
        customer_keywords = ['customer', 'user', 'market', 'adoption', 'demand']
        
        customer_mentions = 0
        for signal in opportunity_signals:
            customer_mentions += sum(1 for keyword in customer_keywords if keyword in signal['insight_text'])
        
        customer_ratio = customer_mentions / len(opportunity_signals) if opportunity_signals else 0
        
        if customer_ratio > 0.7:
            return 'ready'
        elif customer_ratio > 0.4:
            return 'developing'
        else:
            return 'early'
    
    def _assess_regulatory_readiness(self, opportunity_signals: List[Dict[str, Any]]) -> str:
        """Assess regulatory readiness"""
        regulatory_keywords = ['regulation', 'compliance', 'legal', 'policy', 'standard']
        
        regulatory_mentions = 0
        for signal in opportunity_signals:
            regulatory_mentions += sum(1 for keyword in regulatory_keywords if keyword in signal['insight_text'])
        
        regulatory_ratio = regulatory_mentions / len(opportunity_signals) if opportunity_signals else 0
        
        if regulatory_ratio > 0.5:
            return 'established'
        elif regulatory_ratio > 0.2:
            return 'developing'
        else:
            return 'unclear'
    
    def _calculate_overall_readiness(self, tech: str, customer: str, regulatory: str) -> str:
        """Calculate overall market readiness"""
        readiness_scores = {
            'ready': 3, 'developing': 2, 'emerging': 1, 'early': 1, 'unclear': 0,
            'established': 3
        }
        
        tech_score = readiness_scores.get(tech, 2)
        customer_score = readiness_scores.get(customer, 2)
        regulatory_score = readiness_scores.get(regulatory, 2)
        
        avg_score = (tech_score + customer_score + regulatory_score) / 3
        
        if avg_score >= 2.5:
            return 'ready'
        elif avg_score >= 1.5:
            return 'developing'
        else:
            return 'early'
    
    async def _generate_opportunity_score(self, signal: Dict[str, Any],
                                      market_analysis: Dict[str, Any],
                                      unmet_needs: List[Dict[str, Any]],
                                      market_readiness: Dict[str, Any]) -> Dict[str, Any]:
        """Generate opportunity score for a signal"""
        # Base score from signal confidence
        base_score = signal.get('confidence', 0.5)
        
        # Market size boost
        market_size = market_analysis.get('market_size', 'medium')
        size_boost = {'large': 0.3, 'medium': 0.2, 'small': 0.1, 'niche': 0.05}
        market_size_boost = size_boost.get(market_size, 0.1)
        
        # Growth potential boost
        growth_potential = market_analysis.get('growth_potential', 'medium')
        growth_boost = {'high': 0.3, 'medium': 0.2, 'low': 0.1, 'minimal': 0.05}
        growth_boost = growth_boost.get(growth_potential, 0.1)
        
        # Unmet needs boost
        relevant_needs = [need for need in unmet_needs if need['confidence'] > 0.6]
        needs_boost = min(len(relevant_needs) / 5, 0.3)
        
        # Market readiness boost
        overall_readiness = market_readiness.get('overall_readiness', 'developing')
        readiness_boost = {'ready': 0.2, 'developing': 0.1, 'early': 0.05}
        readiness_boost = readiness_boost.get(overall_readiness, 0.1)
        
        # Calculate final score
        opportunity_score = min(base_score + market_size_boost + growth_boost + needs_boost + readiness_boost, 1.0)
        
        return {
            'signal_id': signal.get('insight_id'),
            'opportunity_area': self._determine_opportunity_area(signal),
            'opportunity_score': opportunity_score,
            'market_size': market_size,
            'growth_potential': growth_potential,
            'unmet_needs_count': len(relevant_needs),
            'market_readiness': overall_readiness,
            'potential': self._assess_opportunity_potential(opportunity_score),
            'evidence': {
                'keywords': signal.get('keywords', []),
                'opportunity_indicators': signal.get('opportunity_indicators', 0),
                'unmet_needs': relevant_needs[:3]
            }
        }
    
    def _determine_opportunity_area(self, signal: Dict[str, Any]) -> str:
        """Determine opportunity area from signal"""
        keywords = signal.get('keywords', [])
        keyword_text = ' '.join(keywords).lower()
        
        area_keywords = {
            'product': ['product', 'feature', 'functionality', 'capability'],
            'market': ['market', 'industry', 'sector', 'space'],
            'technology': ['technology', 'platform', 'infrastructure', 'system'],
            'business': ['business', 'model', 'revenue', 'profit'],
            'user': ['user', 'customer', 'experience', 'journey']
        }
        
        for area, area_keywords in area_keywords.items():
            if any(keyword in keyword_text for keyword in area_keywords):
                return area
        
        return 'general'
    
    def _assess_opportunity_potential(self, opportunity_score: float) -> str:
        """Assess opportunity potential from score"""
        if opportunity_score > 0.8:
            return 'high'
        elif opportunity_score > 0.6:
            return 'medium-high'
        elif opportunity_score > 0.4:
            return 'medium'
        elif opportunity_score > 0.2:
            return 'low-medium'
        else:
            return 'low'
    
    def _calculate_market_analysis_confidence(self, opportunity_signals: List[Dict[str, Any]]) -> float:
        """Calculate confidence in market analysis"""
        if not opportunity_signals:
            return 0.5
        
        # Average confidence from signals
        signal_confidences = [signal.get('confidence', 0.5) for signal in opportunity_signals]
        avg_confidence = np.mean(signal_confidences)
        
        # Boost based on number of signals
        signal_boost = min(len(opportunity_signals) / 10, 0.2)
        
        return min(avg_confidence + signal_boost, 1.0)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for market opportunity analysis"""
        return {
            'status': 'working',
            'analysis_methods': [
                'opportunity_signal_extraction',
                'market_characteristics_analysis',
                'unmet_needs_identification',
                'market_readiness_assessment'
            ],
            'sklearn_available': ANALYSIS_AVAILABLE,
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
