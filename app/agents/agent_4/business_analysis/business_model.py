"""
Business Model Analysis Implementation
Analyzes business model from insights
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


class BusinessModel:
    """Business model analysis implementation following detailed specification"""
    
    async def analyze_business_model(self, insights: List) -> Dict[str, Any]:
        """Analyze business model from insights"""
        if not insights:
            return {}
        
        try:
            # Extract business model signals
            business_signals = await self._extract_business_signals(insights)
            
            # Analyze revenue streams
            revenue_analysis = await self._analyze_revenue_streams(business_signals)
            
            # Analyze value proposition
            value_analysis = await self._analyze_value_proposition(business_signals)
            
            # Analyze cost structure
            cost_analysis = await self._analyze_cost_structure(business_signals)
            
            # Analyze target market
            market_analysis = await self._analyze_target_market(business_signals)
            
            # Analyze competitive advantage
            advantage_analysis = await self._analyze_competitive_advantage(business_signals)
            
            return {
                'business_signals': business_signals,
                'revenue_streams': revenue_analysis,
                'value_proposition': value_analysis,
                'cost_structure': cost_analysis,
                'target_market': market_analysis,
                'competitive_advantage': advantage_analysis,
                'model_maturity': self._assess_model_maturity(business_signals),
                'analysis_confidence': self._calculate_analysis_confidence(insights)
            }
            
        except Exception as e:
            print(f"Business model analysis failed: {e}")
            return {'error': str(e)}
    
    async def _extract_business_signals(self, insights: List) -> List[Dict[str, Any]]:
        """Extract business model signals from insights"""
        signals = []
        
        for insight in insights:
            insight_text = insight.insight_statement.lower()
            
            # Business model indicators
            business_keywords = [
                'revenue', 'profit', 'business', 'model', 'monetization', 'pricing',
                'customer', 'market', 'value', 'cost', 'investment', 'roi'
            ]
            
            # Check for business model mentions
            business_mentions = sum(1 for keyword in business_keywords if keyword in insight_text)
            
            if business_mentions > 0:
                # Extract business-related keywords
                keywords = nlp_utils.extract_keywords(insight.insight_statement, max_keywords=5)
                
                signal = {
                    'insight_id': insight.id,
                    'insight_text': insight.insight_statement,
                    'business_indicators': business_mentions,
                    'keywords': keywords,
                    'confidence': getattr(insight, 'confidence_score', 0.5),
                    'category': getattr(insight, 'insight_category', 'general')
                }
                signals.append(signal)
        
        return signals
    
    async def _analyze_revenue_streams(self, business_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze revenue streams from business signals"""
        if not business_signals:
            return {}
        
        # Revenue stream indicators
        revenue_keywords = {
            'subscription': ['subscription', 'recurring', 'mrr', 'arr'],
            'transactional': ['sale', 'purchase', 'transaction', 'one-time'],
            'advertising': ['ad', 'advertising', 'monetization', 'revenue'],
            'freemium': ['free', 'premium', 'tier', 'upgrade', 'conversion'],
            'enterprise': ['enterprise', 'b2b', 'contract', 'license'],
            'commission': ['commission', 'fee', 'percentage', 'referral']
        }
        
        revenue_streams = []
        for signal in business_signals:
            signal_text = signal['insight_text']
            keywords = signal.get('keywords', [])
            
            for stream_type, type_keywords in revenue_keywords.items():
                if any(keyword in signal_text for keyword in type_keywords):
                    revenue_streams.append({
                        'stream_type': stream_type,
                        'indicators': [kw for kw in type_keywords if kw in signal_text],
                        'confidence': signal.get('confidence', 0.5),
                        'source_insight': signal.get('insight_id'),
                        'evidence': keywords[:3]
                    })
        
        # Analyze revenue characteristics
        revenue_analysis = {
            'identified_streams': revenue_streams,
            'stream_diversity': len(set([rs['stream_type'] for rs in revenue_streams])),
            'revenue_model': self._determine_revenue_model(revenue_streams),
            'monetization_maturity': self._assess_monetization_maturity(revenue_streams)
        }
        
        return revenue_analysis
    
    def _determine_revenue_model(self, revenue_streams: List[Dict[str, Any]]) -> str:
        """Determine primary revenue model"""
        if not revenue_streams:
            return 'unknown'
        
        stream_types = [rs['stream_type'] for rs in revenue_streams]
        
        if 'subscription' in stream_types:
            return 'subscription_based'
        elif 'transactional' in stream_types:
            return 'transactional'
        elif 'advertising' in stream_types:
            return 'advertising_based'
        elif 'freemium' in stream_types:
            return 'freemium'
        elif 'enterprise' in stream_types:
            return 'enterprise'
        elif len(set(stream_types)) > 3:
            return 'diversified'
        else:
            return 'emerging'
    
    def _assess_monetization_maturity(self, revenue_streams: List[Dict[str, Any]]) -> str:
        """Assess monetization maturity"""
        if not revenue_streams:
            return 'unknown'
        
        # Mature monetization indicators
        mature_indicators = ['subscription', 'enterprise', 'recurring']
        emerging_indicators = ['advertising', 'freemium', 'commission']
        
        mature_count = sum(1 for rs in revenue_streams if rs['stream_type'] in mature_indicators)
        emerging_count = sum(1 for rs in revenue_streams if rs['stream_type'] in emerging_indicators)
        
        total_streams = len(revenue_streams)
        if total_streams == 0:
            return 'unknown'
        
        mature_ratio = mature_count / total_streams
        
        if mature_ratio > 0.7:
            return 'mature'
        elif mature_ratio > 0.4:
            return 'developing'
        else:
            return 'emerging'
    
    async def _analyze_value_proposition(self, business_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze value proposition from business signals"""
        if not business_signals:
            return {}
        
        # Value proposition indicators
        value_keywords = {
            'efficiency': ['efficient', 'save', 'productivity', 'optimize', 'automate'],
            'cost_savings': ['cheap', 'affordable', 'cost-effective', 'budget', 'value'],
            'quality': ['quality', 'reliable', 'premium', 'excellent', 'superior'],
            'convenience': ['easy', 'simple', 'convenient', 'accessible', 'fast'],
            'innovation': ['innovative', 'unique', 'new', 'different', 'breakthrough']
        }
        
        value_propositions = []
        for signal in business_signals:
            signal_text = signal['insight_text']
            keywords = signal.get('keywords', [])
            
            for value_type, type_keywords in value_keywords.items():
                if any(keyword in signal_text for keyword in type_keywords):
                    value_propositions.append({
                        'value_type': value_type,
                        'indicators': [kw for kw in type_keywords if kw in signal_text],
                        'confidence': signal.get('confidence', 0.5),
                        'source_insight': signal.get('insight_id'),
                        'evidence': keywords[:3]
                    })
        
        # Analyze value characteristics
        value_analysis = {
            'identified_values': value_propositions,
            'value_diversity': len(set([vp['value_type'] for vp in value_propositions])),
            'primary_value': self._determine_primary_value(value_propositions),
            'value_strength': self._assess_value_strength(value_propositions),
            'differentiation_potential': self._assess_differentiation_potential(value_propositions)
        }
        
        return value_analysis
    
    def _determine_primary_value(self, value_propositions: List[Dict[str, Any]]) -> str:
        """Determine primary value proposition"""
        if not value_propositions:
            return 'unknown'
        
        # Count value types
        value_counts = {}
        for vp in value_propositions:
            value_type = vp['value_type']
            value_counts[value_type] = value_counts.get(value_type, 0) + 1
        
        if not value_counts:
            return 'unknown'
        
        # Find most common value
        primary_value = max(value_counts, key=value_counts.get)
        
        return primary_value
    
    def _assess_value_strength(self, value_propositions: List[Dict[str, Any]]) -> str:
        """Assess strength of value proposition"""
        if not value_propositions:
            return 'unknown'
        
        # Calculate average confidence
        avg_confidence = np.mean([vp.get('confidence', 0.5) for vp in value_propositions])
        
        # Count strong indicators
        strong_indicators = ['quality', 'premium', 'excellent', 'superior']
        strong_count = sum(
            1 for vp in value_propositions 
            for indicator in strong_indicators 
            if indicator in ' '.join(vp.get('indicators', [])).lower()
        )
        
        total_values = len(value_propositions)
        strong_ratio = strong_count / total_values if total_values > 0 else 0
        
        if avg_confidence > 0.7 and strong_ratio > 0.5:
            return 'strong'
        elif avg_confidence > 0.5:
            return 'moderate'
        else:
            return 'weak'
    
    def _assess_differentiation_potential(self, value_propositions: List[Dict[str, Any]]) -> str:
        """Assess differentiation potential"""
        if not value_propositions:
            return 'unknown'
        
        # Innovation indicators
        innovation_indicators = ['innovative', 'unique', 'new', 'different', 'breakthrough']
        
        innovation_count = sum(
            1 for vp in value_propositions 
            for indicator in innovation_indicators 
            if indicator in ' '.join(vp.get('indicators', [])).lower()
        )
        
        total_values = len(value_propositions)
        innovation_ratio = innovation_count / total_values if total_values > 0 else 0
        
        if innovation_ratio > 0.5:
            return 'high'
        elif innovation_ratio > 0.3:
            return 'medium'
        else:
            return 'low'
    
    async def _analyze_cost_structure(self, business_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cost structure from business signals"""
        if not business_signals:
            return {}
        
        # Cost structure indicators
        cost_keywords = {
            'fixed_costs': ['fixed', 'overhead', 'infrastructure', 'maintenance', 'staff'],
            'variable_costs': ['variable', 'usage', 'transaction', 'scaling', 'per_user'],
            'development_costs': ['development', 'r&d', 'innovation', 'engineering'],
            'marketing_costs': ['marketing', 'sales', 'acquisition', 'promotion'],
            'operational_costs': ['operations', 'support', 'service', 'delivery']
        }
        
        cost_structure = []
        for signal in business_signals:
            signal_text = signal['insight_text']
            keywords = signal.get('keywords', [])
            
            for cost_type, type_keywords in cost_keywords.items():
                if any(keyword in signal_text for keyword in type_keywords):
                    cost_structure.append({
                        'cost_type': cost_type,
                        'indicators': [kw for kw in type_keywords if kw in signal_text],
                        'confidence': signal.get('confidence', 0.5),
                        'source_insight': signal.get('insight_id'),
                        'evidence': keywords[:3]
                    })
        
        # Analyze cost characteristics
        cost_analysis = {
            'identified_costs': cost_structure,
            'cost_diversity': len(set([cs['cost_type'] for cs in cost_structure])),
            'cost_model': self._determine_cost_model(cost_structure),
            'scalability_profile': self._assess_scalability_profile(cost_structure)
        }
        
        return cost_analysis
    
    def _determine_cost_model(self, cost_structure: List[Dict[str, Any]]) -> str:
        """Determine cost model"""
        if not cost_structure:
            return 'unknown'
        
        cost_types = [cs['cost_type'] for cs in cost_structure]
        
        if 'fixed_costs' in cost_types and 'variable_costs' in cost_types:
            return 'mixed'
        elif 'fixed_costs' in cost_types:
            return 'fixed_dominant'
        elif 'variable_costs' in cost_types:
            return 'variable_dominant'
        elif len(set(cost_types)) > 3:
            return 'complex'
        else:
            return 'simple'
    
    def _assess_scalability_profile(self, cost_structure: List[Dict[str, Any]]) -> str:
        """Assess scalability profile"""
        if not cost_structure:
            return 'unknown'
        
        # Scalability indicators
        scalable_indicators = ['variable', 'per_user', 'usage', 'scaling']
        fixed_indicators = ['fixed', 'overhead', 'infrastructure', 'maintenance']
        
        scalable_count = sum(
            1 for cs in cost_structure 
            for indicator in scalable_indicators 
            if indicator in ' '.join(cs.get('indicators', [])).lower()
        )
        
        fixed_count = sum(
            1 for cs in cost_structure 
            for indicator in fixed_indicators 
            if indicator in ' '.join(cs.get('indicators', [])).lower()
        )
        
        total_costs = len(cost_structure)
        if total_costs == 0:
            return 'unknown'
        
        scalable_ratio = scalable_count / total_costs
        
        if scalable_ratio > 0.7:
            return 'highly_scalable'
        elif scalable_ratio > 0.4:
            return 'moderately_scalable'
        else:
            return 'limited_scalability'
    
    async def _analyze_target_market(self, business_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze target market from business signals"""
        if not business_signals:
            return {}
        
        # Target market indicators
        market_keywords = {
            'b2b': ['business', 'b2b', 'enterprise', 'company', 'organization'],
            'b2c': ['consumer', 'b2c', 'individual', 'customer', 'user'],
            'niche': ['niche', 'specialized', 'focused', 'segment'],
            'mass_market': ['mass', 'broad', 'general', 'mainstream'],
            'emerging': ['emerging', 'new', 'developing', 'early'],
            'mature': ['mature', 'established', 'existing', 'developed']
        }
        
        target_markets = []
        for signal in business_signals:
            signal_text = signal['insight_text']
            keywords = signal.get('keywords', [])
            
            for market_type, type_keywords in market_keywords.items():
                if any(keyword in signal_text for keyword in type_keywords):
                    target_markets.append({
                        'market_type': market_type,
                        'indicators': [kw for kw in type_keywords if kw in signal_text],
                        'confidence': signal.get('confidence', 0.5),
                        'source_insight': signal.get('insight_id'),
                        'evidence': keywords[:3]
                    })
        
        # Analyze market characteristics
        market_analysis = {
            'identified_markets': target_markets,
            'market_diversity': len(set([tm['market_type'] for tm in target_markets])),
            'primary_market': self._determine_primary_market(target_markets),
            'market_size_indication': self._estimate_market_size_indication(target_markets),
            'growth_potential': self._assess_market_growth_potential(target_markets)
        }
        
        return market_analysis
    
    def _determine_primary_market(self, target_markets: List[Dict[str, Any]]) -> str:
        """Determine primary target market"""
        if not target_markets:
            return 'unknown'
        
        # Count market types
        market_counts = {}
        for tm in target_markets:
            market_type = tm['market_type']
            market_counts[market_type] = market_counts.get(market_type, 0) + 1
        
        if not market_counts:
            return 'unknown'
        
        # Find most common market
        primary_market = max(market_counts, key=market_counts.get)
        
        return primary_market
    
    def _estimate_market_size_indication(self, target_markets: List[Dict[str, Any]]) -> str:
        """Estimate market size indication"""
        if not target_markets:
            return 'unknown'
        
        # Size indicators
        size_indicators = {
            'mass_market': ['mass', 'broad', 'general', 'mainstream', 'large'],
            'enterprise': ['enterprise', 'business', 'company', 'organization'],
            'niche': ['niche', 'specialized', 'focused', 'small', 'targeted']
        }
        
        size_mentions = {}
        for tm in target_markets:
            indicators = tm.get('indicators', [])
            for size_type, type_keywords in size_indicators.items():
                if any(keyword in ' '.join(indicators).lower() for keyword in type_keywords):
                    size_mentions[size_type] = size_mentions.get(size_type, 0) + 1
        
        if not size_mentions:
            return 'unknown'
        
        # Determine most indicated size
        indicated_size = max(size_mentions, key=size_mentions.get)
        
        return indicated_size
    
    def _assess_market_growth_potential(self, target_markets: List[Dict[str, Any]]) -> str:
        """Assess market growth potential"""
        if not target_markets:
            return 'unknown'
        
        # Growth indicators
        growth_indicators = {
            'emerging': ['emerging', 'new', 'developing', 'early', 'high'],
            'mature': ['mature', 'established', 'existing', 'developed', 'low'],
            'stable': ['stable', 'steady', 'consistent', 'moderate']
        }
        
        growth_mentions = {}
        for tm in target_markets:
            indicators = tm.get('indicators', [])
            for growth_type, type_keywords in growth_indicators.items():
                if any(keyword in ' '.join(indicators).lower() for keyword in type_keywords):
                    growth_mentions[growth_type] = growth_mentions.get(growth_type, 0) + 1
        
        if not growth_mentions:
            return 'unknown'
        
        # Determine most indicated growth
        indicated_growth = max(growth_mentions, key=growth_mentions.get)
        
        return indicated_growth
    
    async def _analyze_competitive_advantage(self, business_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze competitive advantage from business signals"""
        if not business_signals:
            return {}
        
        # Competitive advantage indicators
        advantage_keywords = {
            'cost_advantage': ['cost', 'price', 'cheaper', 'affordable', 'value'],
            'product_advantage': ['feature', 'quality', 'performance', 'reliability'],
            'market_advantage': ['market', 'position', 'share', 'reach', 'distribution'],
            'technology_advantage': ['technology', 'innovation', 'patent', 'proprietary'],
            'brand_advantage': ['brand', 'reputation', 'trust', 'recognition']
        }
        
        competitive_advantages = []
        for signal in business_signals:
            signal_text = signal['insight_text']
            keywords = signal.get('keywords', [])
            
            for advantage_type, type_keywords in advantage_keywords.items():
                if any(keyword in signal_text for keyword in type_keywords):
                    competitive_advantages.append({
                        'advantage_type': advantage_type,
                        'indicators': [kw for kw in type_keywords if kw in signal_text],
                        'confidence': signal.get('confidence', 0.5),
                        'source_insight': signal.get('insight_id'),
                        'evidence': keywords[:3]
                    })
        
        # Analyze advantage characteristics
        advantage_analysis = {
            'identified_advantages': competitive_advantages,
            'advantage_diversity': len(set([ca['advantage_type'] for ca in competitive_advantages])),
            'primary_advantage': self._determine_primary_advantage(competitive_advantages),
            'advantage_strength': self._assess_advantage_strength(competitive_advantages),
            'sustainability': self._assess_advantage_sustainability(competitive_advantages)
        }
        
        return advantage_analysis
    
    def _determine_primary_advantage(self, competitive_advantages: List[Dict[str, Any]]) -> str:
        """Determine primary competitive advantage"""
        if not competitive_advantages:
            return 'unknown'
        
        # Count advantage types
        advantage_counts = {}
        for ca in competitive_advantages:
            advantage_type = ca['advantage_type']
            advantage_counts[advantage_type] = advantage_counts.get(advantage_type, 0) + 1
        
        if not advantage_counts:
            return 'unknown'
        
        # Find most common advantage
        primary_advantage = max(advantage_counts, key=advantage_counts.get)
        
        return primary_advantage
    
    def _assess_advantage_strength(self, competitive_advantages: List[Dict[str, Any]]) -> str:
        """Assess strength of competitive advantage"""
        if not competitive_advantages:
            return 'unknown'
        
        # Calculate average confidence
        avg_confidence = np.mean([ca.get('confidence', 0.5) for ca in competitive_advantages])
        
        # Count strong indicators
        strong_indicators = ['sustainable', 'defensible', 'unique', 'patented']
        strong_count = sum(
            1 for ca in competitive_advantages 
            for indicator in strong_indicators 
            if indicator in ' '.join(ca.get('indicators', [])).lower()
        )
        
        total_advantages = len(competitive_advantages)
        strong_ratio = strong_count / total_advantages if total_advantages > 0 else 0
        
        if avg_confidence > 0.7 and strong_ratio > 0.4:
            return 'strong'
        elif avg_confidence > 0.5:
            return 'moderate'
        else:
            return 'weak'
    
    def _assess_advantage_sustainability(self, competitive_advantages: List[Dict[str, Any]]) -> str:
        """Assess sustainability of competitive advantage"""
        if not competitive_advantages:
            return 'unknown'
        
        # Sustainability indicators
        sustainable_indicators = ['sustainable', 'defensible', 'patent', 'proprietary', 'long-term']
        short_term_indicators = ['temporary', 'short-term', 'tactical', 'timing']
        
        sustainable_count = sum(
            1 for ca in competitive_advantages 
            for indicator in sustainable_indicators 
            if indicator in ' '.join(ca.get('indicators', [])).lower()
        )
        
        short_term_count = sum(
            1 for ca in competitive_advantages 
            for indicator in short_term_indicators 
            if indicator in ' '.join(ca.get('indicators', [])).lower()
        )
        
        total_advantages = len(competitive_advantages)
        if total_advantages == 0:
            return 'unknown'
        
        sustainable_ratio = sustainable_count / total_advantages
        
        if sustainable_ratio > 0.6:
            return 'highly_sustainable'
        elif sustainable_ratio > 0.3:
            return 'moderately_sustainable'
        else:
            return 'short_term_advantage'
    
    def _assess_model_maturity(self, business_signals: List[Dict[str, Any]]) -> str:
        """Assess business model maturity"""
        if not business_signals:
            return 'unknown'
        
        # Maturity indicators
        mature_indicators = ['established', 'mature', 'proven', 'stable', 'recurring']
        emerging_indicators = ['new', 'emerging', 'developing', 'testing', 'experimental']
        
        mature_count = 0
        emerging_count = 0
        
        for signal in business_signals:
            signal_text = signal['insight_text']
            
            mature_mentions = sum(1 for keyword in mature_indicators if keyword in signal_text)
            emerging_mentions = sum(1 for keyword in emerging_indicators if keyword in signal_text)
            
            mature_count += mature_mentions
            emerging_count += emerging_mentions
        
        total_signals = len(business_signals)
        if total_signals == 0:
            return 'unknown'
        
        mature_ratio = mature_count / total_signals
        
        if mature_ratio > 0.7:
            return 'mature'
        elif mature_ratio > 0.4:
            return 'developing'
        else:
            return 'emerging'
    
    def _calculate_analysis_confidence(self, insights: List) -> float:
        """Calculate confidence in business model analysis"""
        if not insights:
            return 0.5
        
        # Average confidence from insights
        insight_confidences = [getattr(insight, 'confidence_score', 0.5) for insight in insights]
        avg_confidence = np.mean(insight_confidences)
        
        # Business model analysis has moderate inherent uncertainty
        return avg_confidence * 0.85
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for business model analysis"""
        return {
            'status': 'working',
            'analysis_methods': [
                'business_signal_extraction',
                'revenue_stream_analysis',
                'value_proposition_analysis',
                'cost_structure_analysis',
                'target_market_analysis',
                'competitive_advantage_analysis'
            ],
            'sklearn_available': ANALYSIS_AVAILABLE,
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
