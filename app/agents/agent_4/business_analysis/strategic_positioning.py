"""
Strategic Positioning Analysis Implementation
Analyzes strategic positioning from insights
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
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False


class StrategicPositioning:
    """Strategic positioning analysis implementation following detailed specification"""
    
    async def analyze_strategic_positioning(self, insights: List) -> Dict[str, Any]:
        """Analyze strategic positioning from insights"""
        if not insights:
            return {}
        
        try:
            # Extract positioning signals
            positioning_signals = await self._extract_positioning_signals(insights)
            
            # Analyze market positioning
            market_positioning = await self._analyze_market_positioning(positioning_signals)
            
            # Analyze competitive positioning
            competitive_positioning = await self._analyze_competitive_positioning(positioning_signals)
            
            # Analyze value chain positioning
            value_chain_positioning = await self._analyze_value_chain_positioning(positioning_signals)
            
            # Identify strategic opportunities
            strategic_opportunities = await self._identify_strategic_opportunities(
                positioning_signals, market_positioning, competitive_positioning
            )
            
            return {
                'positioning_signals': positioning_signals,
                'market_positioning': market_positioning,
                'competitive_positioning': competitive_positioning,
                'value_chain_positioning': value_chain_positioning,
                'strategic_opportunities': strategic_opportunities,
                'positioning_maturity': self._assess_positioning_maturity(positioning_signals),
                'analysis_confidence': self._calculate_analysis_confidence(insights)
            }
            
        except Exception as e:
            print(f"Strategic positioning analysis failed: {e}")
            return {'error': str(e)}
    
    async def _extract_positioning_signals(self, insights: List) -> List[Dict[str, Any]]:
        """Extract positioning signals from insights"""
        signals = []
        
        for insight in insights:
            insight_text = insight.insight_statement.lower()
            
            # Positioning indicators
            positioning_keywords = [
                'position', 'strategy', 'market', 'segment', 'niche', 'differentiation',
                'brand', 'value', 'advantage', 'competitive', 'landscape'
            ]
            
            # Check for positioning mentions
            positioning_mentions = sum(1 for keyword in positioning_keywords if keyword in insight_text)
            
            if positioning_mentions > 0:
                # Extract positioning-related keywords
                keywords = nlp_utils.extract_keywords(insight.insight_statement, max_keywords=5)
                
                signal = {
                    'insight_id': insight.id,
                    'insight_text': insight.insight_statement,
                    'positioning_indicators': positioning_mentions,
                    'keywords': keywords,
                    'confidence': getattr(insight, 'confidence_score', 0.5),
                    'category': getattr(insight, 'insight_category', 'general')
                }
                signals.append(signal)
        
        return signals
    
    async def _analyze_market_positioning(self, positioning_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze market positioning from positioning signals"""
        if not positioning_signals:
            return {}
        
        # Market positioning indicators
        market_keywords = {
            'leader': ['leader', 'dominant', 'first', 'market_leader', 'category_leader'],
            'challenger': ['challenger', 'second', 'strong', 'competing', 'growing'],
            'follower': ['follower', 'third', 'weaker', 'smaller', 'niche'],
            'niche_player': ['niche', 'specialized', 'focused', 'segment', 'targeted']
        }
        
        market_positions = []
        for signal in positioning_signals:
            signal_text = signal['insight_text']
            keywords = signal.get('keywords', [])
            
            for position_type, type_keywords in market_keywords.items():
                if any(keyword in signal_text for keyword in type_keywords):
                    market_positions.append({
                        'position_type': position_type,
                        'indicators': [kw for kw in type_keywords if kw in signal_text],
                        'confidence': signal.get('confidence', 0.5),
                        'source_insight': signal.get('insight_id'),
                        'evidence': keywords[:3]
                    })
        
        # Analyze market position characteristics
        position_analysis = {
            'identified_positions': market_positions,
            'position_diversity': len(set([mp['position_type'] for mp in market_positions])),
            'primary_position': self._determine_primary_position(market_positions),
            'position_strength': self._assess_position_strength(market_positions),
            'market_share_indication': self._estimate_market_share_indication(market_positions)
        }
        
        return position_analysis
    
    def _determine_primary_position(self, market_positions: List[Dict[str, Any]]) -> str:
        """Determine primary market position"""
        if not market_positions:
            return 'unknown'
        
        # Count position types
        position_counts = {}
        for mp in market_positions:
            position_type = mp['position_type']
            position_counts[position_type] = position_counts.get(position_type, 0) + 1
        
        if not position_counts:
            return 'unknown'
        
        # Find most common position
        primary_position = max(position_counts, key=position_counts.get)
        
        return primary_position
    
    def _assess_position_strength(self, market_positions: List[Dict[str, Any]]) -> str:
        """Assess strength of market position"""
        if not market_positions:
            return 'unknown'
        
        # Calculate average confidence
        avg_confidence = np.mean([mp.get('confidence', 0.5) for mp in market_positions])
        
        # Count strong position indicators
        strong_indicators = ['leader', 'dominant', 'first', 'market_leader']
        strong_count = sum(
            1 for mp in market_positions 
            for indicator in strong_indicators 
            if indicator in mp.get('position_type', '')
        )
        
        total_positions = len(market_positions)
        strong_ratio = strong_count / total_positions if total_positions > 0 else 0
        
        if avg_confidence > 0.7 and strong_ratio > 0.3:
            return 'strong'
        elif avg_confidence > 0.5:
            return 'moderate'
        else:
            return 'weak'
    
    def _estimate_market_share_indication(self, market_positions: List[Dict[str, Any]]) -> str:
        """Estimate market share indication"""
        if not market_positions:
            return 'unknown'
        
        # Market share indicators
        share_indicators = {
            'high': ['leader', 'dominant', 'major', 'significant', 'large'],
            'medium': ['challenger', 'strong', 'competing', 'moderate'],
            'low': ['follower', 'niche', 'specialized', 'small', 'limited']
        }
        
        share_mentions = {}
        for mp in market_positions:
            position_type = mp.get('position_type', '')
            indicators = mp.get('indicators', [])
            
            for share_type, type_keywords in share_indicators.items():
                if any(keyword in ' '.join(indicators).lower() for keyword in type_keywords):
                    share_mentions[share_type] = share_mentions.get(share_type, 0) + 1
        
        if not share_mentions:
            return 'unknown'
        
        # Determine most indicated share level
        indicated_share = max(share_mentions, key=share_mentions.get)
        
        return indicated_share
    
    async def _analyze_competitive_positioning(self, positioning_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze competitive positioning from positioning signals"""
        if not positioning_signals:
            return {}
        
        # Competitive positioning indicators
        competitive_keywords = {
            'cost_leadership': ['cost', 'price', 'affordable', 'value', 'budget'],
            'differentiation': ['different', 'unique', 'innovative', 'special', 'proprietary'],
            'quality_leadership': ['quality', 'premium', 'excellent', 'superior', 'reliable'],
            'service_leadership': ['service', 'support', 'customer', 'experience', 'satisfaction'],
            'innovation_leadership': ['innovation', 'technology', 'r&d', 'patent', 'advanced']
        }
        
        competitive_positions = []
        for signal in positioning_signals:
            signal_text = signal['insight_text']
            keywords = signal.get('keywords', [])
            
            for position_type, type_keywords in competitive_keywords.items():
                if any(keyword in signal_text for keyword in type_keywords):
                    competitive_positions.append({
                        'position_type': position_type,
                        'indicators': [kw for kw in type_keywords if kw in signal_text],
                        'confidence': signal.get('confidence', 0.5),
                        'source_insight': signal.get('insight_id'),
                        'evidence': keywords[:3]
                    })
        
        # Analyze competitive position characteristics
        position_analysis = {
            'identified_positions': competitive_positions,
            'position_diversity': len(set([cp['position_type'] for cp in competitive_positions])),
            'primary_position': self._determine_primary_competitive_position(competitive_positions),
            'position_strength': self._assess_competitive_position_strength(competitive_positions),
            'sustainability': self._assess_position_sustainability(competitive_positions)
        }
        
        return position_analysis
    
    def _determine_primary_competitive_position(self, competitive_positions: List[Dict[str, Any]]) -> str:
        """Determine primary competitive position"""
        if not competitive_positions:
            return 'unknown'
        
        # Count position types
        position_counts = {}
        for cp in competitive_positions:
            position_type = cp['position_type']
            position_counts[position_type] = position_counts.get(position_type, 0) + 1
        
        if not position_counts:
            return 'unknown'
        
        # Find most common position
        primary_position = max(position_counts, key=position_counts.get)
        
        return primary_position
    
    def _assess_competitive_position_strength(self, competitive_positions: List[Dict[str, Any]]) -> str:
        """Assess strength of competitive position"""
        if not competitive_positions:
            return 'unknown'
        
        # Calculate average confidence
        avg_confidence = np.mean([cp.get('confidence', 0.5) for cp in competitive_positions])
        
        # Count strong position indicators
        strong_indicators = ['leadership', 'dominant', 'superior', 'excellent']
        strong_count = sum(
            1 for cp in competitive_positions 
            for indicator in strong_indicators 
            if indicator in ' '.join(cp.get('indicators', [])).lower()
        )
        
        total_positions = len(competitive_positions)
        strong_ratio = strong_count / total_positions if total_positions > 0 else 0
        
        if avg_confidence > 0.7 and strong_ratio > 0.4:
            return 'strong'
        elif avg_confidence > 0.5:
            return 'moderate'
        else:
            return 'weak'
    
    def _assess_position_sustainability(self, competitive_positions: List[Dict[str, Any]]) -> str:
        """Assess sustainability of competitive position"""
        if not competitive_positions:
            return 'unknown'
        
        # Sustainability indicators
        sustainable_indicators = ['sustainable', 'defensible', 'patent', 'proprietary', 'long_term']
        vulnerable_indicators = ['temporary', 'short_term', 'tactical', 'vulnerable']
        
        sustainable_count = 0
        vulnerable_count = 0
        
        for cp in competitive_positions:
            indicators = cp.get('indicators', [])
            
            sustainable_mentions = sum(1 for indicator in sustainable_indicators if indicator in ' '.join(indicators).lower())
            vulnerable_mentions = sum(1 for indicator in vulnerable_indicators if indicator in ' '.join(indicators).lower())
            
            sustainable_count += sustainable_mentions
            vulnerable_count += vulnerable_mentions
        
        total_positions = len(competitive_positions)
        if total_positions == 0:
            return 'unknown'
        
        sustainable_ratio = sustainable_count / total_positions
        
        if sustainable_ratio > 0.6:
            return 'highly_sustainable'
        elif sustainable_ratio > 0.3:
            return 'moderately_sustainable'
        else:
            return 'vulnerable'
    
    async def _analyze_value_chain_positioning(self, positioning_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze value chain positioning from positioning signals"""
        if not positioning_signals:
            return {}
        
        # Value chain positioning indicators
        value_chain_keywords = {
            'upstream': ['supplier', 'input', 'raw_material', 'upstream', 'source'],
            'core': ['core', 'primary', 'main', 'central', 'essential'],
            'downstream': ['customer', 'user', 'distribution', 'delivery', 'downstream'],
            'horizontal': ['partner', 'collaboration', 'integration', 'ecosystem', 'complementary'],
            'disintermediation': ['direct', 'bypass', 'eliminate', 'disintermediate', 'platform']
        }
        
        value_chain_positions = []
        for signal in positioning_signals:
            signal_text = signal['insight_text']
            keywords = signal.get('keywords', [])
            
            for position_type, type_keywords in value_chain_keywords.items():
                if any(keyword in signal_text for keyword in type_keywords):
                    value_chain_positions.append({
                        'position_type': position_type,
                        'indicators': [kw for kw in type_keywords if kw in signal_text],
                        'confidence': signal.get('confidence', 0.5),
                        'source_insight': signal.get('insight_id'),
                        'evidence': keywords[:3]
                    })
        
        # Analyze value chain position characteristics
        position_analysis = {
            'identified_positions': value_chain_positions,
            'position_diversity': len(set([vcp['position_type'] for vcp in value_chain_positions])),
            'primary_position': self._determine_primary_value_chain_position(value_chain_positions),
            'chain_integration': self._assess_chain_integration(value_chain_positions),
            'value_capture': self._assess_value_capture(value_chain_positions)
        }
        
        return position_analysis
    
    def _determine_primary_value_chain_position(self, value_chain_positions: List[Dict[str, Any]]) -> str:
        """Determine primary value chain position"""
        if not value_chain_positions:
            return 'unknown'
        
        # Count position types
        position_counts = {}
        for vcp in value_chain_positions:
            position_type = vcp['position_type']
            position_counts[position_type] = position_counts.get(position_type, 0) + 1
        
        if not position_counts:
            return 'unknown'
        
        # Find most common position
        primary_position = max(position_counts, key=position_counts.get)
        
        return primary_position
    
    def _assess_chain_integration(self, value_chain_positions: List[Dict[str, Any]]) -> str:
        """Assess value chain integration level"""
        if not value_chain_positions:
            return 'unknown'
        
        # Integration indicators
        integrated_indicators = ['integration', 'ecosystem', 'partner', 'collaboration', 'horizontal']
        isolated_indicators = ['isolated', 'separate', 'standalone', 'independent']
        
        integrated_count = 0
        isolated_count = 0
        
        for vcp in value_chain_positions:
            indicators = vcp.get('indicators', [])
            
            integrated_mentions = sum(1 for indicator in integrated_indicators if indicator in ' '.join(indicators).lower())
            isolated_mentions = sum(1 for indicator in isolated_indicators if indicator in ' '.join(indicators).lower())
            
            integrated_count += integrated_mentions
            isolated_count += isolated_mentions
        
        total_positions = len(value_chain_positions)
        if total_positions == 0:
            return 'unknown'
        
        integrated_ratio = integrated_count / total_positions
        
        if integrated_ratio > 0.6:
            return 'highly_integrated'
        elif integrated_ratio > 0.3:
            return 'moderately_integrated'
        else:
            return 'isolated'
    
    def _assess_value_capture(self, value_chain_positions: List[Dict[str, Any]]) -> str:
        """Assess value capture potential"""
        if not value_chain_positions:
            return 'unknown'
        
        # Value capture indicators
        capture_indicators = ['capture', 'control', 'dominant', 'leverage', 'advantage']
        weak_capture_indicators = ['limited', 'dependent', 'commodity', 'competitive']
        
        capture_count = 0
        weak_count = 0
        
        for vcp in value_chain_positions:
            indicators = vcp.get('indicators', [])
            
            capture_mentions = sum(1 for indicator in capture_indicators if indicator in ' '.join(indicators).lower())
            weak_mentions = sum(1 for indicator in weak_capture_indicators if indicator in ' '.join(indicators).lower())
            
            capture_count += capture_mentions
            weak_count += weak_mentions
        
        total_positions = len(value_chain_positions)
        if total_positions == 0:
            return 'unknown'
        
        capture_ratio = capture_count / total_positions
        
        if capture_ratio > 0.6:
            return 'high_value_capture'
        elif capture_ratio > 0.3:
            return 'moderate_value_capture'
        else:
            return 'low_value_capture'
    
    async def _identify_strategic_opportunities(self, positioning_signals: List[Dict[str, Any]],
                                             market_positioning: Dict[str, Any],
                                             competitive_positioning: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify strategic opportunities from positioning analysis"""
        opportunities = []
        
        # Positioning improvement opportunities
        improvement_opportunities = await self._identify_positioning_improvement_opportunities(
            market_positioning, competitive_positioning
        )
        opportunities.extend(improvement_opportunities)
        
        # Market expansion opportunities
        expansion_opportunities = await self._identify_market_expansion_opportunities(
            market_positioning, positioning_signals
        )
        opportunities.extend(expansion_opportunities)
        
        # Value chain optimization opportunities
        value_chain_opportunities = await self._identify_value_chain_optimization_opportunities(
            competitive_positioning, positioning_signals
        )
        opportunities.extend(value_chain_opportunities)
        
        return opportunities
    
    async def _identify_positioning_improvement_opportunities(self, market_positioning: Dict[str, Any],
                                                        competitive_positioning: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify positioning improvement opportunities"""
        opportunities = []
        
        # Check for weak positions
        market_strength = market_positioning.get('position_strength', 'weak')
        competitive_strength = competitive_positioning.get('position_strength', 'weak')
        
        if market_strength == 'weak' or competitive_strength == 'weak':
            opportunities.append({
                'opportunity_type': 'positioning_improvement',
                'opportunity_description': 'Strengthen market and competitive positioning',
                'strategic_value': 'market_leadership',
                'confidence': 0.7,
                'improvement_areas': ['brand_building', 'product_enhancement', 'customer_experience']
            })
        
        # Check for niche opportunities
        market_position = market_positioning.get('primary_position', '')
        if market_position in ['follower', 'niche_player']:
            opportunities.append({
                'opportunity_type': 'positioning_improvement',
                'opportunity_description': 'Leverage niche positioning for growth',
                'strategic_value': 'market_expansion',
                'confidence': 0.6,
                'improvement_areas': ['niche_expansion', 'specialization_deepening']
            })
        
        return opportunities
    
    async def _identify_market_expansion_opportunities(self, market_positioning: Dict[str, Any],
                                                   positioning_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify market expansion opportunities"""
        opportunities = []
        
        # Check for expansion indicators
        expansion_indicators = ['expand', 'grow', 'new_market', 'segment', 'adjacent']
        
        expansion_count = 0
        for signal in positioning_signals:
            signal_text = signal['insight_text']
            expansion_count += sum(1 for indicator in expansion_indicators if indicator in signal_text)
        
        if expansion_count > 0:
            opportunities.append({
                'opportunity_type': 'market_expansion',
                'opportunity_description': 'Expand into new markets or segments',
                'strategic_value': 'growth',
                'confidence': 0.6,
                'expansion_indicators': expansion_count
            })
        
        return opportunities
    
    async def _identify_value_chain_optimization_opportunities(self, competitive_positioning: Dict[str, Any],
                                                           positioning_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify value chain optimization opportunities"""
        opportunities = []
        
        # Check for integration opportunities
        integration_level = competitive_positioning.get('position_strength', 'weak')
        
        if integration_level == 'weak':
            opportunities.append({
                'opportunity_type': 'value_chain_optimization',
                'opportunity_description': 'Improve value chain integration and partnerships',
                'strategic_value': 'efficiency',
                'confidence': 0.6,
                'optimization_areas': ['partnership_development', 'integration_building', 'ecosystem_participation']
            })
        
        # Check for disintermediation opportunities
        disintermediation_indicators = ['direct', 'bypass', 'platform', 'disintermediate']
        
        disintermediation_count = 0
        for signal in positioning_signals:
            signal_text = signal['insight_text']
            disintermediation_count += sum(1 for indicator in disintermediation_indicators if indicator in signal_text)
        
        if disintermediation_count > 0:
            opportunities.append({
                'opportunity_type': 'value_chain_optimization',
                'opportunity_description': 'Pursue disintermediation opportunities',
                'strategic_value': 'cost_reduction',
                'confidence': 0.7,
                'optimization_areas': ['platform_development', 'direct_customer_access', 'cost_elimination']
            })
        
        return opportunities
    
    def _assess_positioning_maturity(self, positioning_signals: List[Dict[str, Any]]) -> str:
        """Assess positioning maturity"""
        if not positioning_signals:
            return 'unknown'
        
        # Maturity indicators
        mature_indicators = ['established', 'mature', 'clear', 'defined', 'stable']
        emerging_indicators = ['emerging', 'developing', 'unclear', 'evolving', 'testing']
        
        mature_count = 0
        emerging_count = 0
        
        for signal in positioning_signals:
            signal_text = signal['insight_text']
            
            mature_mentions = sum(1 for keyword in mature_indicators if keyword in signal_text)
            emerging_mentions = sum(1 for keyword in emerging_indicators if keyword in signal_text)
            
            mature_count += mature_mentions
            emerging_count += emerging_mentions
        
        total_signals = len(positioning_signals)
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
        """Calculate confidence in strategic positioning analysis"""
        if not insights:
            return 0.5
        
        # Average confidence from insights
        insight_confidences = [getattr(insight, 'confidence_score', 0.5) for insight in insights]
        avg_confidence = np.mean(insight_confidences)
        
        # Strategic positioning analysis has high inherent uncertainty
        return avg_confidence * 0.8
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for strategic positioning analysis"""
        return {
            'status': 'working',
            'analysis_methods': [
                'positioning_signal_extraction',
                'market_positioning_analysis',
                'competitive_positioning_analysis',
                'value_chain_positioning_analysis',
                'strategic_opportunity_identification'
            ],
            'sklearn_available': ANALYSIS_AVAILABLE,
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
