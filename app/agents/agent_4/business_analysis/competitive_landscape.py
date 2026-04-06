"""
Competitive Landscape Analysis Implementation
Analyzes competitive landscape from insights
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


class CompetitiveLandscape:
    """Competitive landscape analysis implementation following detailed specification"""
    
    async def analyze_competitive_landscape(self, insights: List) -> Dict[str, Any]:
        """Analyze competitive landscape from insights"""
        if not insights:
            return {}
        
        try:
            # Extract competitor information
            competitor_info = await self._extract_competitor_information(insights)
            
            # Analyze competitor strengths and weaknesses
            competitor_analysis = await self._analyze_competitor_strengths_weaknesses(competitor_info)
            
            # Identify market positioning
            market_positioning = await self._identify_market_positioning(competitor_info)
            
            # Assess competitive intensity
            competitive_intensity = await self._assess_competitive_intensity(competitor_info)
            
            # Identify strategic opportunities
            strategic_opportunities = await self._identify_strategic_opportunities(
                competitor_info, competitor_analysis
            )
            
            return {
                'competitor_info': competitor_info,
                'competitor_analysis': competitor_analysis,
                'market_positioning': market_positioning,
                'competitive_intensity': competitive_intensity,
                'strategic_opportunities': strategic_opportunities,
                'analysis_confidence': self._calculate_analysis_confidence(insights)
            }
            
        except Exception as e:
            print(f"Competitive landscape analysis failed: {e}")
            return {'error': str(e)}
    
    async def _extract_competitor_information(self, insights: List) -> List[Dict[str, Any]]:
        """Extract competitor information from insights"""
        competitor_info = []
        
        for insight in insights:
            insight_text = insight.insight_statement.lower()
            
            # Competitor indicators
            competitor_keywords = ['competitor', 'competition', 'rival', 'alternative', 'option']
            
            # Company/product indicators
            company_keywords = ['company', 'product', 'service', 'solution', 'platform']
            
            # Check for competitor mentions
            competitor_mentions = sum(1 for keyword in competitor_keywords if keyword in insight_text)
            company_mentions = sum(1 for keyword in company_keywords if keyword in insight_text)
            
            if competitor_mentions > 0 or company_mentions > 0:
                # Extract competitor names
                entities = nlp_utils.extract_entities(insight.insight_statement)
                competitor_names = [entity for entity in entities if entity.get('type') == 'ORG']
                
                # Extract competitive characteristics
                characteristics = await self._extract_competitor_characteristics(insight_text)
                
                for name in competitor_names:
                    info = {
                        'competitor_name': name,
                        'insight_id': insight.id,
                        'insight_text': insight.insight_statement,
                        'characteristics': characteristics,
                        'confidence': getattr(insight, 'confidence_score', 0.5),
                        'category': getattr(insight, 'insight_category', 'general')
                    }
                    competitor_info.append(info)
        
        # Deduplicate competitors
        unique_competitors = {}
        for info in competitor_info:
            name = info['competitor_name']
            if name not in unique_competitors:
                unique_competitors[name] = info
            else:
                # Merge information for same competitor
                existing = unique_competitors[name]
                existing['characteristics'].update(info['characteristics'])
                existing['confidence'] = (existing['confidence'] + info['confidence']) / 2
        
        return list(unique_competitors.values())
    
    async def _extract_competitor_characteristics(self, insight_text: str) -> Dict[str, Any]:
        """Extract competitor characteristics from insight text"""
        characteristics = {}
        
        # Strength indicators
        strength_keywords = ['strong', 'advantage', 'leader', 'dominant', 'established']
        strengths = [keyword for keyword in strength_keywords if keyword in insight_text]
        
        # Weakness indicators
        weakness_keywords = ['weak', 'limitation', 'problem', 'issue', 'challenge']
        weaknesses = [keyword for keyword in weakness_keywords if keyword in insight_text]
        
        # Feature indicators
        feature_keywords = ['feature', 'functionality', 'capability', 'service', 'offering']
        features = nlp_utils.extract_keywords(insight_text, max_keywords=5)
        
        # Market position indicators
        position_keywords = ['market', 'share', 'position', 'ranking', 'leader']
        positions = [keyword for keyword in position_keywords if keyword in insight_text]
        
        # Pricing indicators
        pricing_keywords = ['price', 'cost', 'expensive', 'cheap', 'value', 'premium']
        pricing = [keyword for keyword in pricing_keywords if keyword in insight_text]
        
        characteristics.update({
            'strengths': strengths,
            'weaknesses': weaknesses,
            'features': features,
            'market_positions': positions,
            'pricing_indicators': pricing
        })
        
        return characteristics
    
    async def _analyze_competitor_strengths_weaknesses(self, competitor_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze competitor strengths and weaknesses"""
        if not competitor_info:
            return {}
        
        analysis = {}
        
        for competitor in competitor_info:
            name = competitor['competitor_name']
            characteristics = competitor.get('characteristics', {})
            
            # Analyze strengths
            strengths = characteristics.get('strengths', [])
            strength_analysis = {
                'count': len(strengths),
                'types': self._categorize_strengths(strengths),
                'confidence': competitor.get('confidence', 0.5)
            }
            
            # Analyze weaknesses
            weaknesses = characteristics.get('weaknesses', [])
            weakness_analysis = {
                'count': len(weaknesses),
                'types': self._categorize_weaknesses(weaknesses),
                'confidence': competitor.get('confidence', 0.5)
            }
            
            # Analyze features
            features = characteristics.get('features', [])
            feature_analysis = {
                'count': len(features),
                'categories': self._categorize_features(features),
                'uniqueness': self._assess_feature_uniqueness(features)
            }
            
            # Analyze market position
            positions = characteristics.get('market_positions', [])
            position_analysis = {
                'indicators': positions,
                'position': self._determine_market_position(positions),
                'confidence': competitor.get('confidence', 0.5)
            }
            
            analysis[name] = {
                'strengths': strength_analysis,
                'weaknesses': weakness_analysis,
                'features': feature_analysis,
                'market_position': position_analysis,
                'overall_assessment': self._assess_overall_competitor(competitor)
            }
        
        return analysis
    
    def _categorize_strengths(self, strengths: List[str]) -> List[str]:
        """Categorize competitor strengths"""
        categories = {
            'market': ['market', 'share', 'leader', 'dominant'],
            'product': ['product', 'feature', 'quality', 'innovation'],
            'brand': ['brand', 'reputation', 'recognition', 'trust'],
            'operational': ['efficiency', 'scale', 'resources', 'distribution'],
            'financial': ['revenue', 'profit', 'funding', 'investment']
        }
        
        categorized = []
        for strength in strengths:
            for category, keywords in categories.items():
                if any(keyword in strength for keyword in keywords):
                    categorized.append(category)
                    break
        
        return list(set(categorized))
    
    def _categorize_weaknesses(self, weaknesses: List[str]) -> List[str]:
        """Categorize competitor weaknesses"""
        categories = {
            'product': ['product', 'feature', 'quality', 'reliability'],
            'market': ['market', 'share', 'position', 'reach'],
            'operational': ['efficiency', 'cost', 'scale', 'distribution'],
            'customer': ['customer', 'support', 'service', 'satisfaction'],
            'technology': ['technology', 'innovation', 'infrastructure', 'platform']
        }
        
        categorized = []
        for weakness in weaknesses:
            for category, keywords in categories.items():
                if any(keyword in weakness for keyword in keywords):
                    categorized.append(category)
                    break
        
        return list(set(categorized))
    
    def _categorize_features(self, features: List[str]) -> List[str]:
        """Categorize competitor features"""
        categories = {
            'core': ['core', 'essential', 'basic', 'fundamental'],
            'advanced': ['advanced', 'premium', 'enterprise', 'professional'],
            'technical': ['technical', 'infrastructure', 'platform', 'api'],
            'user': ['user', 'interface', 'experience', 'design'],
            'integration': ['integration', 'connectivity', 'compatibility', 'ecosystem']
        }
        
        categorized = []
        for feature in features:
            for category, keywords in categories.items():
                if any(keyword in feature for keyword in keywords):
                    categorized.append(category)
                    break
        
        return list(set(categorized))
    
    def _assess_feature_uniqueness(self, features: List[str]) -> str:
        """Assess uniqueness of competitor features"""
        if not features:
            return 'unknown'
        
        # Uniqueness indicators
        unique_keywords = ['unique', 'innovative', 'proprietary', 'exclusive', 'differentiated']
        
        unique_mentions = sum(1 for keyword in unique_keywords for feature in features if keyword in feature)
        
        if unique_mentions > 2:
            return 'high'
        elif unique_mentions > 1:
            return 'medium'
        else:
            return 'low'
    
    def _determine_market_position(self, positions: List[str]) -> str:
        """Determine market position from indicators"""
        if not positions:
            return 'unknown'
        
        position_text = ' '.join(positions).lower()
        
        if any(keyword in position_text for keyword in ['leader', 'dominant', 'first']):
            return 'market_leader'
        elif any(keyword in position_text for keyword in ['strong', 'major', 'significant']):
            return 'strong_competitor'
        elif any(keyword in position_text for keyword in ['growing', 'rising', 'emerging']):
            return 'growing_competitor'
        elif any(keyword in position_text for keyword in ['niche', 'specialized', 'focused']):
            return 'niche_player'
        else:
            return 'general_competitor'
    
    def _assess_overall_competitor(self, competitor: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall competitor profile"""
        characteristics = competitor.get('characteristics', {})
        
        # Calculate strength vs weakness ratio
        strengths_count = len(characteristics.get('strengths', []))
        weaknesses_count = len(characteristics.get('weaknesses', []))
        
        if strengths_count + weaknesses_count > 0:
            strength_ratio = strengths_count / (strengths_count + weaknesses_count)
        else:
            strength_ratio = 0.5
        
        # Assess competitive threat
        if strength_ratio > 0.7:
            threat_level = 'high'
        elif strength_ratio > 0.5:
            threat_level = 'medium'
        else:
            threat_level = 'low'
        
        return {
            'strength_weakness_ratio': strength_ratio,
            'threat_level': threat_level,
            'strategic_importance': self._assess_strategic_importance(competitor),
            'monitoring_priority': self._determine_monitoring_priority(threat_level)
        }
    
    def _assess_strategic_importance(self, competitor: Dict[str, Any]) -> str:
        """Assess strategic importance of competitor"""
        characteristics = competitor.get('characteristics', {})
        
        # Market position importance
        positions = characteristics.get('market_positions', [])
        if any(keyword in ' '.join(positions).lower() for keyword in ['leader', 'dominant']):
            return 'critical'
        
        # Feature sophistication
        features = characteristics.get('features', [])
        if len(features) > 5:
            return 'high'
        elif len(features) > 3:
            return 'medium'
        else:
            return 'low'
    
    def _determine_monitoring_priority(self, threat_level: str) -> str:
        """Determine monitoring priority based on threat level"""
        priority_map = {
            'high': 'continuous',
            'medium': 'weekly',
            'low': 'monthly'
        }
        return priority_map.get(threat_level, 'quarterly')
    
    async def _identify_market_positioning(self, competitor_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify market positioning"""
        if not competitor_info:
            return {}
        
        # Group competitors by position
        position_groups = {'market_leader': [], 'strong_competitor': [], 'growing_competitor': [], 'niche_player': []}
        
        for competitor in competitor_info:
            characteristics = competitor.get('characteristics', {})
            positions = characteristics.get('market_positions', [])
            position = self._determine_market_position(positions)
            
            if position in position_groups:
                position_groups[position].append(competitor)
        
        # Analyze market structure
        total_competitors = len(competitor_info)
        market_structure = self._analyze_market_structure(position_groups, total_competitors)
        
        # Identify positioning opportunities
        positioning_opportunities = self._identify_positioning_opportunities(position_groups)
        
        return {
            'competitor_groups': position_groups,
            'market_structure': market_structure,
            'positioning_opportunities': positioning_opportunities,
            'our_position': self._assess_our_position(position_groups)
        }
    
    def _analyze_market_structure(self, position_groups: Dict[str, List], total_competitors: int) -> Dict[str, Any]:
        """Analyze market structure"""
        if total_competitors == 0:
            return {'structure': 'unknown'}
        
        leader_count = len(position_groups.get('market_leader', []))
        strong_count = len(position_groups.get('strong_competitor', []))
        growing_count = len(position_groups.get('growing_competitor', []))
        niche_count = len(position_groups.get('niche_player', []))
        
        if leader_count > 0:
            structure = 'oligopoly' if total_competitors <= 5 else 'fragmented_with_leader'
        elif strong_count > 2:
            structure = 'competitive'
        elif growing_count > 2:
            structure = 'dynamic'
        elif niche_count > 3:
            structure = 'specialized'
        else:
            structure = 'emerging'
        
        return {
            'structure': structure,
            'leader_count': leader_count,
            'strong_competitors': strong_count,
            'growing_competitors': growing_count,
            'niche_players': niche_count,
            'concentration': self._calculate_market_concentration(position_groups, total_competitors)
        }
    
    def _calculate_market_concentration(self, position_groups: Dict[str, List], total_competitors: int) -> str:
        """Calculate market concentration"""
        if total_competitors == 0:
            return 'unknown'
        
        leader_share = len(position_groups.get('market_leader', [])) / total_competitors
        strong_share = len(position_groups.get('strong_competitor', [])) / total_competitors
        
        top_tier_share = (leader_share + strong_share)
        
        if top_tier_share > 0.6:
            return 'high'
        elif top_tier_share > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _identify_positioning_opportunities(self, position_groups: Dict[str, List]) -> List[str]:
        """Identify positioning opportunities"""
        opportunities = []
        
        leader_count = len(position_groups.get('market_leader', []))
        strong_count = len(position_groups.get('strong_competitor', []))
        growing_count = len(position_groups.get('growing_competitor', []))
        
        # Check for gaps in market leadership
        if leader_count == 0:
            opportunities.append('market_leadership_opportunity')
        
        # Check for underserved segments
        if strong_count < 2 and growing_count < 2:
            opportunities.append('segment_specialization_opportunity')
        
        # Check for differentiation opportunities
        if leader_count > 0 and strong_count > 2:
            opportunities.append('differentiation_opportunity')
        
        return opportunities
    
    def _assess_our_position(self, position_groups: Dict[str, List]) -> str:
        """Assess our market position (placeholder for analysis)"""
        # This would typically use our product data
        # For now, return a default assessment
        total_competitors = sum(len(group) for group in position_groups.values())
        
        if total_competitors > 10:
            return 'challenger'
        elif total_competitors > 5:
            return 'follower'
        else:
            return 'emerging'
    
    async def _assess_competitive_intensity(self, competitor_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess competitive intensity"""
        if not competitor_info:
            return {}
        
        # Calculate intensity metrics
        competitor_count = len(competitor_info)
        high_threat_count = len([
            c for c in competitor_info 
            if self._assess_overall_competitor(c).get('threat_level') == 'high'
        ])
        
        # Assess market dynamics
        dynamics = self._assess_market_dynamics(competitor_info)
        
        # Calculate intensity score
        intensity_score = self._calculate_intensity_score(
            competitor_count, high_threat_count, dynamics
        )
        
        return {
            'competitor_count': competitor_count,
            'high_threat_count': high_threat_count,
            'market_dynamics': dynamics,
            'intensity_score': intensity_score,
            'intensity_level': self._determine_intensity_level(intensity_score),
            'entry_barriers': self._assess_entry_barriers(competitor_info)
        }
    
    def _assess_market_dynamics(self, competitor_info: List[Dict[str, Any]]) -> str:
        """Assess market dynamics"""
        if not competitor_info:
            return 'unknown'
        
        # Count growing competitors
        growing_count = 0
        for competitor in competitor_info:
            characteristics = competitor.get('characteristics', {})
            positions = characteristics.get('market_positions', [])
            if self._determine_market_position(positions) == 'growing_competitor':
                growing_count += 1
        
        total_competitors = len(competitor_info)
        growth_ratio = growing_count / total_competitors if total_competitors > 0 else 0
        
        if growth_ratio > 0.4:
            return 'dynamic'
        elif growth_ratio > 0.2:
            return 'evolving'
        else:
            return 'stable'
    
    def _calculate_intensity_score(self, competitor_count: int, high_threat_count: int, dynamics: str) -> float:
        """Calculate competitive intensity score"""
        # Base score from competitor count
        base_score = min(competitor_count / 10, 1.0)
        
        # Threat level boost
        threat_boost = (high_threat_count / max(competitor_count, 1)) * 0.3
        
        # Dynamics boost
        dynamics_boost = {'dynamic': 0.2, 'evolving': 0.1, 'stable': 0.0}
        dynamics_boost = dynamics_boost.get(dynamics, 0.0)
        
        return min(base_score + threat_boost + dynamics_boost, 1.0)
    
    def _determine_intensity_level(self, intensity_score: float) -> str:
        """Determine intensity level from score"""
        if intensity_score > 0.8:
            return 'high'
        elif intensity_score > 0.6:
            return 'medium-high'
        elif intensity_score > 0.4:
            return 'medium'
        elif intensity_score > 0.2:
            return 'low-medium'
        else:
            return 'low'
    
    def _assess_entry_barriers(self, competitor_info: List[Dict[str, Any]]) -> str:
        """Assess entry barriers"""
        if not competitor_info:
            return 'unknown'
        
        # Count established competitors
        established_count = 0
        for competitor in competitor_info:
            characteristics = competitor.get('characteristics', {})
            positions = characteristics.get('market_positions', [])
            if any(keyword in ' '.join(positions).lower() for keyword in ['leader', 'dominant', 'established']):
                established_count += 1
        
        total_competitors = len(competitor_info)
        establishment_ratio = established_count / total_competitors if total_competitors > 0 else 0
        
        if establishment_ratio > 0.6:
            return 'high'
        elif establishment_ratio > 0.3:
            return 'medium'
        else:
            return 'low'
    
    async def _identify_strategic_opportunities(self, competitor_info: List[Dict[str, Any]], 
                                           competitor_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify strategic opportunities from competitive analysis"""
        opportunities = []
        
        # Weakness exploitation opportunities
        weakness_opportunities = await self._identify_weakness_opportunities(competitor_analysis)
        opportunities.extend(weakness_opportunities)
        
        # Gap opportunities
        gap_opportunities = await self._identify_gap_opportunities(competitor_info)
        opportunities.extend(gap_opportunities)
        
        # Differentiation opportunities
        differentiation_opportunities = await self._identify_differentiation_opportunities(competitor_analysis)
        opportunities.extend(differentiation_opportunities)
        
        return opportunities
    
    async def _identify_weakness_opportunities(self, competitor_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities from competitor weaknesses"""
        opportunities = []
        
        for name, analysis in competitor_analysis.items():
            weaknesses = analysis.get('weaknesses', {})
            weakness_types = weaknesses.get('types', [])
            
            for weakness_type in weakness_types:
                if weakness_type in ['product', 'customer', 'operational']:
                    opportunity = {
                        'opportunity_type': 'weakness_exploitation',
                        'target_competitor': name,
                        'weakness_type': weakness_type,
                        'opportunity_description': f"Exploit {weakness_type} weakness of {name}",
                        'strategic_value': self._assess_weakness_strategic_value(weakness_type),
                        'confidence': weaknesses.get('confidence', 0.5)
                    }
                    opportunities.append(opportunity)
        
        return opportunities
    
    async def _identify_gap_opportunities(self, competitor_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify gap opportunities"""
        opportunities = []
        
        # Analyze feature gaps across competitors
        all_features = set()
        for competitor in competitor_info:
            characteristics = competitor.get('characteristics', {})
            features = characteristics.get('features', [])
            all_features.update(features)
        
        # Look for underserved areas
        common_features = [f for f in all_features if list(all_features).count(f) > 1]
        unique_features = [f for f in all_features if list(all_features).count(f) == 1]
        
        if len(unique_features) > 0:
            opportunities.append({
                'opportunity_type': 'feature_gap',
                'gap_description': 'Unique features not widely adopted',
                'opportunity_features': unique_features[:5],
                'strategic_value': 'differentiation',
                'confidence': 0.7
            })
        
        return opportunities
    
    async def _identify_differentiation_opportunities(self, competitor_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify differentiation opportunities"""
        opportunities = []
        
        # Analyze positioning opportunities
        for name, analysis in competitor_analysis.items():
            market_position = analysis.get('market_position', {})
            position = market_position.get('position', 'general_competitor')
            
            if position in ['general_competitor', 'niche_player']:
                opportunity = {
                    'opportunity_type': 'positioning_differentiation',
                    'target_competitor': name,
                    'current_position': position,
                    'opportunity_description': f"Improve market position relative to {name}",
                    'strategic_value': 'market_positioning',
                    'confidence': market_position.get('confidence', 0.5)
                }
                opportunities.append(opportunity)
        
        return opportunities
    
    def _assess_weakness_strategic_value(self, weakness_type: str) -> str:
        """Assess strategic value of exploiting weakness"""
        value_map = {
            'product': 'high',      # Product weaknesses often exploitable
            'customer': 'very_high', # Customer issues are high value
            'operational': 'medium',  # Operational issues moderate value
            'technology': 'high',     # Technology gaps high value
            'market': 'medium'       # Market positioning moderate value
        }
        
        return value_map.get(weakness_type, 'medium')
    
    def _calculate_analysis_confidence(self, insights: List) -> float:
        """Calculate confidence in competitive analysis"""
        if not insights:
            return 0.5
        
        # Average confidence from insights
        insight_confidences = [getattr(insight, 'confidence_score', 0.5) for insight in insights]
        avg_confidence = np.mean(insight_confidences)
        
        # Competitive analysis has moderate inherent uncertainty
        return avg_confidence * 0.9
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for competitive landscape analysis"""
        return {
            'status': 'working',
            'analysis_methods': [
                'competitor_information_extraction',
                'strength_weakness_analysis',
                'market_positioning_identification',
                'competitive_intensity_assessment',
                'strategic_opportunity_identification'
            ],
            'sklearn_available': ANALYSIS_AVAILABLE,
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
