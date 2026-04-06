"""
Strategic Insights Implementation
Generates strategic insights from analysis results
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from .....utils.nlp_utils import nlp_utils
from .....config.settings import settings


class StrategicInsights:
    """Strategic insights implementation following detailed specification"""
    
    async def generate_strategic_insights(self, problems: List, 
                                        graph_results: Dict[str, Any],
                                        pattern_results: Dict[str, Any],
                                        root_cause_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategic insights from analysis results"""
        if not problems:
            return []
        
        try:
            insights = []
            
            # 1. Competitive landscape insights
            competitive_insights = await self._analyze_competitive_landscape(problems, pattern_results)
            insights.extend(competitive_insights)
            
            # 2. Market positioning insights
            market_insights = await self._analyze_market_positioning(problems, root_cause_results)
            insights.extend(market_insights)
            
            # 3. Strategic priority insights
            priority_insights = await self._analyze_strategic_priorities(problems, graph_results)
            insights.extend(priority_insights)
            
            # 4. Risk assessment insights
            risk_insights = await self._analyze_risk_factors(problems, root_cause_results)
            insights.extend(risk_insights)
            
            # 5. Opportunity zone insights
            opportunity_insights = await self._analyze_opportunity_zones(problems, pattern_results)
            insights.extend(opportunity_insights)
            
            return insights
            
        except Exception as e:
            print(f"Strategic insights generation failed: {e}")
            return []
    
    async def _analyze_competitive_landscape(self, problems: List, pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze competitive landscape from problems"""
        insights = []
        
        # Extract competitor-related problems
        competitor_problems = [
            p for p in problems 
            if 'competitor' in p.problem_statement.lower() or 
               'competition' in p.problem_statement.lower() or
               'market' in p.problem_statement.lower()
        ]
        
        if not competitor_problems:
            return insights
        
        # Analyze competitor strengths and weaknesses
        competitor_analysis = await self._analyze_competitor_strengths(competitor_problems)
        
        # Generate competitive insights
        insights.append({
            'insight_type': 'competitive_landscape',
            'insight_category': 'market_analysis',
            'title': 'Competitive Position Analysis',
            'description': 'Analysis of competitive landscape based on problem patterns',
            'findings': competitor_analysis,
            'strategic_implications': self._generate_competitive_implications(competitor_analysis),
            'confidence': self._calculate_insight_confidence(competitor_problems),
            'evidence': [p.id for p in competitor_problems],
            'generated_at': datetime.now().isoformat()
        })
        
        return insights
    
    async def _analyze_competitor_strengths(self, competitor_problems: List) -> Dict[str, Any]:
        """Analyze competitor strengths and weaknesses"""
        # Extract keywords from competitor problems
        all_keywords = []
        for problem in competitor_problems:
            keywords = nlp_utils.extract_keywords(problem.problem_statement, max_keywords=10)
            all_keywords.extend(keywords)
        
        # Categorize keywords
        strength_keywords = [kw for kw in all_keywords if any(s in kw.lower() for s in ['strength', 'advantage', 'feature', 'capability'])]
        weakness_keywords = [kw for kw in all_keywords if any(s in kw.lower() for s in ['weakness', 'limitation', 'issue', 'problem'])]
        
        return {
            'identified_strengths': strength_keywords,
            'identified_weaknesses': weakness_keywords,
            'competitor_count': len(competitor_problems),
            'analysis_confidence': min(len(competitor_problems) / 10, 1.0)
        }
    
    def _generate_competitive_implications(self, competitor_analysis: Dict[str, Any]) -> List[str]:
        """Generate strategic implications from competitor analysis"""
        implications = []
        
        strengths = competitor_analysis.get('identified_strengths', [])
        weaknesses = competitor_analysis.get('identified_weaknesses', [])
        
        if strengths:
            implications.append(f"Competitors show strengths in: {', '.join(strengths[:5])}")
        
        if weaknesses:
            implications.append(f"Competitor weaknesses identified: {', '.join(weaknesses[:5])}")
        
        if competitor_analysis.get('competitor_count', 0) > 3:
            implications.append("High competitive density indicates market saturation")
        
        return implications
    
    async def _analyze_market_positioning(self, problems: List, root_cause_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze market positioning insights"""
        insights = []
        
        # Extract market-related problems
        market_problems = [
            p for p in problems 
            if any(term in p.problem_statement.lower() for term in ['market', 'customer', 'user', 'demand'])
        ]
        
        if not market_problems:
            return insights
        
        # Analyze market gaps and needs
        market_analysis = await self._analyze_market_gaps(market_problems)
        
        insights.append({
            'insight_type': 'market_positioning',
            'insight_category': 'market_analysis',
            'title': 'Market Position & Gap Analysis',
            'description': 'Analysis of market positioning based on user problems and needs',
            'findings': market_analysis,
            'strategic_implications': self._generate_market_implications(market_analysis),
            'confidence': self._calculate_insight_confidence(market_problems),
            'evidence': [p.id for p in market_problems],
            'generated_at': datetime.now().isoformat()
        })
        
        return insights
    
    async def _analyze_market_gaps(self, market_problems: List) -> Dict[str, Any]:
        """Analyze market gaps and unmet needs"""
        # Extract unmet needs
        unmet_needs = []
        for problem in market_problems:
            needs = nlp_utils.extract_keywords(problem.problem_statement, max_keywords=5)
            unmet_needs.extend(needs)
        
        # Identify common themes
        need_frequency = {}
        for need in unmet_needs:
            need_frequency[need] = need_frequency.get(need, 0) + 1
        
        # Top unmet needs
        top_needs = sorted(need_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'unmet_needs': top_needs,
            'market_size_indicator': len(market_problems),
            'need_diversity': len(set(unmet_needs)),
            'urgency_indicators': self._identify_urgency_indicators(market_problems)
        }
    
    def _generate_market_implications(self, market_analysis: Dict[str, Any]) -> List[str]:
        """Generate market implications"""
        implications = []
        
        unmet_needs = market_analysis.get('unmet_needs', [])
        if unmet_needs:
            top_needs = [need for need, _ in unmet_needs[:5]]
            implications.append(f"Primary unmet needs: {', '.join(top_needs)}")
        
        market_size = market_analysis.get('market_size_indicator', 0)
        if market_size > 10:
            implications.append("Large market opportunity indicated by high problem volume")
        
        return implications
    
    async def _analyze_strategic_priorities(self, problems: List, graph_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze strategic priorities"""
        insights = []
        
        # Analyze problem severity distribution
        severity_analysis = self._analyze_severity_distribution(problems)
        
        # Analyze problem connectivity
        connectivity_analysis = self._analyze_problem_connectivity(graph_results)
        
        insights.append({
            'insight_type': 'strategic_priorities',
            'insight_category': 'priority_analysis',
            'title': 'Strategic Priority Assessment',
            'description': 'Analysis of strategic priorities based on problem severity and connectivity',
            'findings': {
                'severity_distribution': severity_analysis,
                'connectivity_analysis': connectivity_analysis
            },
            'strategic_implications': self._generate_priority_implications(severity_analysis, connectivity_analysis),
            'confidence': self._calculate_insight_confidence(problems),
            'evidence': [p.id for p in problems],
            'generated_at': datetime.now().isoformat()
        })
        
        return insights
    
    def _analyze_severity_distribution(self, problems: List) -> Dict[str, Any]:
        """Analyze distribution of problem severities"""
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for problem in problems:
            severity = getattr(problem, 'severity', 'medium')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        total = sum(severity_counts.values())
        
        return {
            'severity_counts': severity_counts,
            'severity_percentages': {k: v/total for k, v in severity_counts.items()},
            'dominant_severity': max(severity_counts, key=severity_counts.get),
            'criticality_ratio': severity_counts['critical'] / total if total > 0 else 0
        }
    
    def _analyze_problem_connectivity(self, graph_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze problem connectivity from graph results"""
        if not graph_results.get('key_nodes'):
            return {'connectivity_score': 0.0, 'hub_problems': []}
        
        key_nodes = graph_results['key_nodes']
        
        # Calculate connectivity metrics
        hub_problems = [
            node.get('node_data', {}).get('id') 
            for node in key_nodes 
            if node.get('key_score', 0) > settings.KEY_NODE_THRESHOLD
        ]
        
        avg_connectivity = np.mean([
            node.get('centrality_measures', {}).get('degree', 0) 
            for node in key_nodes
        ]) if key_nodes else 0.0
        
        return {
            'connectivity_score': min(avg_connectivity / 10, 1.0),  # Normalized
            'hub_problems': hub_problems,
            'network_density': graph_results.get('graph_metrics', {}).get('density', 0.0)
        }
    
    def _generate_priority_implications(self, severity_analysis: Dict[str, Any], connectivity_analysis: Dict[str, Any]) -> List[str]:
        """Generate priority implications"""
        implications = []
        
        dominant_severity = severity_analysis.get('dominant_severity', 'medium')
        if dominant_severity in ['high', 'critical']:
            implications.append(f"High severity problems ({dominant_severity}) require immediate attention")
        
        criticality_ratio = severity_analysis.get('criticality_ratio', 0)
        if criticality_ratio > 0.2:
            implications.append("High criticality ratio indicates systemic issues")
        
        connectivity_score = connectivity_analysis.get('connectivity_score', 0)
        if connectivity_score > 0.7:
            implications.append("High connectivity suggests cascading failure risks")
        
        return implications
    
    async def _analyze_risk_factors(self, problems: List, root_cause_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze risk factors"""
        insights = []
        
        # Analyze underlying factors for risks
        underlying_factors = root_cause_results.get('underlying_factors', [])
        risk_factors = [f for f in underlying_factors if f.get('factor_type') == 'technical']
        
        if not risk_factors:
            return insights
        
        # Categorize risks
        risk_analysis = self._categorize_risks(risk_factors)
        
        insights.append({
            'insight_type': 'risk_assessment',
            'insight_category': 'risk_analysis',
            'title': 'Risk Factor Analysis',
            'description': 'Analysis of risk factors from underlying causes',
            'findings': risk_analysis,
            'strategic_implications': self._generate_risk_implications(risk_analysis),
            'confidence': self._calculate_insight_confidence(risk_factors),
            'evidence': [f.get('contributing_problems', []) for f in risk_factors],
            'generated_at': datetime.now().isoformat()
        })
        
        return insights
    
    def _categorize_risks(self, risk_factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Categorize identified risks"""
        risk_categories = {'technical': 0, 'operational': 0, 'strategic': 0, 'external': 0}
        
        for factor in risk_factors:
            category = factor.get('factor_category', 'other')
            if category in risk_categories:
                risk_categories[category] += 1
            else:
                risk_categories['external'] += 1
        
        return {
            'risk_counts': risk_categories,
            'dominant_risk_type': max(risk_categories, key=risk_categories.get),
            'high_impact_risks': [f for f in risk_factors if f.get('importance_score', 0) > 0.8]
        }
    
    def _generate_risk_implications(self, risk_analysis: Dict[str, Any]) -> List[str]:
        """Generate risk implications"""
        implications = []
        
        dominant_risk = risk_analysis.get('dominant_risk_type', 'technical')
        implications.append(f"Dominant risk category: {dominant_risk}")
        
        high_impact_risks = risk_analysis.get('high_impact_risks', [])
        if high_impact_risks:
            implications.append(f"High-impact risks identified: {len(high_impact_risks)} factors")
        
        return implications
    
    async def _analyze_opportunity_zones(self, problems: List, pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze opportunity zones"""
        insights = []
        
        # Extract positive patterns and unmet needs
        patterns = pattern_results.get('patterns', [])
        opportunity_patterns = [p for p in patterns if 'opportunity' in str(p).lower()]
        
        if not opportunity_patterns:
            # Look for unmet needs as opportunities
            unmet_patterns = [p for p in patterns if 'unmet' in str(p).lower() or 'need' in str(p).lower()]
            if unmet_patterns:
                insights.append({
                    'insight_type': 'opportunity_analysis',
                    'insight_category': 'opportunity_identification',
                    'title': 'Opportunity Zone Analysis',
                    'description': 'Analysis of opportunity zones from unmet needs and patterns',
                    'findings': {'opportunity_patterns': unmet_patterns},
                    'strategic_implications': ["Unmet needs represent potential opportunities"],
                    'confidence': 0.6,
                    'evidence': [],
                    'generated_at': datetime.now().isoformat()
                })
        
        return insights
    
    def _calculate_insight_confidence(self, evidence_problems: List) -> float:
        """Calculate confidence in insight based on evidence"""
        if not evidence_problems:
            return 0.5
        
        # Base confidence from evidence count
        evidence_confidence = min(len(evidence_problems) / 10, 1.0)
        
        # Boost for diverse evidence
        if len(evidence_problems) >= 5:
            evidence_confidence += 0.2
        
        return min(evidence_confidence, 1.0)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for strategic insights"""
        return {
            'status': 'working',
            'insight_types': [
                'competitive_landscape',
                'market_positioning',
                'strategic_priorities',
                'risk_assessment',
                'opportunity_analysis'
            ],
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
