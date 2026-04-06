"""
Agent 4: Product Brief - Main Orchestrator
Complete implementation following detailed specification from docs/agent-4-implementation.md
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import numpy as np

from ..base_agent import BaseAgent
from ...models.insight import Insight
from ...models.brief import ProductBrief, ProductBriefCreate
from ...config.database import insight_repo, brief_repo
from ...utils.nlp_utils import nlp_utils
from ...utils.cache import cache_manager, CacheKeys
from ...config.settings import settings

from .business_analysis import BusinessAnalysis
from .feature_ideation import FeatureIdeation
from .ux_design import UXDesign
from .impact_assessment import ImpactAssessment

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False


class ProductBriefAgent(BaseAgent):
    """
    Product Brief Agent - Complete Implementation
    
    Following detailed specification:
    1. Business Analysis (market opportunity, competitive landscape)
    2. Feature Ideation (feature concepts, prioritization)
    3. UX Design (user experience, interface design)
    4. Impact Assessment (business impact, success metrics)
    5. Brief Generation (structured product briefs)
    6. Storage (briefs database with evidence)
    """
    
    def __init__(self):
        super().__init__("agent_4")
        self.business_analysis = BusinessAnalysis()
        self.feature_ideation = FeatureIdeation()
        self.ux_design = UXDesign()
        self.impact_assessment = ImpactAssessment()
    
    async def validate_input(self, insight_ids: List[str] = None, **kwargs) -> bool:
        """Validate input parameters"""
        if insight_ids is None:
            return True  # Will use recent insights
        
        if not isinstance(insight_ids, list):
            self.log_error("insight_ids must be a list")
            return False
        
        return True
    
    async def run(self, insight_ids: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Run complete product brief pipeline
        
        Following detailed specification:
        1. Insight Retrieval
        2. Business Analysis
        3. Feature Ideation
        4. UX Design
        5. Impact Assessment
        6. Brief Generation
        7. Storage
        """
        self.log_info("Starting product brief generation")
        
        try:
            # Step 1: Insight Retrieval
            insights = await self._retrieve_insights(insight_ids)
            self.log_info(f"Retrieved {len(insights)} insights")
            
            if len(insights) < settings.MIN_INSIGHT_THRESHOLD:
                return {
                    'status': 'insufficient_data',
                    'message': f'Need at least {settings.MIN_INSIGHT_THRESHOLD} insights for brief generation',
                    'insights_count': len(insights),
                    'agent_id': self.agent_id
                }
            
            # Step 2: Business Analysis
            business_analysis = await self._run_business_analysis(insights)
            self.log_info(f"Business analysis completed: {len(business_analysis.get('opportunities', []))} opportunities")
            
            # Step 3: Feature Ideation
            feature_ideation = await self._run_feature_ideation(insights, business_analysis)
            self.log_info(f"Feature ideation completed: {len(feature_ideation.get('features', []))} features")
            
            # Step 4: UX Design
            ux_design = await self._run_ux_design(insights, business_analysis, feature_ideation)
            self.log_info(f"UX design completed: {len(ux_design.get('design_concepts', []))} concepts")
            
            # Step 5: Impact Assessment
            impact_assessment = await self._run_impact_assessment(
                insights, business_analysis, feature_ideation, ux_design
            )
            self.log_info(f"Impact assessment completed: {len(impact_assessment.get('impact_metrics', []))} metrics")
            
            # Step 6: Brief Generation
            briefs = await self._run_brief_generation(
                insights, business_analysis, feature_ideation, ux_design, impact_assessment
            )
            self.log_info(f"Generated {len(briefs)} product briefs")
            
            # Step 7: Storage
            stored_briefs = await self._store_briefs(briefs)
            self.log_info(f"Stored {len(stored_briefs)} briefs")
            
            return {
                'status': 'success',
                'insights_processed': len(insights),
                'business_analysis': business_analysis,
                'feature_ideation': feature_ideation,
                'ux_design': ux_design,
                'impact_assessment': impact_assessment,
                'briefs_generated': len(briefs),
                'briefs_stored': len(stored_briefs),
                'brief_ids': [b.get('id') for b in stored_briefs],
                'processing_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log_error(f"Pipeline execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'agent_id': self.agent_id,
                'processing_time': datetime.now().isoformat()
            }
    
    async def _retrieve_insights(self, insight_ids: List[str] = None) -> List[Insight]:
        """Retrieve insights from database"""
        if insight_ids:
            # Retrieve specific insights
            insights = []
            for insight_id in insight_ids:
                insight_data = await insight_repo.get_by_id(insight_id)
                if insight_data:
                    insights.append(Insight.from_dict(insight_data))
            return insights
        else:
            # Retrieve recent insights
            insight_data_list = await insight_repo.get_recent_insights(limit=50)
            return [Insight.from_dict(insight_data) for insight_data in insight_data_list]
    
    async def _run_business_analysis(self, insights: List[Insight]) -> Dict[str, Any]:
        """Run business analysis following detailed specification"""
        try:
            # Market opportunity analysis
            market_opportunities = await self.business_analysis.analyze_market_opportunities(insights)
            
            # Competitive landscape analysis
            competitive_landscape = await self.business_analysis.analyze_competitive_landscape(insights)
            
            # Business model analysis
            business_model = await self.business_analysis.analyze_business_model(insights)
            
            # Strategic positioning
            strategic_positioning = await self.business_analysis.analyze_strategic_positioning(insights)
            
            return {
                'market_opportunities': market_opportunities,
                'competitive_landscape': competitive_landscape,
                'business_model': business_model,
                'strategic_positioning': strategic_positioning,
                'analysis_confidence': self._calculate_analysis_confidence(insights)
            }
            
        except Exception as e:
            self.log_error(f"Business analysis failed: {e}")
            return {'error': str(e)}
    
    async def _run_feature_ideation(self, insights: List[Insight], 
                                   business_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Run feature ideation following detailed specification"""
        try:
            # Feature concept generation
            feature_concepts = await self.feature_ideation.generate_feature_concepts(
                insights, business_analysis
            )
            
            # Feature prioritization
            prioritized_features = await self.feature_ideation.prioritize_features(
                feature_concepts, business_analysis
            )
            
            # Feature specification
            feature_specs = await self.feature_ideation.create_feature_specifications(
                prioritized_features
            )
            
            return {
                'feature_concepts': feature_concepts,
                'prioritized_features': prioritized_features,
                'feature_specifications': feature_specs,
                'ideation_confidence': self._calculate_ideation_confidence(insights, feature_concepts)
            }
            
        except Exception as e:
            self.log_error(f"Feature ideation failed: {e}")
            return {'error': str(e)}
    
    async def _run_ux_design(self, insights: List[Insight], 
                              business_analysis: Dict[str, Any],
                              feature_ideation: Dict[str, Any]) -> Dict[str, Any]:
        """Run UX design following detailed specification"""
        try:
            # User journey mapping
            user_journeys = await self.ux_design.map_user_journeys(
                insights, business_analysis, feature_ideation
            )
            
            # Interface design concepts
            interface_concepts = await self.ux_design.design_interface_concepts(
                insights, feature_ideation
            )
            
            # Interaction design
            interaction_design = await self.ux_design.design_interactions(
                insights, user_journeys, interface_concepts
            )
            
            # Design principles
            design_principles = await self.ux_design.establish_design_principles(
                insights, business_analysis
            )
            
            return {
                'user_journeys': user_journeys,
                'interface_concepts': interface_concepts,
                'interaction_design': interaction_design,
                'design_principles': design_principles,
                'design_confidence': self._calculate_design_confidence(insights)
            }
            
        except Exception as e:
            self.log_error(f"UX design failed: {e}")
            return {'error': str(e)}
    
    async def _run_impact_assessment(self, insights: List[Insight],
                                    business_analysis: Dict[str, Any],
                                    feature_ideation: Dict[str, Any],
                                    ux_design: Dict[str, Any]) -> Dict[str, Any]:
        """Run impact assessment following detailed specification"""
        try:
            # Business impact analysis
            business_impact = await self.impact_assessment.assess_business_impact(
                insights, business_analysis, feature_ideation
            )
            
            # User impact analysis
            user_impact = await self.impact_assessment.assess_user_impact(
                insights, ux_design, feature_ideation
            )
            
            # Technical impact analysis
            technical_impact = await self.impact_assessment.assess_technical_impact(
                insights, feature_ideation, ux_design
            )
            
            # Success metrics definition
            success_metrics = await self.impact_assessment.define_success_metrics(
                insights, business_analysis, feature_ideation, ux_design
            )
            
            # Risk assessment
            risk_assessment = await self.impact_assessment.assess_risks(
                insights, feature_ideation, business_analysis
            )
            
            return {
                'business_impact': business_impact,
                'user_impact': user_impact,
                'technical_impact': technical_impact,
                'success_metrics': success_metrics,
                'risk_assessment': risk_assessment,
                'assessment_confidence': self._calculate_assessment_confidence(insights)
            }
            
        except Exception as e:
            self.log_error(f"Impact assessment failed: {e}")
            return {'error': str(e)}
    
    async def _run_brief_generation(self, insights: List[Insight],
                                  business_analysis: Dict[str, Any],
                                  feature_ideation: Dict[str, Any],
                                  ux_design: Dict[str, Any],
                                  impact_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run brief generation following detailed specification"""
        briefs = []
        
        try:
            # Generate briefs for top-priority features
            prioritized_features = feature_ideation.get('prioritized_features', [])
            
            # Generate briefs for top 5 features
            for i, feature in enumerate(prioritized_features[:5]):
                brief = await self._generate_feature_brief(
                    feature, insights, business_analysis, ux_design, impact_assessment
                )
                
                if brief:
                    brief['brief_priority'] = i + 1
                    briefs.append(brief)
            
            # Generate overall product brief
            if len(briefs) > 0:
                overall_brief = await self._generate_overall_brief(
                    briefs, insights, business_analysis, feature_ideation, ux_design, impact_assessment
                )
                if overall_brief:
                    overall_brief['brief_priority'] = 0  # Highest priority
                    briefs.append(overall_brief)
            
            return briefs
            
        except Exception as e:
            self.log_error(f"Brief generation failed: {e}")
            return []
    
    async def _generate_feature_brief(self, feature: Dict[str, Any],
                                   insights: List[Insight],
                                   business_analysis: Dict[str, Any],
                                   ux_design: Dict[str, Any],
                                   impact_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate brief for a specific feature"""
        feature_name = feature.get('name', 'Unknown Feature')
        
        # Extract relevant insights
        relevant_insights = [
            insight for insight in insights 
            if any(keyword in insight.insight_statement.lower() 
                   for keyword in feature.get('keywords', []))
        ]
        
        brief = {
            'brief_title': f"Feature Brief: {feature_name}",
            'feature_name': feature_name,
            'problem_statement': self._generate_problem_statement(feature, relevant_insights),
            'opportunity_assessment': self._generate_opportunity_assessment(
                feature, business_analysis
            ),
            'solution_design': self._generate_solution_design(feature, ux_design),
            'impact_assessment': self._generate_feature_impact_assessment(
                feature, impact_assessment
            ),
            'implementation_plan': self._generate_implementation_plan(feature, impact_assessment),
            'success_metrics': self._generate_feature_success_metrics(feature, impact_assessment),
            'prioritization_score': feature.get('priority_score', 0.5),
            'confidence_score': self._calculate_brief_confidence(feature, insights),
            'evidence': {
                'supporting_insights': [insight.id for insight in relevant_insights],
                'business_analysis_evidence': self._extract_business_evidence(feature, business_analysis),
                'ux_design_evidence': self._extract_ux_evidence(feature, ux_design)
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return brief
    
    async def _generate_overall_brief(self, briefs: List[Dict[str, Any]],
                                    insights: List[Insight],
                                    business_analysis: Dict[str, Any],
                                    feature_ideation: Dict[str, Any],
                                    ux_design: Dict[str, Any],
                                    impact_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall product brief"""
        # Aggregate all feature data
        all_features = feature_ideation.get('prioritized_features', [])
        
        brief = {
            'brief_title': 'Product Brief: Overall Strategy',
            'feature_name': 'Product Strategy',
            'problem_statement': self._generate_overall_problem_statement(insights, business_analysis),
            'opportunity_assessment': self._generate_overall_opportunity_assessment(business_analysis),
            'solution_design': self._generate_overall_solution_design(all_features, ux_design),
            'impact_assessment': self._generate_overall_impact_assessment(impact_assessment),
            'implementation_plan': self._generate_overall_implementation_plan(all_features, impact_assessment),
            'success_metrics': self._generate_overall_success_metrics(briefs, impact_assessment),
            'prioritization_score': np.mean([b.get('prioritization_score', 0.5) for b in briefs]),
            'confidence_score': self._calculate_brief_confidence(None, insights),
            'feature_briefs': [b.get('brief_title') for b in briefs],
            'evidence': {
                'supporting_insights': [insight.id for insight in insights],
                'business_analysis_evidence': business_analysis.get('market_opportunities', {}),
                'ux_design_evidence': ux_design.get('design_principles', {}),
                'feature_count': len(all_features)
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return brief
    
    def _generate_problem_statement(self, feature: Dict[str, Any], insights: List[Insight]) -> str:
        """Generate problem statement for feature"""
        feature_name = feature.get('name', 'Feature')
        user_problems = feature.get('user_problems', [])
        
        if user_problems:
            problem_text = f"Users are experiencing {', '.join(user_problems[:3])} when using {feature_name}"
        else:
            # Extract from insights
            insight_texts = [insight.insight_statement for insight in insights[:3]]
            problem_text = f"Based on analysis, key challenges include: {', '.join(insight_texts)}"
        
        return problem_text
    
    def _generate_opportunity_assessment(self, feature: Dict[str, Any], 
                                       business_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate opportunity assessment"""
        market_opportunities = business_analysis.get('market_opportunities', [])
        
        return {
            'market_size': feature.get('market_size', 'medium'),
            'growth_potential': feature.get('growth_potential', 'medium'),
            'competitive_advantage': feature.get('competitive_advantage', 'moderate'),
            'revenue_potential': feature.get('revenue_potential', 'medium'),
            'strategic_alignment': feature.get('strategic_alignment', 'high'),
            'opportunity_score': feature.get('opportunity_score', 0.7)
        }
    
    def _generate_solution_design(self, feature: Dict[str, Any], ux_design: Dict[str, Any]) -> Dict[str, Any]:
        """Generate solution design"""
        interface_concepts = ux_design.get('interface_concepts', [])
        
        return {
            'core_functionality': feature.get('core_functionality', []),
            'user_interface': feature.get('ui_requirements', []),
            'interaction_patterns': feature.get('interaction_patterns', []),
            'design_principles': ux_design.get('design_principles', {}),
            'technical_requirements': feature.get('technical_requirements', []),
            'integration_points': feature.get('integration_points', [])
        }
    
    def _generate_feature_impact_assessment(self, feature: Dict[str, Any], 
                                          impact_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate impact assessment for feature"""
        return {
            'business_impact': feature.get('business_impact', 'medium'),
            'user_impact': feature.get('user_impact', 'high'),
            'technical_impact': feature.get('technical_impact', 'medium'),
            'implementation_complexity': feature.get('implementation_complexity', 'medium'),
            'resource_requirements': feature.get('resource_requirements', 'medium'),
            'risk_level': feature.get('risk_level', 'low')
        }
    
    def _generate_implementation_plan(self, feature: Dict[str, Any], 
                                    impact_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate implementation plan"""
        complexity = feature.get('implementation_complexity', 'medium')
        
        if complexity == 'low':
            timeline = '2-4 weeks'
            phases = ['design', 'development', 'testing', 'deployment']
        elif complexity == 'medium':
            timeline = '1-3 months'
            phases = ['research', 'design', 'development', 'testing', 'deployment']
        else:  # high
            timeline = '3-6 months'
            phases = ['research', 'prototyping', 'design', 'development', 'testing', 'deployment']
        
        return {
            'timeline': timeline,
            'phases': phases,
            'milestones': self._generate_milestones(feature, phases),
            'dependencies': feature.get('dependencies', []),
            'resource_allocation': feature.get('resource_allocation', {})
        }
    
    def _generate_feature_success_metrics(self, feature: Dict[str, Any], 
                                       impact_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate success metrics for feature"""
        return [
            {
                'metric_name': 'User Adoption Rate',
                'metric_type': 'adoption',
                'target_value': feature.get('adoption_target', '70%'),
                'measurement_method': 'user_analytics',
                'timeframe': '3 months'
            },
            {
                'metric_name': 'User Satisfaction Score',
                'metric_type': 'satisfaction',
                'target_value': feature.get('satisfaction_target', '4.5/5'),
                'measurement_method': 'surveys',
                'timeframe': '6 months'
            },
            {
                'metric_name': 'Feature Usage Frequency',
                'metric_type': 'engagement',
                'target_value': feature.get('usage_target', '3x/week'),
                'measurement_method': 'usage_analytics',
                'timeframe': '3 months'
            }
        ]
    
    def _generate_overall_problem_statement(self, insights: List[Insight], 
                                         business_analysis: Dict[str, Any]) -> str:
        """Generate overall problem statement"""
        market_opportunities = business_analysis.get('market_opportunities', [])
        
        if market_opportunities:
            top_opportunity = market_opportunities[0] if market_opportunities else None
            if top_opportunity:
                return f"Market analysis reveals opportunity in {top_opportunity.get('area', 'multiple areas')} with significant unmet needs"
        
        # Fallback to insights
        insight_summary = ', '.join([insight.insight_statement for insight in insights[:3]])
        return f"Strategic analysis indicates key opportunities: {insight_summary}"
    
    def _generate_overall_opportunity_assessment(self, business_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall opportunity assessment"""
        market_opportunities = business_analysis.get('market_opportunities', [])
        competitive_landscape = business_analysis.get('competitive_landscape', {})
        
        return {
            'total_market_opportunities': len(market_opportunities),
            'high_potential_opportunities': len([op for op in market_opportunities if op.get('potential') == 'high']),
            'competitive_advantage': competitive_landscape.get('our_position', 'moderate'),
            'market_readiness': business_analysis.get('market_readiness', 'medium'),
            'strategic_alignment': 'high'
        }
    
    def _generate_overall_solution_design(self, features: List[Dict[str, Any]], 
                                       ux_design: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall solution design"""
        return {
            'product_architecture': 'modular',
            'core_features': [f.get('name', 'Feature') for f in features[:5]],
            'design_system': ux_design.get('design_principles', {}),
            'technology_stack': 'modern_web_stack',
            'integration_strategy': 'api_first',
            'scalability_approach': 'microservices'
        }
    
    def _generate_overall_impact_assessment(self, impact_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall impact assessment"""
        return {
            'business_impact': impact_assessment.get('business_impact', {}),
            'user_impact': impact_assessment.get('user_impact', {}),
            'technical_impact': impact_assessment.get('technical_impact', {}),
            'market_impact': 'high',
            'competitive_impact': 'moderate',
            'organizational_impact': 'high'
        }
    
    def _generate_overall_implementation_plan(self, features: List[Dict[str, Any]], 
                                            impact_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall implementation plan"""
        return {
            'timeline': '6-12 months',
            'phases': ['discovery', 'design', 'development', 'testing', 'launch', 'optimization'],
            'resource_requirements': {
                'development_team': '5-7 developers',
                'design_team': '2-3 designers',
                'product_team': '1-2 product managers',
                'qa_team': '2-3 QA engineers'
            },
            'budget_estimate': 'medium',
            'risk_mitigation': impact_assessment.get('risk_assessment', {})
        }
    
    def _generate_overall_success_metrics(self, briefs: List[Dict[str, Any]], 
                                         impact_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate overall success metrics"""
        return [
            {
                'metric_name': 'Product Market Fit',
                'metric_type': 'market_fit',
                'target_value': '80%',
                'measurement_method': 'market_research',
                'timeframe': '12 months'
            },
            {
                'metric_name': 'Revenue Growth',
                'metric_type': 'revenue',
                'target_value': '25%',
                'measurement_method': 'financial_analysis',
                'timeframe': '12 months'
            },
            {
                'metric_name': 'Customer Satisfaction',
                'metric_type': 'satisfaction',
                'target_value': '4.2/5',
                'measurement_method': 'nps_surveys',
                'timeframe': '6 months'
            }
        ]
    
    def _generate_milestones(self, feature: Dict[str, Any], phases: List[str]) -> List[Dict[str, Any]]:
        """Generate milestones for implementation"""
        milestones = []
        total_duration = len(phases) * 2  # 2 weeks per phase
        
        for i, phase in enumerate(phases):
            milestone = {
                'milestone_name': f"{phase.capitalize()} Complete",
                'phase': phase,
                'estimated_completion': f"Week {i*2 + 2}",
                'deliverables': self._get_phase_deliverables(phase, feature),
                'success_criteria': self._get_phase_success_criteria(phase)
            }
            milestones.append(milestone)
        
        return milestones
    
    def _get_phase_deliverables(self, phase: str, feature: Dict[str, Any]) -> List[str]:
        """Get deliverables for a phase"""
        deliverable_map = {
            'research': ['market_analysis', 'user_research', 'competitive_analysis'],
            'design': ['wireframes', 'mockups', 'design_specifications'],
            'development': ['working_code', 'unit_tests', 'documentation'],
            'testing': ['test_cases', 'bug_reports', 'performance_tests'],
            'deployment': ['production_deployment', 'monitoring_setup', 'user_documentation']
        }
        
        return deliverable_map.get(phase, ['phase_completion'])
    
    def _get_phase_success_criteria(self, phase: str) -> List[str]:
        """Get success criteria for a phase"""
        criteria_map = {
            'research': ['insights_generated', 'opportunities_identified', 'requirements_defined'],
            'design': ['design_approved', 'user_feedback_positive', 'specifications_complete'],
            'development': ['functionality_implemented', 'tests_passing', 'code_reviewed'],
            'testing': ['critical_bugs_fixed', 'performance_meets_requirements', 'security_approved'],
            'deployment': ['production_ready', 'monitoring_active', 'users_trained']
        }
        
        return criteria_map.get(phase, ['phase_completed'])
    
    def _extract_business_evidence(self, feature: Dict[str, Any], business_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract business evidence for brief"""
        return {
            'market_opportunities': business_analysis.get('market_opportunities', [])[:3],
            'competitive_landscape': business_analysis.get('competitive_landscape', {}),
            'business_model': business_analysis.get('business_model', {}),
            'strategic_positioning': business_analysis.get('strategic_positioning', {})
        }
    
    def _extract_ux_evidence(self, feature: Dict[str, Any], ux_design: Dict[str, Any]) -> Dict[str, Any]:
        """Extract UX evidence for brief"""
        return {
            'user_journeys': ux_design.get('user_journeys', [])[:2],
            'interface_concepts': ux_design.get('interface_concepts', [])[:2],
            'interaction_design': ux_design.get('interaction_design', {}),
            'design_principles': ux_design.get('design_principles', {})
        }
    
    def _calculate_analysis_confidence(self, insights: List[Insight]) -> float:
        """Calculate confidence in analysis"""
        if not insights:
            return 0.5
        
        # Average confidence from insights
        insight_confidences = [getattr(insight, 'confidence_score', 0.5) for insight in insights]
        return np.mean(insight_confidences)
    
    def _calculate_ideation_confidence(self, insights: List[Insight], feature_concepts: List[Dict[str, Any]]) -> float:
        """Calculate confidence in ideation"""
        base_confidence = self._calculate_analysis_confidence(insights)
        
        # Boost based on number of concepts
        concept_boost = min(len(feature_concepts) / 10, 0.2)
        
        return min(base_confidence + concept_boost, 1.0)
    
    def _calculate_design_confidence(self, insights: List[Insight]) -> float:
        """Calculate confidence in design"""
        base_confidence = self._calculate_analysis_confidence(insights)
        
        # Design has slightly lower confidence due to subjectivity
        return base_confidence * 0.9
    
    def _calculate_assessment_confidence(self, insights: List[Insight]) -> float:
        """Calculate confidence in assessment"""
        base_confidence = self._calculate_analysis_confidence(insights)
        
        # Assessment confidence based on multiple data points
        return min(base_confidence * 1.1, 1.0)
    
    def _calculate_brief_confidence(self, feature: Dict[str, Any] = None, insights: List[Insight] = None) -> float:
        """Calculate confidence in brief"""
        if feature:
            # Feature-specific confidence
            feature_confidence = feature.get('confidence_score', 0.5)
            insight_confidence = self._calculate_analysis_confidence(insights) if insights else 0.5
            return (feature_confidence + insight_confidence) / 2
        else:
            # Overall brief confidence
            return self._calculate_analysis_confidence(insights) if insights else 0.5
    
    async def _store_briefs(self, briefs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Store briefs in database following detailed specification"""
        self.log_info(f"Storing {len(briefs)} briefs")
        
        stored_briefs = []
        
        for brief_data in briefs:
            try:
                # Create brief model
                brief = ProductBriefCreate(**brief_data)
                
                # Store in database
                stored_brief = await brief_repo.create(brief.dict())
                
                if stored_brief:
                    stored_briefs.append(stored_brief)
                    
                    # Cache for quick retrieval
                    cache_key = f"brief:{stored_brief['id']}"
                    cache_manager.set(cache_key, stored_brief, ttl=3600)
                
            except Exception as e:
                self.log_error(f"Failed to store brief: {e}")
        
        self.log_info(f"Stored {len(stored_briefs)} briefs successfully")
        return stored_briefs
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform agent-specific health checks"""
        health_status = {}
        
        # Check analysis libraries
        if ANALYSIS_AVAILABLE:
            health_status['analysis_libs'] = 'available'
        else:
            health_status['analysis_libs'] = 'not_available'
        
        # Check individual components
        components = [
            ('business_analysis', self.business_analysis),
            ('feature_ideation', self.feature_ideation),
            ('ux_design', self.ux_design),
            ('impact_assessment', self.impact_assessment)
        ]
        
        for component_name, component in components:
            try:
                if hasattr(component, 'health_check'):
                    health_status[component_name] = await component.health_check()
                else:
                    health_status[component_name] = 'working'
            except Exception as e:
                health_status[component_name] = f'error: {str(e)}'
        
        # Check database connections
        try:
            recent_briefs = await brief_repo.get_recent_briefs(limit=1)
            health_status['brief_database'] = 'connected'
        except Exception as e:
            health_status['brief_database'] = f'error: {str(e)}'
        
        try:
            recent_insights = await insight_repo.get_recent_insights(limit=1)
            health_status['insight_database'] = 'connected'
        except Exception as e:
            health_status['insight_database'] = f'error: {str(e)}'
        
        return health_status
