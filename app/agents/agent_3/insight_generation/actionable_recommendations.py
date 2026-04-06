"""
Actionable Recommendations Implementation
Generates actionable recommendations from analysis results
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from .....utils.nlp_utils import nlp_utils
from .....config.settings import settings


class ActionableRecommendations:
    """Actionable recommendations implementation following detailed specification"""
    
    async def generate_actionable_recommendations(self, problems: List, 
                                               root_cause_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from analysis results"""
        if not problems:
            return []
        
        try:
            recommendations = []
            
            # 1. Immediate action recommendations
            immediate_actions = await self._generate_immediate_actions(problems, root_cause_results)
            recommendations.extend(immediate_actions)
            
            # 2. Short-term recommendations
            short_term_actions = await self._generate_short_term_actions(problems, root_cause_results)
            recommendations.extend(short_term_actions)
            
            # 3. Long-term strategic recommendations
            long_term_actions = await self._generate_long_term_actions(problems, root_cause_results)
            recommendations.extend(long_term_actions)
            
            # 4. Resource allocation recommendations
            resource_actions = await self._generate_resource_actions(problems, root_cause_results)
            recommendations.extend(resource_actions)
            
            return recommendations
            
        except Exception as e:
            print(f"Actionable recommendations generation failed: {e}")
            return []
    
    async def _generate_immediate_actions(self, problems: List, root_cause_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate immediate action recommendations"""
        recommendations = []
        
        # Focus on high-severity and critical issues
        critical_problems = [
            p for p in problems 
            if getattr(p, 'severity', 'medium') in ['high', 'critical']
        ]
        
        for problem in critical_problems:
            # Generate specific actions based on problem category
            actions = await self._generate_category_specific_actions(problem)
            
            for action in actions:
                action.update({
                    'recommendation_type': 'immediate_action',
                    'priority': 'critical',
                    'timeframe': 'immediate',
                    'target_problem_id': problem.id,
                    'confidence': 0.9,  # High confidence for critical issues
                    'generated_at': datetime.now().isoformat()
                })
                recommendations.append(action)
        
        return recommendations
    
    async def _generate_short_term_actions(self, problems: List, root_cause_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate short-term action recommendations"""
        recommendations = []
        
        # Focus on medium-severity issues
        medium_problems = [
            p for p in problems 
            if getattr(p, 'severity', 'medium') == 'medium'
        ]
        
        for problem in medium_problems:
            actions = await self._generate_category_specific_actions(problem)
            
            for action in actions:
                action.update({
                    'recommendation_type': 'short_term_action',
                    'priority': 'high',
                    'timeframe': '1-4 weeks',
                    'target_problem_id': problem.id,
                    'confidence': 0.8,
                    'generated_at': datetime.now().isoformat()
                })
                recommendations.append(action)
        
        return recommendations
    
    async def _generate_long_term_actions(self, problems: List, root_cause_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate long-term strategic action recommendations"""
        recommendations = []
        
        # Focus on systemic issues and patterns
        underlying_factors = root_cause_results.get('underlying_factors', [])
        systemic_factors = [f for f in underlying_factors if f.get('importance_score', 0) > 0.7]
        
        for factor in systemic_factors:
            # Generate strategic actions based on factor type
            strategic_actions = await self._generate_strategic_actions(factor)
            
            for action in strategic_actions:
                action.update({
                    'recommendation_type': 'long_term_strategic',
                    'priority': 'medium',
                    'timeframe': '3-12 months',
                    'target_factor_id': factor.get('contributing_problems', []),
                    'confidence': 0.7,
                    'generated_at': datetime.now().isoformat()
                })
                recommendations.append(action)
        
        return recommendations
    
    async def _generate_resource_actions(self, problems: List, root_cause_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate resource allocation recommendations"""
        recommendations = []
        
        # Analyze resource requirements
        resource_analysis = await self._analyze_resource_requirements(problems, root_cause_results)
        
        for resource_need in resource_analysis:
            recommendations.append({
                'recommendation_type': 'resource_allocation',
                'priority': 'medium',
                'timeframe': '1-3 months',
                'resource_type': resource_need['resource_type'],
                'recommendation': resource_need['recommendation'],
                'estimated_effort': resource_need['estimated_effort'],
                'impact_areas': resource_need['impact_areas'],
                'confidence': 0.75,
                'generated_at': datetime.now().isoformat()
            })
        
        return recommendations
    
    async def _generate_category_specific_actions(self, problem) -> List[Dict[str, Any]]:
        """Generate actions specific to problem category"""
        category = getattr(problem, 'problem_category', 'general')
        problem_text = problem.problem_statement.lower()
        
        actions = []
        
        if category == 'ui':
            actions = await self._generate_ui_actions(problem_text)
        elif category == 'feature':
            actions = await self._generate_feature_actions(problem_text)
        elif category == 'performance':
            actions = await self._generate_performance_actions(problem_text)
        elif category == 'support':
            actions = await self._generate_support_actions(problem_text)
        elif category == 'pricing':
            actions = await self._generate_pricing_actions(problem_text)
        else:
            actions = await self._generate_general_actions(problem_text)
        
        return actions
    
    async def _generate_ui_actions(self, problem_text: str) -> List[Dict[str, Any]]:
        """Generate UI-specific actions"""
        actions = []
        
        ui_issues = {
            'navigation': ['conduct usability testing', 'redesign navigation flow', 'improve information architecture'],
            'design': ['perform design audit', 'update visual design system', 'implement responsive design'],
            'usability': ['simplify user interface', 'reduce cognitive load', 'improve accessibility']
        }
        
        for issue_type, action_list in ui_issues.items():
            if any(indicator in problem_text for indicator in [issue_type]):
                for action in action_list:
                    actions.append({
                        'action_description': action,
                        'action_category': 'ui_improvement',
                        'issue_type': issue_type
                    })
        
        return actions
    
    async def _generate_feature_actions(self, problem_text: str) -> List[Dict[str, Any]]:
        """Generate feature-specific actions"""
        actions = []
        
        feature_issues = {
            'missing_feature': ['conduct feature research', 'prioritize feature roadmap', 'develop mvp'],
            'broken_feature': ['fix bugs', 'improve reliability', 'enhance functionality'],
            'feature_request': ['evaluate feasibility', 'create prototype', 'gather user feedback']
        }
        
        for issue_type, action_list in feature_issues.items():
            if any(indicator in problem_text for indicator in [issue_type]):
                for action in action_list:
                    actions.append({
                        'action_description': action,
                        'action_category': 'feature_development',
                        'issue_type': issue_type
                    })
        
        return actions
    
    async def _generate_performance_actions(self, problem_text: str) -> List[Dict[str, Any]]:
        """Generate performance-specific actions"""
        actions = []
        
        performance_issues = {
            'slow_performance': ['optimize code', 'upgrade infrastructure', 'implement caching'],
            'reliability': ['improve error handling', 'add monitoring', 'enhance testing'],
            'scalability': ['implement load balancing', 'optimize database', 'scale infrastructure']
        }
        
        for issue_type, action_list in performance_issues.items():
            if any(indicator in problem_text for indicator in [issue_type]):
                for action in action_list:
                    actions.append({
                        'action_description': action,
                        'action_category': 'performance_optimization',
                        'issue_type': issue_type
                    })
        
        return actions
    
    async def _generate_support_actions(self, problem_text: str) -> List[Dict[str, Any]]:
        """Generate support-specific actions"""
        actions = []
        
        support_issues = {
            'customer_support': ['improve response time', 'enhance knowledge base', 'implement chatbot'],
            'documentation': ['create better guides', 'add video tutorials', 'improve faq']
        }
        
        for issue_type, action_list in support_issues.items():
            if any(indicator in problem_text for indicator in [issue_type]):
                for action in action_list:
                    actions.append({
                        'action_description': action,
                        'action_category': 'support_improvement',
                        'issue_type': issue_type
                    })
        
        return actions
    
    async def _generate_pricing_actions(self, problem_text: str) -> List[Dict[str, Any]]:
        """Generate pricing-specific actions"""
        actions = []
        
        pricing_issues = {
            'price_too_high': ['conduct price analysis', 'review value proposition', 'consider tiered pricing'],
            'pricing_confusion': ['simplify pricing structure', 'improve cost communication', 'add price calculator']
        }
        
        for issue_type, action_list in pricing_issues.items():
            if any(indicator in problem_text for indicator in [issue_type]):
                for action in action_list:
                    actions.append({
                        'action_description': action,
                        'action_category': 'pricing_optimization',
                        'issue_type': issue_type
                    })
        
        return actions
    
    async def _generate_general_actions(self, problem_text: str) -> List[Dict[str, Any]]:
        """Generate general actions for unspecified categories"""
        actions = []
        
        general_actions = [
            'conduct user research', 'perform competitive analysis', 'gather stakeholder feedback',
            'analyze usage data', 'create improvement roadmap', 'implement monitoring system'
        ]
        
        for action in general_actions:
            actions.append({
                'action_description': action,
                'action_category': 'general_improvement'
            })
        
        return actions
    
    async def _generate_strategic_actions(self, factor: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategic actions based on underlying factors"""
        factor_type = factor.get('factor_type', 'general')
        actions = []
        
        strategic_mappings = {
            'technical': ['invest in technical debt reduction', 'upgrade infrastructure', 'implement best practices'],
            'user': ['improve user experience', 'conduct user research', 'enhance onboarding'],
            'business': ['optimize business processes', 'review strategic alignment', 'improve go-to-market'],
            'environmental': ['monitor market trends', 'adapt to regulatory changes', 'optimize for scalability']
        }
        
        action_list = strategic_mappings.get(factor_type, ['conduct comprehensive analysis'])
        
        for action in action_list:
            actions.append({
                'action_description': action,
                'action_category': 'strategic_initiative',
                'factor_type': factor_type
            })
        
        return actions
    
    async def _analyze_resource_requirements(self, problems: List, root_cause_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze resource requirements for addressing problems"""
        resource_needs = []
        
        # Analyze technical resource needs
        technical_factors = [f for f in root_cause_results.get('underlying_factors', []) if f.get('factor_type') == 'technical']
        
        if technical_factors:
            resource_needs.append({
                'resource_type': 'technical_resources',
                'recommendation': 'Allocate development resources for technical improvements',
                'estimated_effort': self._estimate_technical_effort(technical_factors),
                'impact_areas': ['infrastructure', 'development', 'testing'],
                'priority_factors': technical_factors[:3]
            })
        
        # Analyze human resource needs
        user_factors = [f for f in root_cause_results.get('underlying_factors', []) if f.get('factor_type') == 'user']
        
        if user_factors:
            resource_needs.append({
                'resource_type': 'user_experience_resources',
                'recommendation': 'Invest in UX research and design resources',
                'estimated_effort': self._estimate_ux_effort(user_factors),
                'impact_areas': ['research', 'design', 'testing'],
                'priority_factors': user_factors[:3]
            })
        
        # Analyze business resource needs
        business_factors = [f for f in root_cause_results.get('underlying_factors', []) if f.get('factor_type') == 'business']
        
        if business_factors:
            resource_needs.append({
                'resource_type': 'business_resources',
                'recommendation': 'Allocate strategic planning and analysis resources',
                'estimated_effort': self._estimate_business_effort(business_factors),
                'impact_areas': ['planning', 'analysis', 'strategy'],
                'priority_factors': business_factors[:3]
            })
        
        return resource_needs
    
    def _estimate_technical_effort(self, technical_factors: List[Dict[str, Any]]) -> str:
        """Estimate technical effort required"""
        if not technical_factors:
            return 'low'
        
        avg_importance = np.mean([f.get('importance_score', 0) for f in technical_factors])
        
        if avg_importance > 0.8:
            return 'high'
        elif avg_importance > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_ux_effort(self, user_factors: List[Dict[str, Any]]) -> str:
        """Estimate UX effort required"""
        if not user_factors:
            return 'low'
        
        avg_importance = np.mean([f.get('importance_score', 0) for f in user_factors])
        
        if avg_importance > 0.7:
            return 'high'
        elif avg_importance > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_business_effort(self, business_factors: List[Dict[str, Any]]) -> str:
        """Estimate business effort required"""
        if not business_factors:
            return 'low'
        
        avg_importance = np.mean([f.get('importance_score', 0) for f in business_factors])
        
        if avg_importance > 0.75:
            return 'high'
        elif avg_importance > 0.5:
            return 'medium'
        else:
            return 'low'
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for actionable recommendations"""
        return {
            'status': 'working',
            'recommendation_types': [
                'immediate_actions',
                'short_term_actions',
                'long_term_strategic',
                'resource_allocation'
            ],
            'action_categories': [
                'ui_improvement',
                'feature_development',
                'performance_optimization',
                'support_improvement',
                'pricing_optimization',
                'general_improvement',
                'strategic_initiative'
            ],
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
