"""
Problem Classifier Implementation
Classifies signals into problem categories using keyword matching and rules
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from ....utils.nlp_utils import nlp_utils


class ProblemClassifier:
    """Problem classifier implementation following detailed specification"""
    
    def __init__(self):
        self.problem_categories = {
            'ui': ['interface', 'design', 'layout', 'navigation', 'usability', 'user experience'],
            'feature': ['feature', 'functionality', 'capability', 'option', 'setting'],
            'pricing': ['price', 'cost', 'expensive', 'cheap', 'subscription', 'payment'],
            'support': ['support', 'help', 'customer service', 'assistance', 'response'],
            'performance': ['slow', 'fast', 'performance', 'speed', 'lag', 'crash'],
            'onboarding': ['onboarding', 'tutorial', 'guide', 'getting started', 'setup']
        }
    
    async def classify_signal(self, signal) -> Dict[str, Any]:
        """Classify signal into problem categories"""
        content = signal.content.lower()
        
        # Calculate category scores
        category_scores = {}
        for category, keywords in self.problem_categories.items():
            score = self._calculate_category_score(content, keywords)
            category_scores[category] = score
        
        # Determine primary category
        primary_category = max(category_scores, key=category_scores.get)
        primary_score = category_scores[primary_category]
        
        # Extract problem-specific information
        entities = nlp_utils.extract_entities(signal.content)
        keywords = nlp_utils.extract_keywords(signal.content, max_keywords=5)
        
        classification = {
            'signal_id': signal.id,
            'primary_category': primary_category,
            'category_scores': category_scores,
            'confidence': primary_score,
            'entities': entities,
            'keywords': keywords,
            'is_problem': self._is_problem_statement(content),
            'problem_type': self._determine_problem_type(content, primary_category),
            'urgency': self._assess_urgency(content),
            'severity': self._assess_severity(content, primary_score),
            'classification_metadata': {
                'method': 'keyword_based',
                'model_version': '1.0',
                'classified_at': datetime.now().isoformat()
            }
        }
        
        return classification
    
    def _calculate_category_score(self, content: str, keywords: List[str]) -> float:
        """Calculate score for a category based on keyword matches"""
        score = 0.0
        total_keywords = len(keywords)
        
        for keyword in keywords:
            if keyword in content:
                score += 1
        
        # Normalize by total keywords and content length
        normalized_score = score / (total_keywords * len(content.split()) / 100)
        return min(normalized_score, 1.0)
    
    def _is_problem_statement(self, content: str) -> bool:
        """Determine if content is a problem statement"""
        problem_indicators = [
            'problem', 'issue', 'bug', 'error', 'broken', 'not working',
            'difficult', 'confusing', 'frustrating', 'annoying',
            'can\'t', 'unable', 'failed', 'issue with'
        ]
        
        return any(indicator in content for indicator in problem_indicators)
    
    def _determine_problem_type(self, content: str, category: str) -> str:
        """Determine specific problem type"""
        if category == 'ui':
            if any(word in content for word in ['find', 'locate', 'discover']):
                return 'navigation_issue'
            elif any(word in content for word in ['look', 'design', 'appearance']):
                return 'visual_design_issue'
            else:
                return 'general_ui_issue'
        
        elif category == 'feature':
            if any(word in content for word in ['missing', 'need', 'want']):
                return 'missing_feature'
            elif any(word in content for word in ['broken', 'not working']):
                return 'broken_feature'
            else:
                return 'feature_request'
        
        elif category == 'pricing':
            if any(word in content for word in ['expensive', 'costly']):
                return 'price_too_high'
            elif any(word in content for word in ['confusing', 'unclear']):
                return 'pricing_confusion'
            else:
                return 'general_pricing_issue'
        
        else:
            return 'general_problem'
    
    def _assess_urgency(self, content: str) -> str:
        """Assess urgency of problem"""
        high_urgency = ['urgent', 'critical', 'emergency', 'immediately', 'asap']
        medium_urgency = ['important', 'significant', 'major', 'serious']
        
        if any(word in content for word in high_urgency):
            return 'high'
        elif any(word in content for word in medium_urgency):
            return 'medium'
        else:
            return 'low'
    
    def _assess_severity(self, content: str, confidence: float) -> str:
        """Assess severity based on content and confidence"""
        if confidence > 0.8 and any(word in content for word in ['critical', 'urgent', 'emergency']):
            return 'critical'
        elif confidence > 0.6:
            return 'high'
        elif confidence > 0.4:
            return 'medium'
        else:
            return 'low'
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for problem classifier"""
        return {
            'status': 'working',
            'categories_configured': len(self.problem_categories),
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
