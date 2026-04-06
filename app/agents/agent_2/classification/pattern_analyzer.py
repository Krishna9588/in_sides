"""
Pattern Analyzer Implementation
Analyzes patterns in signals including repetition, sentiment, temporal patterns
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from ....utils.nlp_utils import nlp_utils


class PatternAnalyzer:
    """Pattern analyzer implementation following detailed specification"""
    
    async def analyze_signal(self, signal) -> Dict[str, Any]:
        """Analyze patterns in signal"""
        content = signal.content.lower()
        
        # Pattern detection
        patterns = {
            'repetition': self._detect_repetition(content),
            'sentiment': self._analyze_sentiment(content),
            'temporal': self._analyze_temporal_patterns(signal),
            'frequency': self._analyze_frequency_patterns(signal)
        }
        
        # Calculate pattern confidence
        pattern_confidence = self._calculate_pattern_confidence(patterns)
        
        return {
            'signal_id': signal.id,
            'patterns': patterns,
            'confidence': pattern_confidence,
            'analysis_metadata': {
                'method': 'rule_based',
                'analyzed_at': datetime.now().isoformat()
            }
        }
    
    def _detect_repetition(self, content: str) -> Dict[str, Any]:
        """Detect repetitive patterns"""
        words = content.split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find repeated words
        repeated_words = {word: count for word, count in word_counts.items() if count > 1}
        
        return {
            'repeated_words': repeated_words,
            'repetition_score': len(repeated_words) / len(words) if words else 0
        }
    
    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment patterns"""
        positive_words = ['good', 'great', 'excellent', 'love', 'amazing', 'perfect']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'worst', 'broken']
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment_score = 0.5  # Neutral
        else:
            sentiment_score = positive_count / total_sentiment_words
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': 'positive' if sentiment_score > 0.6 else 'negative' if sentiment_score < 0.4 else 'neutral',
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def _analyze_temporal_patterns(self, signal) -> Dict[str, Any]:
        """Analyze temporal patterns"""
        if not signal.created_at:
            return {'temporal_patterns': 'unknown'}
        
        # Simple temporal analysis
        created_time = datetime.fromisoformat(signal.created_at.replace('Z', '+00:00'))
        hour = created_time.hour
        day_of_week = created_time.weekday()
        
        return {
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'is_weekend': day_of_week >= 5,
            'time_category': 'morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening'
        }
    
    def _analyze_frequency_patterns(self, signal) -> Dict[str, Any]:
        """Analyze frequency patterns"""
        return {
            'source_frequency': 'single_occurrence',  # Would need more data for real analysis
            'content_length': len(signal.content),
            'word_count': len(signal.content.split())
        }
    
    def _calculate_pattern_confidence(self, patterns: Dict[str, Any]) -> float:
        """Calculate confidence in pattern analysis"""
        confidence = 0.5  # Base confidence
        
        # Add confidence based on pattern strength
        if patterns.get('repetition', {}).get('repetition_score', 0) > 0.1:
            confidence += 0.1
        
        if patterns.get('sentiment', {}).get('sentiment_score') != 0.5:  # Not neutral
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for pattern analyzer"""
        return {
            'status': 'working',
            'pattern_types': ['repetition', 'sentiment', 'temporal', 'frequency'],
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
