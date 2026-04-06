"""
Trend Analyzer Implementation
Analyzes time-based and content-based trends in signals
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from ....utils.nlp_utils import nlp_utils


class TrendAnalyzer:
    """Trend analyzer implementation following detailed specification"""
    
    async def analyze_trends(self, signals: List) -> List[Dict[str, Any]]:
        """Analyze trends in signals"""
        trends = []
        
        # Time-based trends
        time_trends = self._analyze_time_trends(signals)
        trends.extend(time_trends)
        
        # Content-based trends
        content_trends = self._analyze_content_trends(signals)
        trends.extend(content_trends)
        
        return trends
    
    def _analyze_time_trends(self, signals: List) -> List[Dict[str, Any]]:
        """Analyze time-based trends"""
        # Group signals by time periods
        time_groups = {}
        for signal in signals:
            if signal.created_at:
                created_time = datetime.fromisoformat(signal.created_at.replace('Z', '+00:00'))
                time_key = created_time.strftime('%Y-%m-%d')  # Daily grouping
                
                if time_key not in time_groups:
                    time_groups[time_key] = []
                time_groups[time_key].append(signal)
        
        # Calculate trends
        trends = []
        for time_key, group_signals in time_groups.items():
            if len(group_signals) >= 3:  # Minimum threshold for trend
                trends.append({
                    'trend_type': 'time_based',
                    'time_period': time_key,
                    'signal_count': len(group_signals),
                    'avg_confidence': np.mean([s.confidence_score for s in group_signals]),
                    'trend_strength': len(group_signals) / len(signals)
                })
        
        return trends
    
    def _analyze_content_trends(self, signals: List) -> List[Dict[str, Any]]:
        """Analyze content-based trends"""
        # Extract keywords from all signals
        all_keywords = []
        for signal in signals:
            keywords = nlp_utils.extract_keywords(signal.content, max_keywords=5)
            all_keywords.extend(keywords)
        
        # Count keyword frequencies
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Identify trending keywords
        trending_keywords = sorted(
            keyword_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10 trending keywords
        
        trends = []
        for keyword, count in trending_keywords:
            if count >= 2:  # Minimum threshold
                trends.append({
                    'trend_type': 'content_based',
                    'keyword': keyword,
                    'frequency': count,
                    'trend_strength': count / len(signals)
                })
        
        return trends
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for trend analyzer"""
        return {
            'status': 'working',
            'trend_types': ['time_based', 'content_based'],
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
