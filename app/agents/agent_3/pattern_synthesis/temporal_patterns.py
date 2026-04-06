"""
Temporal Patterns Implementation
Analyzes temporal patterns in problem evolution
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
import numpy as np

from .....utils.nlp_utils import nlp_utils
from .....config.settings import settings


class TemporalPatterns:
    """Temporal patterns implementation following detailed specification"""
    
    async def detect_temporal_patterns(self, problems: List) -> List[Dict[str, Any]]:
        """Detect temporal patterns in problems"""
        if not problems:
            return []
        
        try:
            # Sort problems by creation time
            time_sorted_problems = sorted(
                [p for p in problems if hasattr(p, 'created_at')],
                key=lambda x: x.created_at
            )
            
            if len(time_sorted_problems) < 3:
                return []
            
            # Analyze different temporal patterns
            patterns = []
            
            # 1. Frequency patterns
            frequency_patterns = await self._analyze_frequency_patterns(time_sorted_problems)
            patterns.extend(frequency_patterns)
            
            # 2. Seasonal patterns
            seasonal_patterns = await self._analyze_seasonal_patterns(time_sorted_problems)
            patterns.extend(seasonal_patterns)
            
            # 3. Trend patterns
            trend_patterns = await self._analyze_trend_patterns(time_sorted_problems)
            patterns.extend(trend_patterns)
            
            # 4. Burst patterns
            burst_patterns = await self._analyze_burst_patterns(time_sorted_problems)
            patterns.extend(burst_patterns)
            
            return patterns
            
        except Exception as e:
            print(f"Temporal pattern detection failed: {e}")
            return []
    
    async def _analyze_frequency_patterns(self, time_sorted_problems: List) -> List[Dict[str, Any]]:
        """Analyze frequency patterns over time"""
        patterns = []
        
        # Group problems by time periods
        daily_groups = self._group_by_time_period(time_sorted_problems, 'daily')
        weekly_groups = self._group_by_time_period(time_sorted_problems, 'weekly')
        monthly_groups = self._group_by_time_period(time_sorted_problems, 'monthly')
        
        # Analyze frequency changes
        for period, groups in [('daily', daily_groups), ('weekly', weekly_groups), ('monthly', monthly_groups)]:
            frequency_changes = self._calculate_frequency_changes(groups)
            
            if frequency_changes['variance'] > settings.FREQUENCY_VARIANCE_THRESHOLD:
                patterns.append({
                    'pattern_type': 'frequency_change',
                    'time_period': period,
                    'frequency_data': groups,
                    'variance': frequency_changes['variance'],
                    'trend_direction': frequency_changes['trend'],
                    'significance': frequency_changes['significance']
                })
        
        return patterns
    
    async def _analyze_seasonal_patterns(self, time_sorted_problems: List) -> List[Dict[str, Any]]:
        """Analyze seasonal patterns"""
        patterns = []
        
        # Group by month and day of week
        monthly_patterns = self._group_by_month(time_sorted_problems)
        weekly_patterns = self._group_by_day_of_week(time_sorted_problems)
        
        # Analyze monthly seasonality
        monthly_avg = np.mean([len(problems) for problems in monthly_patterns.values()])
        monthly_variance = np.var([len(problems) for problems in monthly_patterns.values()])
        
        # Identify seasonal peaks
        for month, problems in monthly_patterns.items():
            count = len(problems)
            seasonal_index = (count - monthly_avg) / (monthly_avg + 1e-6)  # Avoid division by zero
            
            if abs(seasonal_index) > settings.SEASONALITY_THRESHOLD:
                patterns.append({
                    'pattern_type': 'seasonal_pattern',
                    'season_type': 'monthly',
                    'time_period': month,
                    'problem_count': count,
                    'seasonal_index': seasonal_index,
                    'is_peak': seasonal_index > 0,
                    'significance': abs(seasonal_index)
                })
        
        # Analyze weekly patterns
        for day, problems in weekly_patterns.items():
            count = len(problems)
            weekly_avg = np.mean([len(p) for p in weekly_patterns.values()])
            weekly_index = (count - weekly_avg) / (weekly_avg + 1e-6)
            
            if abs(weekly_index) > settings.WEEKLY_PATTERN_THRESHOLD:
                patterns.append({
                    'pattern_type': 'weekly_pattern',
                    'season_type': 'weekly',
                    'time_period': day,
                    'problem_count': count,
                    'weekly_index': weekly_index,
                    'is_peak': weekly_index > 0,
                    'significance': abs(weekly_index)
                })
        
        return patterns
    
    async def _analyze_trend_patterns(self, time_sorted_problems: List) -> List[Dict[str, Any]]:
        """Analyze trend patterns"""
        patterns = []
        
        # Calculate moving averages
        window_sizes = [7, 14, 30]  # Weekly, bi-weekly, monthly
        
        for window_size in window_sizes:
            if len(time_sorted_problems) >= window_size:
                moving_avg = self._calculate_moving_average(time_sorted_problems, window_size)
                trend_direction = self._calculate_trend_direction(moving_avg)
                
                patterns.append({
                    'pattern_type': 'trend_pattern',
                    'window_size': window_size,
                    'moving_average': moving_avg,
                    'trend_direction': trend_direction,
                    'trend_strength': self._calculate_trend_strength(moving_avg),
                    'significance': self._calculate_trend_significance(moving_avg, window_size)
                })
        
        return patterns
    
    async def _analyze_burst_patterns(self, time_sorted_problems: List) -> List[Dict[str, Any]]:
        """Analyze burst patterns (sudden increases in problem reports)"""
        patterns = []
        
        # Calculate daily problem counts
        daily_counts = self._group_by_time_period(time_sorted_problems, 'daily')
        dates = sorted(daily_counts.keys())
        
        if len(daily_counts) < 7:
            return patterns
        
        # Calculate baseline and detect bursts
        baseline_counts = list(daily_counts.values())[:7]  # First week as baseline
        baseline_avg = np.mean(baseline_counts)
        baseline_std = np.std(baseline_counts)
        
        # Detect bursts
        burst_threshold = baseline_avg + (2 * baseline_std)  # 2 sigma threshold
        
        for date, count in daily_counts.items():
            if date > dates[6]:  # After baseline period
                if count > burst_threshold:
                    burst_intensity = (count - baseline_avg) / baseline_std
                    
                    patterns.append({
                        'pattern_type': 'burst_pattern',
                        'burst_date': date,
                        'problem_count': count,
                        'baseline_avg': baseline_avg,
                        'burst_intensity': burst_intensity,
                        'burst_level': self._classify_burst_level(burst_intensity),
                        'significance': abs(burst_intensity)
                    })
        
        return patterns
    
    def _group_by_time_period(self, problems: List, period_type: str) -> Dict[str, List]:
        """Group problems by time period"""
        groups = {}
        
        for problem in problems:
            if not hasattr(problem, 'created_at'):
                continue
            
            created_time = datetime.fromisoformat(problem.created_at.replace('Z', '+00:00'))
            
            if period_type == 'daily':
                period_key = created_time.strftime('%Y-%m-%d')
            elif period_type == 'weekly':
                period_key = created_time.strftime('%Y-%U')
            elif period_type == 'monthly':
                period_key = created_time.strftime('%Y-%m')
            else:
                continue
            
            if period_key not in groups:
                groups[period_key] = []
            groups[period_key].append(problem)
        
        return groups
    
    def _group_by_month(self, problems: List) -> Dict[str, List]:
        """Group problems by month"""
        monthly_groups = {}
        
        for problem in problems:
            if not hasattr(problem, 'created_at'):
                continue
            
            created_time = datetime.fromisoformat(problem.created_at.replace('Z', '+00:00'))
            month_key = created_time.strftime('%Y-%m')
            
            if month_key not in monthly_groups:
                monthly_groups[month_key] = []
            monthly_groups[month_key].append(problem)
        
        return monthly_groups
    
    def _group_by_day_of_week(self, problems: List) -> Dict[str, List]:
        """Group problems by day of week"""
        weekly_groups = {}
        
        for problem in problems:
            if not hasattr(problem, 'created_at'):
                continue
            
            created_time = datetime.fromisoformat(problem.created_at.replace('Z', '+00:00'))
            day_key = created_time.strftime('%A')  # Full day name
            
            if day_key not in weekly_groups:
                weekly_groups[day_key] = []
            weekly_groups[day_key].append(problem)
        
        return weekly_groups
    
    def _calculate_frequency_changes(self, time_groups: Dict[str, List]) -> Dict[str, float]:
        """Calculate frequency changes over time"""
        counts = [len(problems) for problems in time_groups.values()]
        
        if len(counts) < 2:
            return {'variance': 0, 'trend': 'stable', 'significance': 0}
        
        # Calculate variance and trend
        variance = np.var(counts)
        
        # Simple trend detection
        if len(counts) >= 3:
            recent_avg = np.mean(counts[-3:])
            earlier_avg = np.mean(counts[:-3]) if len(counts) > 3 else np.mean(counts[:len(counts)//2])
            
            if recent_avg > earlier_avg * 1.2:
                trend = 'increasing'
            elif recent_avg < earlier_avg * 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'variance': variance,
            'trend': trend,
            'significance': min(abs(variance) / (np.mean(counts) + 1e-6), 1.0)
        }
    
    def _calculate_moving_average(self, time_sorted_problems: List, window_size: int) -> List[float]:
        """Calculate moving average of problem counts"""
        if len(time_sorted_problems) < window_size:
            return []
        
        daily_counts = self._group_by_time_period(time_sorted_problems, 'daily')
        dates = sorted(daily_counts.keys())
        counts = [daily_counts[date] for date in dates]
        
        moving_averages = []
        for i in range(window_size - 1, len(counts)):
            window_counts = counts[i - window_size + 1:i + 1]
            moving_avg = np.mean(window_counts)
            moving_averages.append(moving_avg)
        
        return moving_averages
    
    def _calculate_trend_direction(self, moving_average: List[float]) -> str:
        """Calculate trend direction from moving average"""
        if len(moving_average) < 3:
            return 'insufficient_data'
        
        # Compare recent values to earlier values
        recent_avg = np.mean(moving_average[-3:])
        earlier_avg = np.mean(moving_average[:-3])
        
        if recent_avg > earlier_avg * 1.1:
            return 'increasing'
        elif recent_avg < earlier_avg * 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_trend_strength(self, moving_average: List[float]) -> float:
        """Calculate trend strength"""
        if len(moving_average) < 2:
            return 0.0
        
        # Calculate slope as trend strength
        x = np.arange(len(moving_average))
        if len(x) > 1:
            slope, _ = np.polyfit(x, moving_average, 1)
            return min(abs(slope), 1.0)
        
        return 0.0
    
    def _calculate_trend_significance(self, moving_average: List[float], window_size: int) -> float:
        """Calculate significance of trend"""
        if len(moving_average) < 3:
            return 0.0
        
        # Significance based on consistency and magnitude
        variance = np.var(moving_average)
        mean_val = np.mean(moving_average)
        
        # Higher variance reduces significance, higher mean increases it
        significance = (mean_val / (variance + 1e-6)) * window_size / 100
        return min(significance, 1.0)
    
    def _classify_burst_level(self, burst_intensity: float) -> str:
        """Classify burst level"""
        if burst_intensity > 4:
            return 'extreme'
        elif burst_intensity > 3:
            return 'high'
        elif burst_intensity > 2:
            return 'moderate'
        elif burst_intensity > 1:
            return 'low'
        else:
            return 'minimal'
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for temporal patterns"""
        return {
            'status': 'working',
            'analysis_methods': [
                'frequency_patterns',
                'seasonal_patterns',
                'trend_patterns',
                'burst_patterns'
            ],
            'nlp_utils_available': hasattr(nlp_utils, 'extract_keywords')
        }
