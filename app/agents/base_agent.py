"""
Base agent class for Founder Intelligence System
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..utils.logger import LoggerMixin
from ..utils.cache import cache_manager, CacheKeys


class BaseAgent(LoggerMixin, ABC):
    """Base class for all AI agents"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.status = "idle"
        self.last_run = None
        self.error_count = 0
        self.max_errors = 3
    
    @abstractmethod
    async def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Run the agent's main logic"""
        pass
    
    @abstractmethod
    async def validate_input(self, *args, **kwargs) -> bool:
        """Validate input parameters"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'last_run': self.last_run,
            'error_count': self.error_count,
            'health': 'healthy' if self.error_count < self.max_errors else 'unhealthy'
        }
    
    def set_status(self, status: str):
        """Set agent status"""
        self.status = status
        self.log_info(f"Agent {self.agent_id} status changed to: {status}")
    
    def increment_error(self):
        """Increment error count"""
        self.error_count += 1
        self.log_error(f"Agent {self.agent_id} error count: {self.error_count}")
    
    def reset_errors(self):
        """Reset error count"""
        self.error_count = 0
        self.log_info(f"Agent {self.agent_id} error count reset")
    
    async def run_with_error_handling(self, *args, **kwargs) -> Dict[str, Any]:
        """Run agent with error handling"""
        try:
            self.set_status("running")
            self.last_run = datetime.now().isoformat()
            
            # Validate input
            if not await self.validate_input(*args, **kwargs):
                return {
                    'status': 'error',
                    'error': 'Invalid input parameters',
                    'agent_id': self.agent_id
                }
            
            # Run agent logic
            result = await self.run(*args, **kwargs)
            
            # Add metadata
            result['agent_id'] = self.agent_id
            result['run_time'] = self.last_run
            
            self.set_status("completed")
            self.reset_errors()
            
            return result
            
        except Exception as e:
            self.increment_error()
            self.set_status("error")
            
            error_result = {
                'status': 'error',
                'error': str(e),
                'agent_id': self.agent_id,
                'run_time': datetime.now().isoformat()
            }
            
            self.log_error(f"Agent {self.agent_id} failed: {e}")
            return error_result
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key for this agent"""
        key_data = f"{self.agent_id}:{str(args)}:{str(sorted(kwargs.items()))}"
        return f"agent_result:{hash(key_data)}"
    
    def get_cached_result(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        cache_key = self.get_cache_key(*args, **kwargs)
        return cache_manager.get(cache_key)
    
    def set_cached_result(self, result: Dict[str, Any], ttl: int = 3600, *args, **kwargs):
        """Cache result"""
        cache_key = self.get_cache_key(*args, **kwargs)
        cache_manager.set(cache_key, result, ttl)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        status = self.get_status()
        
        # Add agent-specific health checks
        health_details = await self._perform_health_check()
        status.update(health_details)
        
        return status
    
    @abstractmethod
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform agent-specific health checks"""
        pass


class AgentMetrics:
    """Agent metrics tracking"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.run_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_run_time = 0
        self.avg_run_time = 0
    
    def record_run(self, run_time: float, success: bool):
        """Record a run"""
        self.run_count += 1
        self.total_run_time += run_time
        self.avg_run_time = self.total_run_time / self.run_count
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            'agent_id': self.agent_id,
            'run_count': self.run_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / self.run_count if self.run_count > 0 else 0,
            'total_run_time': self.total_run_time,
            'avg_run_time': self.avg_run_time
        }


class AgentConfig:
    """Agent configuration"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.enabled = config.get('enabled', True)
        self.schedule = config.get('schedule', None)
        self.max_retries = config.get('max_retries', 3)
        self.timeout = config.get('timeout', 300)
        self.cache_ttl = config.get('cache_ttl', 3600)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def is_enabled(self) -> bool:
        """Check if agent is enabled"""
        return self.enabled
    
    def should_run(self) -> bool:
        """Check if agent should run based on schedule"""
        if not self.enabled:
            return False
        
        if not self.schedule:
            return True
        
        # TODO: Implement schedule checking
        return True
