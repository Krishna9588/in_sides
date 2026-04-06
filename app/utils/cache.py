"""
Caching utility for Founder Intelligence System
"""
import json
import pickle
import hashlib
from typing import Any, Optional, Union
from datetime import datetime, timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..config.settings import settings


class CacheManager:
    """Cache manager with Redis fallback to memory"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis client if available"""
        if REDIS_AVAILABLE and settings.REDIS_URL:
            try:
                self.redis_client = redis.from_url(settings.REDIS_URL)
                # Test connection
                self.redis_client.ping()
                print("Connected to Redis cache")
            except Exception as e:
                print(f"Redis connection failed, using memory cache: {e}")
                self.redis_client = None
        else:
            print("Redis not available, using memory cache")
    
    def _get_key(self, key: str) -> str:
        """Get full cache key"""
        return f"founder_intelligence:{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        return pickle.dumps(value)
    
    def _deserialize(self, value: bytes) -> Any:
        """Deserialize value from storage"""
        return pickle.loads(value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        full_key = self._get_key(key)
        
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(full_key)
                if cached_data:
                    return self._deserialize(cached_data)
            except Exception as e:
                print(f"Redis get error: {e}")
        
        # Fallback to memory cache
        if full_key in self.memory_cache:
            cache_entry = self.memory_cache[full_key]
            if cache_entry['expires_at'] > datetime.now():
                return cache_entry['value']
            else:
                del self.memory_cache[full_key]
        
        return default
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        full_key = self._get_key(key)
        ttl = ttl or settings.REDIS_CACHE_TTL
        
        if self.redis_client:
            try:
                serialized_value = self._serialize(value)
                return self.redis_client.setex(full_key, ttl, serialized_value)
            except Exception as e:
                print(f"Redis set error: {e}")
        
        # Fallback to memory cache
        expires_at = datetime.now() + timedelta(seconds=ttl)
        self.memory_cache[full_key] = {
            'value': value,
            'expires_at': expires_at
        }
        return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        full_key = self._get_key(key)
        
        if self.redis_client:
            try:
                return bool(self.redis_client.delete(full_key))
            except Exception as e:
                print(f"Redis delete error: {e}")
        
        # Fallback to memory cache
        if full_key in self.memory_cache:
            del self.memory_cache[full_key]
            return True
        
        return False
    
    def clear(self) -> bool:
        """Clear all cache"""
        if self.redis_client:
            try:
                # Clear all keys with our prefix
                pattern = self._get_key("*")
                keys = self.redis_client.keys(pattern)
                if keys:
                    return bool(self.redis_client.delete(*keys))
            except Exception as e:
                print(f"Redis clear error: {e}")
        
        # Fallback to memory cache
        self.memory_cache.clear()
        return True
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        full_key = self._get_key(key)
        
        if self.redis_client:
            try:
                return bool(self.redis_client.exists(full_key))
            except Exception as e:
                print(f"Redis exists error: {e}")
        
        # Fallback to memory cache
        if full_key in self.memory_cache:
            cache_entry = self.memory_cache[full_key]
            return cache_entry['expires_at'] > datetime.now()
        
        return False
    
    def get_ttl(self, key: str) -> int:
        """Get TTL for key"""
        full_key = self._get_key(key)
        
        if self.redis_client:
            try:
                return self.redis_client.ttl(full_key)
            except Exception as e:
                print(f"Redis TTL error: {e}")
        
        # Fallback to memory cache
        if full_key in self.memory_cache:
            cache_entry = self.memory_cache[full_key]
            remaining_time = cache_entry['expires_at'] - datetime.now()
            return max(0, int(remaining_time.total_seconds()))
        
        return -1


class CacheDecorator:
    """Decorator for caching function results"""
    
    def __init__(self, key_prefix: str = "", ttl: int = None):
        self.key_prefix = key_prefix
        self.ttl = ttl
    
    def __call__(self, func):
        """Decorator implementation"""
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key_data = f"{self.key_prefix}:{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
            
            # Try to get from cache
            cache_manager = CacheManager()
            cached_result = cache_manager.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, self.ttl)
            
            return result
        
        return wrapper


# Global cache manager instance
cache_manager = CacheManager()


def cached(key_prefix: str = "", ttl: int = None):
    """Decorator for caching function results"""
    return CacheDecorator(key_prefix, ttl)


# Cache key constants
class CacheKeys:
    """Cache key constants"""
    
    SIGNALS_BY_SOURCE = "signals_by_source"
    RECENT_SIGNALS = "recent_signals"
    PROBLEMS_BY_CATEGORY = "problems_by_category"
    RECENT_PROBLEMS = "recent_problems"
    INSIGHTS_BY_CATEGORY = "insights_by_category"
    RECENT_INSIGHTS = "recent_insights"
    RECENT_BRIEFS = "recent_briefs"
    QUERY_RESULT = "query_result"
    EMBEDDING = "embedding"
    MODEL_PREDICTION = "model_prediction"
