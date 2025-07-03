# Performance Optimization Framework

## Executive Summary

This document outlines the implementation of a comprehensive performance optimization framework for the multi-agent RAG system. The framework includes performance profiling, caching strategies, database optimization, memory management, and real-time performance monitoring integration.

## Development Objectives

1. **Performance Profiling System**: Implement comprehensive profiling tools for identifying bottlenecks
2. **Caching Framework**: Deploy multi-level caching strategies for improved response times
3. **Database Optimization**: Optimize database queries, indexing, and connection pooling
4. **Memory Management**: Implement efficient memory usage and garbage collection strategies
5. **Load Balancing**: Add intelligent load distribution for agent workloads
6. **Performance Monitoring Integration**: Connect with the monitoring system for real-time optimization

## Backend Implementation

### Performance Profiler Service

**File**: `backend/core/performance/profiler_service.py`

```python
import time
import psutil
import asyncio
import functools
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import cProfile
import pstats
import io
from memory_profiler import profile as memory_profile

@dataclass
class ProfileMetric:
    """Performance metric data"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceProfile:
    """Complete performance profile"""
    function_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    call_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stack_trace: Optional[str] = None
    bottlenecks: List[str] = field(default_factory=list)

@dataclass
class SystemResourceUsage:
    """System resource usage snapshot"""
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

class PerformanceProfiler:
    """Advanced performance profiling service"""
    
    def __init__(self):
        self.profiles: Dict[str, List[PerformanceProfile]] = {}
        self.metrics: Dict[str, List[ProfileMetric]] = {}
        self.resource_history: List[SystemResourceUsage] = []
        self.profiling_enabled = True
        self.profiling_threshold = 0.1  # seconds
        self.max_history_size = 1000
        
    def enable_profiling(self, enabled: bool = True):
        """Enable or disable profiling"""
        self.profiling_enabled = enabled
        
    def set_threshold(self, threshold: float):
        """Set minimum execution time threshold for profiling"""
        self.profiling_threshold = threshold
        
    def profile_function(self, func_name: str = None):
        """Decorator for profiling function performance"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.profiling_enabled:
                    return await func(*args, **kwargs)
                    
                name = func_name or f"{func.__module__}.{func.__name__}"
                
                # Start profiling
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                start_cpu = psutil.Process().cpu_percent()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Calculate metrics
                    execution_time = time.time() - start_time
                    end_memory = psutil.Process().memory_info().rss
                    memory_usage = end_memory - start_memory
                    cpu_usage = psutil.Process().cpu_percent() - start_cpu
                    
                    # Record profile if above threshold
                    if execution_time >= self.profiling_threshold:
                        profile = PerformanceProfile(
                            function_name=name,
                            execution_time=execution_time,
                            memory_usage=memory_usage,
                            cpu_usage=cpu_usage,
                            call_count=1
                        )
                        
                        self._record_profile(profile)
                        
                    return result
                    
                except Exception as e:
                    # Record failed execution
                    execution_time = time.time() - start_time
                    profile = PerformanceProfile(
                        function_name=name,
                        execution_time=execution_time,
                        memory_usage=0,
                        cpu_usage=0,
                        call_count=1,
                        bottlenecks=[f"Exception: {str(e)}"]
                    )
                    self._record_profile(profile)
                    raise
                    
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.profiling_enabled:
                    return func(*args, **kwargs)
                    
                name = func_name or f"{func.__module__}.{func.__name__}"
                
                # Start profiling
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                start_cpu = psutil.Process().cpu_percent()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Calculate metrics
                    execution_time = time.time() - start_time
                    end_memory = psutil.Process().memory_info().rss
                    memory_usage = end_memory - start_memory
                    cpu_usage = psutil.Process().cpu_percent() - start_cpu
                    
                    # Record profile if above threshold
                    if execution_time >= self.profiling_threshold:
                        profile = PerformanceProfile(
                            function_name=name,
                            execution_time=execution_time,
                            memory_usage=memory_usage,
                            cpu_usage=cpu_usage,
                            call_count=1
                        )
                        
                        self._record_profile(profile)
                        
                    return result
                    
                except Exception as e:
                    # Record failed execution
                    execution_time = time.time() - start_time
                    profile = PerformanceProfile(
                        function_name=name,
                        execution_time=execution_time,
                        memory_usage=0,
                        cpu_usage=0,
                        call_count=1,
                        bottlenecks=[f"Exception: {str(e)}"]
                    )
                    self._record_profile(profile)
                    raise
                    
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
        
    @asynccontextmanager
    async def profile_block(self, block_name: str):
        """Context manager for profiling code blocks"""
        if not self.profiling_enabled:
            yield
            return
            
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_cpu = psutil.Process().cpu_percent()
        
        try:
            yield
            
        finally:
            execution_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss
            memory_usage = end_memory - start_memory
            cpu_usage = psutil.Process().cpu_percent() - start_cpu
            
            if execution_time >= self.profiling_threshold:
                profile = PerformanceProfile(
                    function_name=block_name,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    cpu_usage=cpu_usage,
                    call_count=1
                )
                
                self._record_profile(profile)
                
    def detailed_profile(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Perform detailed profiling with cProfile"""
        pr = cProfile.Profile()
        
        # Profile the function
        pr.enable()
        try:
            if asyncio.iscoroutinefunction(func):
                result = asyncio.run(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)
        finally:
            pr.disable()
            
        # Generate stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats()
        
        return {
            'result': result,
            'profile_stats': s.getvalue(),
            'total_calls': ps.total_calls,
            'total_time': ps.total_tt
        }
        
    def record_metric(self, name: str, value: float, unit: str, metadata: Dict[str, Any] = None):
        """Record a custom performance metric"""
        metric = ProfileMetric(
            name=name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
            
        self.metrics[name].append(metric)
        
        # Limit history size
        if len(self.metrics[name]) > self.max_history_size:
            self.metrics[name] = self.metrics[name][-self.max_history_size:]
            
    def capture_system_resources(self) -> SystemResourceUsage:
        """Capture current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        usage = SystemResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available,
            disk_io_read=disk_io.read_bytes if disk_io else 0,
            disk_io_write=disk_io.write_bytes if disk_io else 0,
            network_io_sent=network_io.bytes_sent if network_io else 0,
            network_io_recv=network_io.bytes_recv if network_io else 0
        )
        
        self.resource_history.append(usage)
        
        # Limit history size
        if len(self.resource_history) > self.max_history_size:
            self.resource_history = self.resource_history[-self.max_history_size:]
            
        return usage
        
    def _record_profile(self, profile: PerformanceProfile):
        """Record a performance profile"""
        if profile.function_name not in self.profiles:
            self.profiles[profile.function_name] = []
            
        self.profiles[profile.function_name].append(profile)
        
        # Limit history size
        if len(self.profiles[profile.function_name]) > self.max_history_size:
            self.profiles[profile.function_name] = self.profiles[profile.function_name][-self.max_history_size:]
            
    def get_function_stats(self, function_name: str) -> Dict[str, Any]:
        """Get statistics for a specific function"""
        if function_name not in self.profiles:
            return {}
            
        profiles = self.profiles[function_name]
        
        if not profiles:
            return {}
            
        execution_times = [p.execution_time for p in profiles]
        memory_usages = [p.memory_usage for p in profiles]
        cpu_usages = [p.cpu_usage for p in profiles]
        
        return {
            'function_name': function_name,
            'call_count': len(profiles),
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'avg_memory_usage': sum(memory_usages) / len(memory_usages),
            'max_memory_usage': max(memory_usages),
            'avg_cpu_usage': sum(cpu_usages) / len(cpu_usages),
            'max_cpu_usage': max(cpu_usages),
            'recent_bottlenecks': [p.bottlenecks for p in profiles[-10:] if p.bottlenecks]
        }
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        total_functions = len(self.profiles)
        total_calls = sum(len(profiles) for profiles in self.profiles.values())
        
        # Find slowest functions
        slowest_functions = []
        for func_name in self.profiles:
            stats = self.get_function_stats(func_name)
            if stats:
                slowest_functions.append({
                    'name': func_name,
                    'avg_time': stats['avg_execution_time'],
                    'max_time': stats['max_execution_time'],
                    'call_count': stats['call_count']
                })
                
        slowest_functions.sort(key=lambda x: x['avg_time'], reverse=True)
        
        # Recent resource usage
        recent_resources = self.resource_history[-10:] if self.resource_history else []
        
        return {
            'total_functions_profiled': total_functions,
            'total_function_calls': total_calls,
            'slowest_functions': slowest_functions[:10],
            'recent_resource_usage': [vars(r) for r in recent_resources],
            'profiling_enabled': self.profiling_enabled,
            'profiling_threshold': self.profiling_threshold
        }
        
    def identify_bottlenecks(self, threshold_percentile: float = 95) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        for func_name, profiles in self.profiles.items():
            if not profiles:
                continue
                
            execution_times = [p.execution_time for p in profiles]
            execution_times.sort()
            
            # Calculate threshold
            threshold_index = int(len(execution_times) * threshold_percentile / 100)
            if threshold_index >= len(execution_times):
                threshold_index = len(execution_times) - 1
                
            threshold_time = execution_times[threshold_index]
            
            # Find calls above threshold
            slow_calls = [p for p in profiles if p.execution_time >= threshold_time]
            
            if slow_calls:
                bottlenecks.append({
                    'function_name': func_name,
                    'threshold_time': threshold_time,
                    'slow_call_count': len(slow_calls),
                    'avg_slow_time': sum(p.execution_time for p in slow_calls) / len(slow_calls),
                    'max_slow_time': max(p.execution_time for p in slow_calls),
                    'recommendations': self._generate_recommendations(func_name, slow_calls)
                })
                
        # Sort by impact (slow call count * avg time)
        bottlenecks.sort(key=lambda x: x['slow_call_count'] * x['avg_slow_time'], reverse=True)
        
        return bottlenecks
        
    def _generate_recommendations(self, func_name: str, slow_calls: List[PerformanceProfile]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        avg_memory = sum(p.memory_usage for p in slow_calls) / len(slow_calls)
        avg_cpu = sum(p.cpu_usage for p in slow_calls) / len(slow_calls)
        
        if avg_memory > 100 * 1024 * 1024:  # 100MB
            recommendations.append("Consider memory optimization - high memory usage detected")
            
        if avg_cpu > 80:
            recommendations.append("Consider CPU optimization - high CPU usage detected")
            
        if len(slow_calls) > 10:
            recommendations.append("Consider caching - function called frequently")
            
        if any('database' in func_name.lower() or 'query' in func_name.lower() for p in slow_calls):
            recommendations.append("Consider database query optimization")
            
        if any('api' in func_name.lower() or 'request' in func_name.lower() for p in slow_calls):
            recommendations.append("Consider API response caching or connection pooling")
            
        return recommendations
        
    def clear_profiles(self, function_name: str = None):
        """Clear profiling data"""
        if function_name:
            if function_name in self.profiles:
                del self.profiles[function_name]
        else:
            self.profiles.clear()
            self.metrics.clear()
            self.resource_history.clear()

# Global profiler instance
profiler = PerformanceProfiler()
```

### Caching Framework

**File**: `backend/core/performance/cache_service.py`

```python
import asyncio
import json
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import redis
from functools import wraps
import weakref

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)

@dataclass
class CacheStats:
    """Cache statistics"""
    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    hit_rate: float
    average_access_time: float
    memory_usage_percent: float

class CacheBackend(ABC):
    """Abstract cache backend"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
        
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass
        
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
        
    @abstractmethod
    async def clear(self) -> bool:
        pass
        
    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass

class MemoryCache(CacheBackend):
    """In-memory cache backend"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.stats = CacheStats(
            total_entries=0,
            total_size_bytes=0,
            hit_count=0,
            miss_count=0,
            eviction_count=0,
            hit_rate=0.0,
            average_access_time=0.0,
            memory_usage_percent=0.0
        )
        
    async def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            self.stats.miss_count += 1
            self._update_hit_rate()
            return None
            
        entry = self.cache[key]
        
        # Check expiration
        if entry.expires_at and datetime.utcnow() > entry.expires_at:
            await self.delete(key)
            self.stats.miss_count += 1
            self._update_hit_rate()
            return None
            
        # Update access info
        entry.access_count += 1
        entry.last_accessed = datetime.utcnow()
        
        self.stats.hit_count += 1
        self._update_hit_rate()
        
        return entry.value
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        # Calculate expiration
        expires_at = None
        if ttl or self.default_ttl:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl or self.default_ttl)
            
        # Calculate size
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = len(str(value).encode('utf-8'))
            
        # Check if we need to evict
        if len(self.cache) >= self.max_size and key not in self.cache:
            await self._evict_lru()
            
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
            size_bytes=size_bytes
        )
        
        # Update stats
        if key in self.cache:
            self.stats.total_size_bytes -= self.cache[key].size_bytes
        else:
            self.stats.total_entries += 1
            
        self.stats.total_size_bytes += size_bytes
        
        self.cache[key] = entry
        return True
        
    async def delete(self, key: str) -> bool:
        if key in self.cache:
            entry = self.cache[key]
            self.stats.total_entries -= 1
            self.stats.total_size_bytes -= entry.size_bytes
            del self.cache[key]
            return True
        return False
        
    async def clear(self) -> bool:
        self.cache.clear()
        self.stats.total_entries = 0
        self.stats.total_size_bytes = 0
        return True
        
    async def exists(self, key: str) -> bool:
        if key not in self.cache:
            return False
            
        entry = self.cache[key]
        if entry.expires_at and datetime.utcnow() > entry.expires_at:
            await self.delete(key)
            return False
            
        return True
        
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
            
        # Find LRU entry
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
        await self.delete(lru_key)
        self.stats.eviction_count += 1
        
    def _update_hit_rate(self):
        """Update hit rate statistics"""
        total_requests = self.stats.hit_count + self.stats.miss_count
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hit_count / total_requests

class RedisCache(CacheBackend):
    """Redis cache backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.redis_client = None
        
    async def _get_client(self):
        """Get Redis client"""
        if not self.redis_client:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
        return self.redis_client
        
    async def get(self, key: str) -> Optional[Any]:
        try:
            client = await self._get_client()
            data = client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception:
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            client = await self._get_client()
            data = pickle.dumps(value)
            return client.setex(key, ttl or self.default_ttl, data)
        except Exception:
            return False
            
    async def delete(self, key: str) -> bool:
        try:
            client = await self._get_client()
            return bool(client.delete(key))
        except Exception:
            return False
            
    async def clear(self) -> bool:
        try:
            client = await self._get_client()
            return client.flushdb()
        except Exception:
            return False
            
    async def exists(self, key: str) -> bool:
        try:
            client = await self._get_client()
            return bool(client.exists(key))
        except Exception:
            return False

class CacheService:
    """Multi-level caching service"""
    
    def __init__(self, 
                 memory_cache: Optional[MemoryCache] = None,
                 redis_cache: Optional[RedisCache] = None,
                 enable_compression: bool = True):
        self.memory_cache = memory_cache or MemoryCache()
        self.redis_cache = redis_cache
        self.enable_compression = enable_compression
        self.cache_hierarchy = [self.memory_cache]
        
        if self.redis_cache:
            self.cache_hierarchy.append(self.redis_cache)
            
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create a deterministic key from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"{prefix}:{key_hash}"
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy"""
        for i, cache in enumerate(self.cache_hierarchy):
            value = await cache.get(key)
            if value is not None:
                # Populate higher-level caches
                for j in range(i):
                    await self.cache_hierarchy[j].set(key, value)
                return value
                
        return None
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in all cache levels"""
        results = []
        for cache in self.cache_hierarchy:
            result = await cache.set(key, value, ttl)
            results.append(result)
            
        return any(results)
        
    async def delete(self, key: str) -> bool:
        """Delete from all cache levels"""
        results = []
        for cache in self.cache_hierarchy:
            result = await cache.delete(key)
            results.append(result)
            
        return any(results)
        
    async def clear(self, pattern: Optional[str] = None) -> bool:
        """Clear cache"""
        results = []
        for cache in self.cache_hierarchy:
            result = await cache.clear()
            results.append(result)
            
        return any(results)
        
    def cached(self, ttl: Optional[int] = None, key_prefix: str = "cached"):
        """Decorator for caching function results"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(f"{key_prefix}:{func.__name__}", *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                    
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl)
                
                return result
                
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, we need to handle caching differently
                cache_key = self._generate_key(f"{key_prefix}:{func.__name__}", *args, **kwargs)
                
                # Try to get from cache (sync)
                try:
                    loop = asyncio.get_event_loop()
                    cached_result = loop.run_until_complete(self.get(cache_key))
                    if cached_result is not None:
                        return cached_result
                except:
                    pass
                    
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result (sync)
                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(self.set(cache_key, result, ttl))
                except:
                    pass
                    
                return result
                
            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
        
    async def invalidate_by_tags(self, tags: List[str]):
        """Invalidate cache entries by tags"""
        # This is a simplified implementation
        # In a real system, you'd need to track tags separately
        pass
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {}
        
        if isinstance(self.memory_cache, MemoryCache):
            stats['memory_cache'] = vars(self.memory_cache.stats)
            
        # Add Redis stats if available
        if self.redis_cache:
            try:
                client = await self.redis_cache._get_client()
                info = client.info()
                stats['redis_cache'] = {
                    'used_memory': info.get('used_memory', 0),
                    'connected_clients': info.get('connected_clients', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0)
                }
            except:
                stats['redis_cache'] = {'error': 'Unable to get Redis stats'}
                
        return stats

# Global cache service instance
cache_service = CacheService()
```

### Database Optimization Service

**File**: `backend/core/performance/database_optimizer.py`

```python
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import psutil

@dataclass
class QueryPerformance:
    """Query performance metrics"""
    query_hash: str
    query_text: str
    execution_time: float
    rows_affected: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    explain_plan: Optional[str] = None
    index_usage: List[str] = field(default_factory=list)
    table_scans: int = 0
    memory_usage: int = 0

@dataclass
class IndexRecommendation:
    """Database index recommendation"""
    table_name: str
    columns: List[str]
    index_type: str
    estimated_benefit: float
    query_patterns: List[str]
    creation_sql: str
    impact_score: float

@dataclass
class ConnectionPoolStats:
    """Connection pool statistics"""
    pool_size: int
    checked_out: int
    overflow: int
    checked_in: int
    total_connections: int
    avg_connection_time: float
    peak_connections: int
    connection_errors: int

class DatabaseOptimizer:
    """Database performance optimization service"""
    
    def __init__(self, database_url: str, pool_size: int = 20, max_overflow: int = 30):
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        
        # Create optimized engine
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=False
        )
        
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.query_performance: Dict[str, List[QueryPerformance]] = {}
        self.slow_queries: List[QueryPerformance] = []
        self.index_recommendations: List[IndexRecommendation] = []
        self.connection_stats = ConnectionPoolStats(
            pool_size=pool_size,
            checked_out=0,
            overflow=0,
            checked_in=0,
            total_connections=0,
            avg_connection_time=0.0,
            peak_connections=0,
            connection_errors=0
        )
        
    async def analyze_query_performance(self, query: str, params: Dict[str, Any] = None) -> QueryPerformance:
        """Analyze query performance"""
        query_hash = self._hash_query(query)
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            with self.engine.connect() as conn:
                # Get explain plan
                explain_query = f"EXPLAIN (ANALYZE, BUFFERS) {query}"
                explain_result = conn.execute(text(explain_query), params or {})
                explain_plan = "\n".join([str(row[0]) for row in explain_result])
                
                # Execute actual query
                result = conn.execute(text(query), params or {})
                rows_affected = result.rowcount if hasattr(result, 'rowcount') else 0
                
                execution_time = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss
                memory_usage = end_memory - start_memory
                
                # Analyze explain plan
                index_usage = self._extract_index_usage(explain_plan)
                table_scans = self._count_table_scans(explain_plan)
                
                performance = QueryPerformance(
                    query_hash=query_hash,
                    query_text=query,
                    execution_time=execution_time,
                    rows_affected=rows_affected,
                    explain_plan=explain_plan,
                    index_usage=index_usage,
                    table_scans=table_scans,
                    memory_usage=memory_usage
                )
                
                # Store performance data
                if query_hash not in self.query_performance:
                    self.query_performance[query_hash] = []
                    
                self.query_performance[query_hash].append(performance)
                
                # Track slow queries
                if execution_time > 1.0:  # Queries taking more than 1 second
                    self.slow_queries.append(performance)
                    
                return performance
                
        except Exception as e:
            # Return error performance record
            return QueryPerformance(
                query_hash=query_hash,
                query_text=query,
                execution_time=time.time() - start_time,
                rows_affected=0,
                explain_plan=f"Error: {str(e)}"
            )
            
    def _hash_query(self, query: str) -> str:
        """Generate hash for query normalization"""
        import hashlib
        # Normalize query (remove extra whitespace, convert to lowercase)
        normalized = ' '.join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
        
    def _extract_index_usage(self, explain_plan: str) -> List[str]:
        """Extract index usage from explain plan"""
        indexes = []
        lines = explain_plan.split('\n')
        
        for line in lines:
            if 'Index Scan' in line or 'Index Only Scan' in line:
                # Extract index name
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'using' in part.lower() and i + 1 < len(parts):
                        indexes.append(parts[i + 1])
                        
        return indexes
        
    def _count_table_scans(self, explain_plan: str) -> int:
        """Count table scans in explain plan"""
        return explain_plan.count('Seq Scan')
        
    async def generate_index_recommendations(self) -> List[IndexRecommendation]:
        """Generate index recommendations based on query patterns"""
        recommendations = []
        
        # Analyze slow queries for index opportunities
        for performance in self.slow_queries:
            if performance.table_scans > 0:
                # Suggest indexes for queries with table scans
                tables = self._extract_tables_from_query(performance.query_text)
                columns = self._extract_where_columns(performance.query_text)
                
                for table in tables:
                    if columns:
                        recommendation = IndexRecommendation(
                            table_name=table,
                            columns=columns,
                            index_type='btree',
                            estimated_benefit=performance.execution_time * 0.7,  # Estimated 70% improvement
                            query_patterns=[performance.query_text],
                            creation_sql=f"CREATE INDEX idx_{table}_{'_'.join(columns)} ON {table} ({', '.join(columns)});",
                            impact_score=performance.execution_time * performance.rows_affected
                        )
                        recommendations.append(recommendation)
                        
        # Remove duplicates and sort by impact
        unique_recommendations = {}
        for rec in recommendations:
            key = f"{rec.table_name}_{','.join(rec.columns)}"
            if key not in unique_recommendations or rec.impact_score > unique_recommendations[key].impact_score:
                unique_recommendations[key] = rec
                
        self.index_recommendations = sorted(unique_recommendations.values(), 
                                          key=lambda x: x.impact_score, reverse=True)
        
        return self.index_recommendations
        
    def _extract_tables_from_query(self, query: str) -> List[str]:
        """Extract table names from SQL query"""
        import re
        # Simple regex to find table names after FROM and JOIN
        pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(pattern, query, re.IGNORECASE)
        return list(set(matches))
        
    def _extract_where_columns(self, query: str) -> List[str]:
        """Extract column names from WHERE clause"""
        import re
        # Simple regex to find column names in WHERE conditions
        pattern = r'WHERE.*?([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>!]'
        matches = re.findall(pattern, query, re.IGNORECASE)
        return list(set(matches))
        
    async def optimize_connection_pool(self) -> Dict[str, Any]:
        """Optimize database connection pool settings"""
        # Get current pool stats
        pool = self.engine.pool
        
        current_stats = {
            'size': pool.size(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'checked_in': pool.checkedin()
        }
        
        # Calculate optimal pool size based on usage patterns
        peak_usage = max(current_stats['checked_out'], self.connection_stats.peak_connections)
        
        recommendations = {
            'current_pool_size': self.pool_size,
            'current_usage': current_stats,
            'peak_usage': peak_usage,
            'recommendations': []
        }
        
        # Recommend pool size adjustments
        if peak_usage > self.pool_size * 0.8:
            recommendations['recommendations'].append({
                'type': 'increase_pool_size',
                'current': self.pool_size,
                'recommended': min(self.pool_size + 10, 50),
                'reason': 'High pool utilization detected'
            })
            
        if current_stats['overflow'] > 0:
            recommendations['recommendations'].append({
                'type': 'increase_max_overflow',
                'current': self.max_overflow,
                'recommended': self.max_overflow + 10,
                'reason': 'Connection overflow detected'
            })
            
        return recommendations
        
    async def vacuum_analyze_tables(self, tables: List[str] = None) -> Dict[str, Any]:
        """Perform VACUUM ANALYZE on specified tables"""
        results = {}
        
        try:
            with self.engine.connect() as conn:
                if not tables:
                    # Get all tables
                    inspector = inspect(self.engine)
                    tables = inspector.get_table_names()
                    
                for table in tables:
                    start_time = time.time()
                    
                    try:
                        # Note: VACUUM cannot be run inside a transaction
                        conn.execute(text(f"VACUUM ANALYZE {table}"))
                        execution_time = time.time() - start_time
                        
                        results[table] = {
                            'status': 'success',
                            'execution_time': execution_time
                        }
                        
                    except Exception as e:
                        results[table] = {
                            'status': 'error',
                            'error': str(e)
                        }
                        
        except Exception as e:
            results['error'] = str(e)
            
        return results
        
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        stats = {
            'connection_pool': vars(self.connection_stats),
            'query_performance': {},
            'slow_queries_count': len(self.slow_queries),
            'index_recommendations_count': len(self.index_recommendations)
        }
        
        # Aggregate query performance stats
        total_queries = sum(len(queries) for queries in self.query_performance.values())
        if total_queries > 0:
            all_execution_times = []
            for queries in self.query_performance.values():
                all_execution_times.extend([q.execution_time for q in queries])
                
            stats['query_performance'] = {
                'total_queries': total_queries,
                'avg_execution_time': sum(all_execution_times) / len(all_execution_times),
                'max_execution_time': max(all_execution_times),
                'min_execution_time': min(all_execution_times),
                'queries_over_1s': len([t for t in all_execution_times if t > 1.0])
            }
            
        return stats
        
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest queries"""
        sorted_queries = sorted(self.slow_queries, 
                              key=lambda x: x.execution_time, reverse=True)
        
        return [{
            'query_text': q.query_text[:200] + '...' if len(q.query_text) > 200 else q.query_text,
            'execution_time': q.execution_time,
            'rows_affected': q.rows_affected,
            'table_scans': q.table_scans,
            'timestamp': q.timestamp.isoformat()
        } for q in sorted_queries[:limit]]

# Global database optimizer instance
database_optimizer = DatabaseOptimizer("postgresql://user:pass@localhost/db")
```

### Memory Management Service

**File**: `backend/core/performance/memory_manager.py`

```python
import gc
import psutil
import asyncio
import weakref
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import tracemalloc
from functools import wraps

@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_memory: int = 0
    available_memory: int = 0
    used_memory: int = 0
    memory_percent: float = 0.0
    process_memory: int = 0
    gc_collections: Dict[int, int] = field(default_factory=dict)
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class MemoryLeak:
    """Memory leak detection result"""
    object_type: str
    count: int
    size_bytes: int
    growth_rate: float
    first_seen: datetime
    last_seen: datetime = field(default_factory=datetime.utcnow)
    stack_trace: Optional[str] = None

@dataclass
class MemoryPool:
    """Memory pool for object reuse"""
    name: str
    object_factory: Callable
    max_size: int
    current_size: int = 0
    objects: List[Any] = field(default_factory=list)
    hits: int = 0
    misses: int = 0

class MemoryManager:
    """Advanced memory management service"""
    
    def __init__(self, 
                 monitoring_interval: int = 60,
                 gc_threshold: float = 80.0,
                 leak_detection_enabled: bool = True):
        self.monitoring_interval = monitoring_interval
        self.gc_threshold = gc_threshold
        self.leak_detection_enabled = leak_detection_enabled
        
        self.memory_history: List[MemorySnapshot] = []
        self.detected_leaks: List[MemoryLeak] = []
        self.memory_pools: Dict[str, MemoryPool] = {}
        self.object_tracking: Dict[type, List[weakref.ref]] = {}
        
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Enable tracemalloc for detailed memory tracking
        if leak_detection_enabled:
            tracemalloc.start()
            
    async def start_monitoring(self):
        """Start memory monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
    async def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
                
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Take memory snapshot
                snapshot = await self.take_memory_snapshot()
                self.memory_history.append(snapshot)
                
                # Limit history size
                if len(self.memory_history) > 1000:
                    self.memory_history = self.memory_history[-1000:]
                    
                # Check for memory pressure
                if snapshot.memory_percent > self.gc_threshold:
                    await self.force_garbage_collection()
                    
                # Detect memory leaks
                if self.leak_detection_enabled:
                    await self.detect_memory_leaks()
                    
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
                
    async def take_memory_snapshot(self) -> MemorySnapshot:
        """Take a comprehensive memory snapshot"""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss
        
        # Garbage collection stats
        gc_stats = {i: gc.get_count()[i] for i in range(3)}
        
        # Top memory allocations (if tracemalloc is enabled)
        top_allocations = []
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            for stat in top_stats:
                top_allocations.append({
                    'filename': stat.traceback.format()[-1] if stat.traceback else 'unknown',
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                })
                
        return MemorySnapshot(
            total_memory=memory.total,
            available_memory=memory.available,
            used_memory=memory.used,
            memory_percent=memory.percent,
            process_memory=process_memory,
            gc_collections=gc_stats,
            top_allocations=top_allocations
        )
        
    async def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return collected counts"""
        collected = {}
        
        for generation in range(3):
            collected[generation] = gc.collect(generation)
            
        # Also run a full collection
        collected['full'] = gc.collect()
        
        return collected
        
    async def detect_memory_leaks(self) -> List[MemoryLeak]:
        """Detect potential memory leaks"""
        if not tracemalloc.is_tracing():
            return []
            
        current_snapshot = tracemalloc.take_snapshot()
        
        # Compare with previous snapshots to detect growth
        if len(self.memory_history) < 2:
            return []
            
        # Analyze memory growth patterns
        leaks = []
        
        # Get statistics grouped by filename
        stats = current_snapshot.statistics('filename')
        
        for stat in stats[:20]:  # Check top 20 allocations
            # Simple heuristic: if allocation is growing consistently
            if stat.size > 10 * 1024 * 1024:  # More than 10MB
                leak = MemoryLeak(
                    object_type=stat.traceback.format()[-1] if stat.traceback else 'unknown',
                    count=stat.count,
                    size_bytes=stat.size,
                    growth_rate=0.0,  # Would need historical data to calculate
                    first_seen=datetime.utcnow(),
                    stack_trace='\n'.join(stat.traceback.format()) if stat.traceback else None
                )
                leaks.append(leak)
                
        self.detected_leaks.extend(leaks)
        return leaks
        
    def create_memory_pool(self, name: str, object_factory: Callable, max_size: int = 100) -> MemoryPool:
        """Create a memory pool for object reuse"""
        pool = MemoryPool(
            name=name,
            object_factory=object_factory,
            max_size=max_size
        )
        
        self.memory_pools[name] = pool
        return pool
        
    def get_from_pool(self, pool_name: str) -> Optional[Any]:
        """Get object from memory pool"""
        if pool_name not in self.memory_pools:
            return None
            
        pool = self.memory_pools[pool_name]
        
        if pool.objects:
            pool.hits += 1
            return pool.objects.pop()
        else:
            pool.misses += 1
            return pool.object_factory()
            
    def return_to_pool(self, pool_name: str, obj: Any) -> bool:
        """Return object to memory pool"""
        if pool_name not in self.memory_pools:
            return False
            
        pool = self.memory_pools[pool_name]
        
        if len(pool.objects) < pool.max_size:
            # Reset object state if it has a reset method
            if hasattr(obj, 'reset'):
                obj.reset()
                
            pool.objects.append(obj)
            pool.current_size = len(pool.objects)
            return True
            
        return False
        
    def track_object(self, obj: Any):
        """Track object for memory leak detection"""
        obj_type = type(obj)
        
        if obj_type not in self.object_tracking:
            self.object_tracking[obj_type] = []
            
        # Use weak reference to avoid keeping objects alive
        weak_ref = weakref.ref(obj)
        self.object_tracking[obj_type].append(weak_ref)
        
        # Clean up dead references periodically
        if len(self.object_tracking[obj_type]) % 100 == 0:
            self.object_tracking[obj_type] = [
                ref for ref in self.object_tracking[obj_type] if ref() is not None
            ]
            
    def memory_efficient(self, pool_name: str = None):
        """Decorator for memory-efficient function execution"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Take initial memory snapshot
                initial_memory = psutil.Process().memory_info().rss
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Check memory usage after execution
                    final_memory = psutil.Process().memory_info().rss
                    memory_delta = final_memory - initial_memory
                    
                    # If significant memory increase, suggest garbage collection
                    if memory_delta > 50 * 1024 * 1024:  # 50MB
                        await self.force_garbage_collection()
                        
                    return result
                    
                finally:
                    # Clean up any temporary objects
                    gc.collect(0)
                    
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                initial_memory = psutil.Process().memory_info().rss
                
                try:
                    result = func(*args, **kwargs)
                    
                    final_memory = psutil.Process().memory_info().rss
                    memory_delta = final_memory - initial_memory
                    
                    if memory_delta > 50 * 1024 * 1024:  # 50MB
                        gc.collect()
                        
                    return result
                    
                finally:
                    gc.collect(0)
                    
            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        current_memory = psutil.virtual_memory()
        process_memory = psutil.Process().memory_info()
        
        stats = {
            'system_memory': {
                'total_gb': current_memory.total / (1024**3),
                'available_gb': current_memory.available / (1024**3),
                'used_percent': current_memory.percent
            },
            'process_memory': {
                'rss_mb': process_memory.rss / (1024**2),
                'vms_mb': process_memory.vms / (1024**2)
            },
            'garbage_collection': {
                'counts': gc.get_count(),
                'thresholds': gc.get_threshold()
            },
            'memory_pools': {},
            'detected_leaks': len(self.detected_leaks),
            'tracking_objects': sum(len(refs) for refs in self.object_tracking.values())
        }
        
        # Add pool statistics
        for name, pool in self.memory_pools.items():
            hit_rate = pool.hits / (pool.hits + pool.misses) if (pool.hits + pool.misses) > 0 else 0
            stats['memory_pools'][name] = {
                'current_size': pool.current_size,
                'max_size': pool.max_size,
                'hits': pool.hits,
                'misses': pool.misses,
                'hit_rate': hit_rate
            }
            
        return stats
        
    def get_memory_recommendations(self) -> List[Dict[str, Any]]:
        """Get memory optimization recommendations"""
        recommendations = []
        
        current_memory = psutil.virtual_memory()
        
        # High memory usage
        if current_memory.percent > 85:
            recommendations.append({
                'type': 'high_memory_usage',
                'severity': 'high',
                'message': f'System memory usage is {current_memory.percent:.1f}%',
                'action': 'Consider increasing system memory or optimizing memory usage'
            })
            
        # Memory leaks
        if len(self.detected_leaks) > 0:
            recommendations.append({
                'type': 'memory_leaks',
                'severity': 'high',
                'message': f'{len(self.detected_leaks)} potential memory leaks detected',
                'action': 'Review code for objects not being properly released'
            })
            
        # Pool efficiency
        for name, pool in self.memory_pools.items():
            hit_rate = pool.hits / (pool.hits + pool.misses) if (pool.hits + pool.misses) > 0 else 0
            if hit_rate < 0.5:
                recommendations.append({
                    'type': 'low_pool_efficiency',
                    'severity': 'medium',
                    'message': f'Memory pool "{name}" has low hit rate ({hit_rate:.1%})',
                    'action': 'Consider increasing pool size or reviewing usage patterns'
                })
                
        return recommendations

# Global memory manager instance
memory_manager = MemoryManager()
```

### Load Balancing Service

**File**: `backend/core/performance/load_balancer.py`

```python
import asyncio
import time
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import psutil
from collections import defaultdict

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"

class ServerStatus(Enum):
    """Server status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

@dataclass
class ServerNode:
    """Server node configuration"""
    id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    status: ServerStatus = ServerStatus.HEALTHY
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0
    success_count: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    load_score: float = 0.0

@dataclass
class LoadBalancerStats:
    """Load balancer statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    requests_per_second: float = 0.0
    active_connections: int = 0
    server_distribution: Dict[str, int] = field(default_factory=dict)

class LoadBalancer:
    """Advanced load balancing service"""
    
    def __init__(self, 
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
                 health_check_interval: int = 30,
                 max_retries: int = 3):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.max_retries = max_retries
        
        self.servers: Dict[str, ServerNode] = {}
        self.current_index = 0
        self.stats = LoadBalancerStats()
        self.request_history: List[Dict[str, Any]] = []
        
        self.health_check_task = None
        self.monitoring_active = False
        
    async def add_server(self, server: ServerNode):
        """Add server to load balancer"""
        self.servers[server.id] = server
        self.stats.server_distribution[server.id] = 0
        
    async def remove_server(self, server_id: str):
        """Remove server from load balancer"""
        if server_id in self.servers:
            del self.servers[server_id]
            if server_id in self.stats.server_distribution:
                del self.stats.server_distribution[server_id]
                
    async def start_monitoring(self):
        """Start health monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
                
    async def _health_check_loop(self):
        """Health check monitoring loop"""
        while self.monitoring_active:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                print(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)
                
    async def _perform_health_checks(self):
        """Perform health checks on all servers"""
        tasks = []
        for server in self.servers.values():
            if server.status != ServerStatus.MAINTENANCE:
                tasks.append(self._check_server_health(server))
                
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def _check_server_health(self, server: ServerNode):
        """Check individual server health"""
        try:
            start_time = time.time()
            
            # Perform HTTP health check
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                health_url = f"http://{server.host}:{server.port}/health"
                async with session.get(health_url) as response:
                    response_time = time.time() - start_time
                    
                    # Update response times
                    server.response_times.append(response_time)
                    if len(server.response_times) > 100:
                        server.response_times = server.response_times[-100:]
                        
                    if response.status == 200:
                        server.status = ServerStatus.HEALTHY
                        server.success_count += 1
                        server.error_count = max(0, server.error_count - 1)
                    else:
                        server.error_count += 1
                        if server.error_count > 3:
                            server.status = ServerStatus.DEGRADED
                            
        except Exception as e:
            server.error_count += 1
            if server.error_count > 5:
                server.status = ServerStatus.UNHEALTHY
                
        server.last_health_check = datetime.utcnow()
        
        # Update load score
        await self._update_server_load_score(server)
        
    async def _update_server_load_score(self, server: ServerNode):
        """Update server load score based on various metrics"""
        # Base score from response time
        avg_response_time = sum(server.response_times) / len(server.response_times) if server.response_times else 0
        response_score = min(avg_response_time * 100, 100)  # Normalize to 0-100
        
        # Connection utilization score
        connection_score = (server.current_connections / server.max_connections) * 100
        
        # Error rate score
        total_requests = server.success_count + server.error_count
        error_rate = (server.error_count / total_requests) * 100 if total_requests > 0 else 0
        
        # Resource usage score
        resource_score = (server.cpu_usage + server.memory_usage) / 2
        
        # Combined load score (lower is better)
        server.load_score = (response_score * 0.3 + 
                           connection_score * 0.3 + 
                           error_rate * 0.2 + 
                           resource_score * 0.2)
                           
    async def get_next_server(self) -> Optional[ServerNode]:
        """Get next server based on load balancing strategy"""
        healthy_servers = [s for s in self.servers.values() 
                          if s.status in [ServerStatus.HEALTHY, ServerStatus.DEGRADED]]
        
        if not healthy_servers:
            return None
            
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_selection(healthy_servers)
        else:
            return random.choice(healthy_servers)
            
    def _round_robin_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Round robin server selection"""
        server = servers[self.current_index % len(servers)]
        self.current_index += 1
        return server
        
    def _weighted_round_robin_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Weighted round robin server selection"""
        total_weight = sum(s.weight for s in servers)
        random_weight = random.uniform(0, total_weight)
        
        current_weight = 0
        for server in servers:
            current_weight += server.weight
            if random_weight <= current_weight:
                return server
                
        return servers[-1]  # Fallback
        
    def _least_connections_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Least connections server selection"""
        return min(servers, key=lambda s: s.current_connections)
        
    def _least_response_time_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Least response time server selection"""
        def avg_response_time(server):
            return sum(server.response_times) / len(server.response_times) if server.response_times else float('inf')
            
        return min(servers, key=avg_response_time)
        
    def _resource_based_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Resource-based server selection"""
        def resource_score(server):
            return server.cpu_usage + server.memory_usage + (server.current_connections / server.max_connections) * 100
            
        return min(servers, key=resource_score)
        
    def _adaptive_selection(self, servers: List[ServerNode]) -> ServerNode:
        """Adaptive server selection based on load score"""
        return min(servers, key=lambda s: s.load_score)
        
    async def execute_request(self, 
                            request_func: Callable,
                            *args, 
                            **kwargs) -> Any:
        """Execute request with load balancing and retry logic"""
        start_time = time.time()
        last_exception = None
        
        for attempt in range(self.max_retries):
            server = await self.get_next_server()
            if not server:
                raise Exception("No healthy servers available")
                
            try:
                # Increment connection count
                server.current_connections += 1
                
                # Execute request
                result = await request_func(server, *args, **kwargs)
                
                # Record successful request
                execution_time = time.time() - start_time
                await self._record_request(server, True, execution_time)
                
                return result
                
            except Exception as e:
                last_exception = e
                execution_time = time.time() - start_time
                await self._record_request(server, False, execution_time)
                
                # Mark server as degraded if multiple failures
                server.error_count += 1
                if server.error_count > 3:
                    server.status = ServerStatus.DEGRADED
                    
            finally:
                # Decrement connection count
                server.current_connections = max(0, server.current_connections - 1)
                
        # All retries failed
        raise last_exception or Exception("Request failed after all retries")
        
    async def _record_request(self, server: ServerNode, success: bool, execution_time: float):
        """Record request statistics"""
        self.stats.total_requests += 1
        
        if success:
            self.stats.successful_requests += 1
            server.success_count += 1
        else:
            self.stats.failed_requests += 1
            
        # Update server distribution
        self.stats.server_distribution[server.id] += 1
        
        # Update response time statistics
        if success:
            server.response_times.append(execution_time)
            if len(server.response_times) > 100:
                server.response_times = server.response_times[-100:]
                
        # Record request history
        self.request_history.append({
            'timestamp': datetime.utcnow(),
            'server_id': server.id,
            'success': success,
            'execution_time': execution_time
        })
        
        # Limit history size
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
            
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics"""
        # Calculate average response time
        if self.stats.successful_requests > 0:
            total_response_time = sum(
                req['execution_time'] for req in self.request_history 
                if req['success']
            )
            self.stats.avg_response_time = total_response_time / self.stats.successful_requests
            
        # Calculate requests per second
        if len(self.request_history) > 1:
            time_span = (self.request_history[-1]['timestamp'] - self.request_history[0]['timestamp']).total_seconds()
            if time_span > 0:
                self.stats.requests_per_second = len(self.request_history) / time_span
                
        # Calculate active connections
        self.stats.active_connections = sum(s.current_connections for s in self.servers.values())
        
        return {
            'strategy': self.strategy.value,
            'total_servers': len(self.servers),
            'healthy_servers': len([s for s in self.servers.values() if s.status == ServerStatus.HEALTHY]),
            'stats': {
                'total_requests': self.stats.total_requests,
                'successful_requests': self.stats.successful_requests,
                'failed_requests': self.stats.failed_requests,
                'success_rate': (self.stats.successful_requests / self.stats.total_requests * 100) if self.stats.total_requests > 0 else 0,
                'avg_response_time': self.stats.avg_response_time,
                'requests_per_second': self.stats.requests_per_second,
                'active_connections': self.stats.active_connections
            },
            'server_distribution': self.stats.server_distribution,
            'servers': {
                server.id: {
                    'status': server.status.value,
                    'current_connections': server.current_connections,
                    'load_score': server.load_score,
                    'avg_response_time': sum(server.response_times) / len(server.response_times) if server.response_times else 0,
                    'error_rate': (server.error_count / (server.success_count + server.error_count) * 100) if (server.success_count + server.error_count) > 0 else 0
                }
                for server in self.servers.values()
            }
        }
        
    async def optimize_strategy(self) -> LoadBalancingStrategy:
        """Automatically optimize load balancing strategy"""
        if len(self.request_history) < 100:
            return self.strategy
            
        # Analyze performance metrics for different strategies
        strategies_performance = {}
        
        # Simulate different strategies on recent requests
        recent_requests = self.request_history[-100:]
        
        for strategy in LoadBalancingStrategy:
            # Simulate strategy performance
            simulated_response_times = []
            simulated_success_rate = 0
            
            # Simple simulation based on server characteristics
            for req in recent_requests:
                server = self.servers.get(req['server_id'])
                if server:
                    if strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                        # Would likely choose servers with better response times
                        simulated_response_times.append(req['execution_time'] * 0.9)
                    elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                        # Would distribute load more evenly
                        simulated_response_times.append(req['execution_time'] * 0.95)
                    elif strategy == LoadBalancingStrategy.ADAPTIVE:
                        # Would optimize based on multiple factors
                        simulated_response_times.append(req['execution_time'] * 0.85)
                    else:
                        simulated_response_times.append(req['execution_time'])
                        
                    if req['success']:
                        simulated_success_rate += 1
                        
            avg_response_time = sum(simulated_response_times) / len(simulated_response_times) if simulated_response_times else float('inf')
            success_rate = simulated_success_rate / len(recent_requests) if recent_requests else 0
            
            # Combined performance score (lower is better)
            performance_score = avg_response_time * (1 - success_rate)
            strategies_performance[strategy] = performance_score
            
        # Choose best performing strategy
        best_strategy = min(strategies_performance.keys(), key=lambda k: strategies_performance[k])
        
        if best_strategy != self.strategy:
            print(f"Optimizing load balancing strategy from {self.strategy.value} to {best_strategy.value}")
            self.strategy = best_strategy
            
        return self.strategy

# Global load balancer instance
load_balancer = LoadBalancer()
```

## Frontend Performance Dashboard

**File**: `frontend/components/performance/performance-dashboard.tsx`

```tsx
import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { RefreshCw, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';

interface PerformanceMetric {
  timestamp: string;
  cpu_usage: number;
  memory_usage: number;
  response_time: number;
  requests_per_second: number;
  error_rate: number;
}

interface CacheStats {
  hit_rate: number;
  miss_rate: number;
  total_requests: number;
  cache_size: number;
  evictions: number;
}

interface DatabaseStats {
  slow_queries_count: number;
  avg_query_time: number;
  connection_pool_usage: number;
  index_recommendations: number;
}

interface MemoryStats {
  system_memory: {
    total_gb: number;
    available_gb: number;
    used_percent: number;
  };
  process_memory: {
    rss_mb: number;
    vms_mb: number;
  };
  detected_leaks: number;
  memory_pools: Record<string, {
    hit_rate: number;
    current_size: number;
    max_size: number;
  }>;
}

interface LoadBalancerStats {
  strategy: string;
  total_servers: number;
  healthy_servers: number;
  stats: {
    total_requests: number;
    success_rate: number;
    avg_response_time: number;
    requests_per_second: number;
  };
  server_distribution: Record<string, number>;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export default function PerformanceDashboard() {
  const [performanceData, setPerformanceData] = useState<PerformanceMetric[]>([]);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [databaseStats, setDatabaseStats] = useState<DatabaseStats | null>(null);
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null);
  const [loadBalancerStats, setLoadBalancerStats] = useState<LoadBalancerStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  const fetchPerformanceData = async () => {
    try {
      setIsLoading(true);
      
      // Fetch performance metrics
      const metricsResponse = await fetch('/api/performance/metrics');
      const metrics = await metricsResponse.json();
      setPerformanceData(metrics.data || []);
      
      // Fetch cache statistics
      const cacheResponse = await fetch('/api/performance/cache/stats');
      const cache = await cacheResponse.json();
      setCacheStats(cache.data);
      
      // Fetch database statistics
      const dbResponse = await fetch('/api/performance/database/stats');
      const db = await dbResponse.json();
      setDatabaseStats(db.data);
      
      // Fetch memory statistics
      const memoryResponse = await fetch('/api/performance/memory/stats');
      const memory = await memoryResponse.json();
      setMemoryStats(memory.data);
      
      // Fetch load balancer statistics
      const lbResponse = await fetch('/api/performance/load-balancer/stats');
      const lb = await lbResponse.json();
      setLoadBalancerStats(lb.data);
      
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching performance data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPerformanceData();
    
    // Set up real-time updates
    const interval = setInterval(fetchPerformanceData, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
 }
 ```

## API Endpoints

**File**: `backend/api/performance.py`

```python
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

from ..core.performance.cache_manager import cache_manager
from ..core.performance.database_optimizer import database_optimizer
from ..core.performance.memory_manager import memory_manager
from ..core.performance.load_balancer import load_balancer
from ..core.performance.metrics_collector import metrics_collector
from ..core.auth import get_current_user
from ..models.user import User

router = APIRouter(prefix="/api/performance", tags=["performance"])

@router.get("/metrics")
async def get_performance_metrics(
    hours: int = 24,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get performance metrics for the specified time period"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = await metrics_collector.get_metrics_range(start_time, end_time)
        
        return {
            "success": True,
            "data": metrics,
            "metadata": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_points": len(metrics)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")

@router.get("/metrics/current")
async def get_current_metrics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current system metrics"""
    try:
        current_metrics = await metrics_collector.collect_current_metrics()
        
        return {
            "success": True,
            "data": current_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch current metrics: {str(e)}")

@router.get("/cache/stats")
async def get_cache_stats(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get cache performance statistics"""
    try:
        stats = cache_manager.get_cache_stats()
        
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch cache stats: {str(e)}")

@router.post("/cache/clear")
async def clear_cache(
    cache_type: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Clear cache (all or specific type)"""
    try:
        if cache_type:
            await cache_manager.clear_cache_type(cache_type)
            message = f"Cleared {cache_type} cache"
        else:
            await cache_manager.clear_all_caches()
            message = "Cleared all caches"
            
        return {
            "success": True,
            "message": message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/database/stats")
async def get_database_stats(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get database performance statistics"""
    try:
        stats = await database_optimizer.get_database_stats()
        
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch database stats: {str(e)}")

@router.get("/database/slow-queries")
async def get_slow_queries(
    limit: int = 50,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get slow query analysis"""
    try:
        slow_queries = await database_optimizer.get_slow_queries(limit)
        
        return {
            "success": True,
            "data": slow_queries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch slow queries: {str(e)}")

@router.post("/database/optimize")
async def optimize_database(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Run database optimization"""
    try:
        optimization_results = await database_optimizer.optimize_database()
        
        return {
            "success": True,
            "data": optimization_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize database: {str(e)}")

@router.get("/memory/stats")
async def get_memory_stats(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get memory usage statistics"""
    try:
        stats = memory_manager.get_memory_stats()
        
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch memory stats: {str(e)}")

@router.post("/memory/gc")
async def trigger_garbage_collection(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Trigger garbage collection"""
    try:
        gc_results = memory_manager.force_garbage_collection()
        
        return {
            "success": True,
            "data": gc_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger GC: {str(e)}")

@router.get("/load-balancer/stats")
async def get_load_balancer_stats(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get load balancer statistics"""
    try:
        stats = load_balancer.get_load_balancer_stats()
        
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch load balancer stats: {str(e)}")

@router.post("/load-balancer/optimize")
async def optimize_load_balancer(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Optimize load balancing strategy"""
    try:
        new_strategy = await load_balancer.optimize_strategy()
        
        return {
            "success": True,
            "data": {
                "new_strategy": new_strategy.value,
                "message": f"Load balancing strategy optimized to {new_strategy.value}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize load balancer: {str(e)}")

@router.get("/health")
async def performance_health_check() -> Dict[str, Any]:
    """Performance system health check"""
    try:
        health_status = {
            "cache": "healthy" if cache_manager.is_healthy() else "unhealthy",
            "database": "healthy" if await database_optimizer.is_healthy() else "unhealthy",
            "memory": "healthy" if memory_manager.is_healthy() else "unhealthy",
            "load_balancer": "healthy" if len(load_balancer.servers) > 0 else "unhealthy"
        }
        
        overall_health = "healthy" if all(status == "healthy" for status in health_status.values()) else "degraded"
        
        return {
            "success": True,
            "data": {
                "overall_health": overall_health,
                "components": health_status,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/recommendations")
async def get_performance_recommendations(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get performance optimization recommendations"""
    try:
        recommendations = []
        
        # Cache recommendations
        cache_stats = cache_manager.get_cache_stats()
        if cache_stats['hit_rate'] < 0.8:
            recommendations.append({
                "type": "cache",
                "priority": "high",
                "title": "Low Cache Hit Rate",
                "description": f"Cache hit rate is {cache_stats['hit_rate']*100:.1f}%. Consider optimizing cache keys or increasing cache size.",
                "action": "Review cache configuration and usage patterns"
            })
            
        # Database recommendations
        db_stats = await database_optimizer.get_database_stats()
        if db_stats['slow_queries_count'] > 10:
            recommendations.append({
                "type": "database",
                "priority": "high",
                "title": "High Number of Slow Queries",
                "description": f"Found {db_stats['slow_queries_count']} slow queries. Database performance may be degraded.",
                "action": "Review and optimize slow queries, consider adding indexes"
            })
            
        # Memory recommendations
        memory_stats = memory_manager.get_memory_stats()
        if memory_stats['detected_leaks'] > 0:
            recommendations.append({
                "type": "memory",
                "priority": "critical",
                "title": "Memory Leaks Detected",
                "description": f"Detected {memory_stats['detected_leaks']} potential memory leaks.",
                "action": "Investigate and fix memory leaks immediately"
            })
            
        # Load balancer recommendations
        lb_stats = load_balancer.get_load_balancer_stats()
        if lb_stats['stats']['success_rate'] < 95:
            recommendations.append({
                "type": "load_balancer",
                "priority": "medium",
                "title": "Low Success Rate",
                "description": f"Load balancer success rate is {lb_stats['stats']['success_rate']:.1f}%.",
                "action": "Check server health and consider adjusting load balancing strategy"
            })
            
        return {
            "success": True,
            "data": {
                "recommendations": recommendations,
                "total_count": len(recommendations),
                "critical_count": len([r for r in recommendations if r['priority'] == 'critical']),
                "high_count": len([r for r in recommendations if r['priority'] == 'high']),
                "medium_count": len([r for r in recommendations if r['priority'] == 'medium'])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")
```

## Integration and Initialization

**File**: `backend/core/performance/__init__.py`

```python
"""Performance optimization module initialization"""

import asyncio
import logging
from typing import Dict, Any

from .cache_manager import cache_manager
from .database_optimizer import database_optimizer
from .memory_manager import memory_manager
from .load_balancer import load_balancer
from .metrics_collector import metrics_collector

logger = logging.getLogger(__name__)

class PerformanceManager:
    """Central performance management coordinator"""
    
    def __init__(self):
        self.cache_manager = cache_manager
        self.database_optimizer = database_optimizer
        self.memory_manager = memory_manager
        self.load_balancer = load_balancer
        self.metrics_collector = metrics_collector
        
        self.initialized = False
        self.monitoring_tasks = []
        
    async def initialize(self, config: Dict[str, Any] = None):
        """Initialize all performance components"""
        if self.initialized:
            return
            
        logger.info("Initializing performance optimization framework...")
        
        try:
            # Initialize cache manager
            await self.cache_manager.initialize(config.get('cache', {}) if config else {})
            logger.info("Cache manager initialized")
            
            # Initialize database optimizer
            await self.database_optimizer.initialize(config.get('database', {}) if config else {})
            logger.info("Database optimizer initialized")
            
            # Initialize memory manager
            self.memory_manager.initialize(config.get('memory', {}) if config else {})
            logger.info("Memory manager initialized")
            
            # Initialize load balancer
            await self.load_balancer.start_monitoring()
            logger.info("Load balancer initialized")
            
            # Initialize metrics collector
            await self.metrics_collector.start_collection()
            logger.info("Metrics collector initialized")
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            self.initialized = True
            logger.info("Performance optimization framework initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance framework: {e}")
            raise
            
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        # Memory monitoring task
        memory_task = asyncio.create_task(self._memory_monitoring_loop())
        self.monitoring_tasks.append(memory_task)
        
        # Cache optimization task
        cache_task = asyncio.create_task(self._cache_optimization_loop())
        self.monitoring_tasks.append(cache_task)
        
        # Database optimization task
        db_task = asyncio.create_task(self._database_optimization_loop())
        self.monitoring_tasks.append(db_task)
        
        logger.info("Started performance monitoring tasks")
        
    async def _memory_monitoring_loop(self):
        """Background memory monitoring and cleanup"""
        while True:
            try:
                # Check for memory leaks
                leaks = self.memory_manager.detect_memory_leaks()
                if leaks:
                    logger.warning(f"Detected {len(leaks)} potential memory leaks")
                    
                # Perform periodic cleanup
                self.memory_manager.cleanup_unused_objects()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(60)  # Shorter sleep on error
                
    async def _cache_optimization_loop(self):
        """Background cache optimization"""
        while True:
            try:
                # Optimize cache configuration
                await self.cache_manager.optimize_cache_sizes()
                
                # Clean up expired entries
                await self.cache_manager.cleanup_expired()
                
                # Sleep for 10 minutes
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
                await asyncio.sleep(300)  # Shorter sleep on error
                
    async def _database_optimization_loop(self):
        """Background database optimization"""
        while True:
            try:
                # Analyze slow queries
                await self.database_optimizer.analyze_slow_queries()
                
                # Update query statistics
                await self.database_optimizer.update_query_stats()
                
                # Sleep for 30 minutes
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Database optimization error: {e}")
                await asyncio.sleep(600)  # Shorter sleep on error
                
    async def shutdown(self):
        """Shutdown performance framework"""
        if not self.initialized:
            return
            
        logger.info("Shutting down performance optimization framework...")
        
        try:
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
                
            # Wait for tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
                
            # Shutdown components
            await self.load_balancer.stop_monitoring()
            await self.metrics_collector.stop_collection()
            await self.cache_manager.shutdown()
            
            self.initialized = False
            logger.info("Performance optimization framework shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during performance framework shutdown: {e}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system performance status"""
        if not self.initialized:
            return {"status": "not_initialized"}
            
        try:
            cache_stats = self.cache_manager.get_cache_stats()
            memory_stats = self.memory_manager.get_memory_stats()
            lb_stats = self.load_balancer.get_load_balancer_stats()
            
            # Calculate overall health score
            health_score = 100
            
            # Cache health (30% weight)
            cache_health = cache_stats['hit_rate'] * 100
            health_score = health_score * 0.7 + cache_health * 0.3
            
            # Memory health (25% weight)
            memory_health = 100 - memory_stats['system_memory']['used_percent']
            if memory_stats['detected_leaks'] > 0:
                memory_health -= 20  # Penalty for memory leaks
            health_score = health_score * 0.75 + memory_health * 0.25
            
            # Load balancer health (25% weight)
            lb_health = lb_stats['stats']['success_rate']
            health_score = health_score * 0.75 + lb_health * 0.25
            
            # Database health (20% weight) - would need async call for real data
            db_health = 90  # Placeholder
            health_score = health_score * 0.8 + db_health * 0.2
            
            status = "excellent" if health_score >= 90 else "good" if health_score >= 75 else "fair" if health_score >= 60 else "poor"
            
            return {
                "status": status,
                "health_score": round(health_score, 1),
                "components": {
                    "cache": {
                        "status": "healthy" if cache_stats['hit_rate'] > 0.8 else "degraded",
                        "hit_rate": cache_stats['hit_rate']
                    },
                    "memory": {
                        "status": "healthy" if memory_stats['detected_leaks'] == 0 else "warning",
                        "usage_percent": memory_stats['system_memory']['used_percent']
                    },
                    "load_balancer": {
                        "status": "healthy" if lb_stats['stats']['success_rate'] > 95 else "degraded",
                        "success_rate": lb_stats['stats']['success_rate']
                    }
                },
                "monitoring_active": len(self.monitoring_tasks) > 0
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"status": "error", "error": str(e)}

# Global performance manager instance
performance_manager = PerformanceManager()

# Convenience exports
__all__ = [
    'performance_manager',
    'cache_manager',
    'database_optimizer', 
    'memory_manager',
    'load_balancer',
    'metrics_collector'
]
```

## Configuration

**File**: `backend/config/performance.py`

```python
"""Performance optimization configuration"""

from typing import Dict, Any
from pydantic import BaseSettings, Field

class PerformanceConfig(BaseSettings):
    """Performance optimization configuration"""
    
    # Cache configuration
    cache_enabled: bool = Field(True, description="Enable caching")
    cache_default_ttl: int = Field(3600, description="Default cache TTL in seconds")
    cache_max_size: int = Field(1000, description="Maximum cache size")
    cache_cleanup_interval: int = Field(300, description="Cache cleanup interval in seconds")
    
    # Database optimization
    db_optimization_enabled: bool = Field(True, description="Enable database optimization")
    db_slow_query_threshold: float = Field(1.0, description="Slow query threshold in seconds")
    db_connection_pool_size: int = Field(20, description="Database connection pool size")
    db_query_cache_size: int = Field(1000, description="Query cache size")
    
    # Memory management
    memory_monitoring_enabled: bool = Field(True, description="Enable memory monitoring")
    memory_gc_threshold: float = Field(0.8, description="Memory usage threshold for GC")
    memory_leak_detection: bool = Field(True, description="Enable memory leak detection")
    memory_pool_sizes: Dict[str, int] = Field(
        default_factory=lambda: {
            "small": 1000,
            "medium": 500,
            "large": 100
        },
        description="Memory pool sizes"
    )
    
    # Load balancing
    load_balancer_enabled: bool = Field(True, description="Enable load balancing")
    load_balancer_strategy: str = Field("adaptive", description="Load balancing strategy")
    load_balancer_health_check_interval: int = Field(30, description="Health check interval in seconds")
    load_balancer_max_retries: int = Field(3, description="Maximum retry attempts")
    
    # Metrics collection
    metrics_enabled: bool = Field(True, description="Enable metrics collection")
    metrics_collection_interval: int = Field(60, description="Metrics collection interval in seconds")
    metrics_retention_days: int = Field(30, description="Metrics retention period in days")
    metrics_batch_size: int = Field(100, description="Metrics batch size")
    
    # Performance thresholds
    response_time_threshold: float = Field(1.0, description="Response time threshold in seconds")
    cpu_usage_threshold: float = Field(80.0, description="CPU usage threshold percentage")
    memory_usage_threshold: float = Field(85.0, description="Memory usage threshold percentage")
    error_rate_threshold: float = Field(5.0, description="Error rate threshold percentage")
    
    class Config:
        env_prefix = "PERFORMANCE_"
        case_sensitive = False

def get_performance_config() -> PerformanceConfig:
    """Get performance configuration"""
    return PerformanceConfig()

# Default configuration
DEFAULT_PERFORMANCE_CONFIG = {
    "cache": {
        "enabled": True,
        "default_ttl": 3600,
        "max_size": 1000,
        "cleanup_interval": 300
    },
    "database": {
        "optimization_enabled": True,
        "slow_query_threshold": 1.0,
        "connection_pool_size": 20,
        "query_cache_size": 1000
    },
    "memory": {
        "monitoring_enabled": True,
        "gc_threshold": 0.8,
        "leak_detection": True,
        "pool_sizes": {
            "small": 1000,
            "medium": 500,
            "large": 100
        }
    },
    "load_balancer": {
        "enabled": True,
        "strategy": "adaptive",
        "health_check_interval": 30,
        "max_retries": 3
    },
    "metrics": {
        "enabled": True,
        "collection_interval": 60,
        "retention_days": 30,
        "batch_size": 100
    },
    "thresholds": {
        "response_time": 1.0,
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "error_rate": 5.0
    }
}
```

## Testing Framework

**File**: `tests/performance/test_performance_framework.py`

```python
"""Performance framework tests"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from backend.core.performance import performance_manager
from backend.core.performance.cache_manager import cache_manager
from backend.core.performance.memory_manager import memory_manager
from backend.core.performance.load_balancer import load_balancer, ServerNode, LoadBalancingStrategy

class TestCacheManager:
    """Test cache manager functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test basic cache operations"""
        # Test set and get
        await cache_manager.set("test_key", "test_value", ttl=60)
        value = await cache_manager.get("test_key")
        assert value == "test_value"
        
        # Test expiration
        await cache_manager.set("expire_key", "expire_value", ttl=1)
        await asyncio.sleep(2)
        value = await cache_manager.get("expire_key")
        assert value is None
        
    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics"""
        # Clear cache first
        await cache_manager.clear_all_caches()
        
        # Perform some operations
        await cache_manager.set("key1", "value1")
        await cache_manager.get("key1")  # Hit
        await cache_manager.get("key2")  # Miss
        
        stats = cache_manager.get_cache_stats()
        assert stats['total_requests'] >= 2
        assert stats['hit_rate'] > 0
        
class TestMemoryManager:
    """Test memory manager functionality"""
    
    def test_memory_stats(self):
        """Test memory statistics collection"""
        stats = memory_manager.get_memory_stats()
        
        assert 'system_memory' in stats
        assert 'process_memory' in stats
        assert 'memory_pools' in stats
        assert stats['system_memory']['total_gb'] > 0
        
    def test_memory_pool_operations(self):
        """Test memory pool operations"""
        # Get object from pool
        obj = memory_manager.get_from_pool('test_pool', lambda: {'data': 'test'})
        assert obj is not None
        
        # Return object to pool
        memory_manager.return_to_pool('test_pool', obj)
        
        # Get same object again (should be reused)
        obj2 = memory_manager.get_from_pool('test_pool', lambda: {'data': 'new'})
        assert obj2 == obj
        
class TestLoadBalancer:
    """Test load balancer functionality"""
    
    @pytest.mark.asyncio
    async def test_server_management(self):
        """Test server addition and removal"""
        server = ServerNode(
            id="test_server",
            host="localhost",
            port=8000
        )
        
        await load_balancer.add_server(server)
        assert "test_server" in load_balancer.servers
        
        await load_balancer.remove_server("test_server")
        assert "test_server" not in load_balancer.servers
        
    @pytest.mark.asyncio
    async def test_load_balancing_strategies(self):
        """Test different load balancing strategies"""
        # Add test servers
        servers = [
            ServerNode(id="server1", host="localhost", port=8001),
            ServerNode(id="server2", host="localhost", port=8002),
            ServerNode(id="server3", host="localhost", port=8003)
        ]
        
        for server in servers:
            await load_balancer.add_server(server)
            
        # Test round robin
        load_balancer.strategy = LoadBalancingStrategy.ROUND_ROBIN
        selected_servers = []
        for _ in range(6):
            server = await load_balancer.get_next_server()
            selected_servers.append(server.id)
            
        # Should cycle through servers
        assert len(set(selected_servers)) == 3
        
        # Clean up
        for server in servers:
            await load_balancer.remove_server(server.id)
            
class TestPerformanceManager:
    """Test performance manager coordination"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test performance manager initialization"""
        config = {
            "cache": {"enabled": True},
            "memory": {"monitoring_enabled": True},
            "load_balancer": {"enabled": True},
            "metrics": {"enabled": True}
        }
        
        await performance_manager.initialize(config)
        assert performance_manager.initialized
        
        # Test system status
        status = performance_manager.get_system_status()
        assert 'status' in status
        assert 'health_score' in status
        assert 'components' in status
        
        await performance_manager.shutdown()
        
    def test_system_status(self):
        """Test system status reporting"""
        status = performance_manager.get_system_status()
        
        if performance_manager.initialized:
            assert status['status'] in ['excellent', 'good', 'fair', 'poor']
            assert 'health_score' in status
            assert 'components' in status
        else:
            assert status['status'] == 'not_initialized'
            
@pytest.mark.asyncio
async def test_integration_workflow():
    """Test complete performance optimization workflow"""
    # Initialize performance framework
    config = {
        "cache": {"enabled": True, "max_size": 100},
        "memory": {"monitoring_enabled": True},
        "load_balancer": {"enabled": True},
        "metrics": {"enabled": True}
    }
    
    await performance_manager.initialize(config)
    
    try:
        # Test cache operations
        await cache_manager.set("integration_test", "test_data")
        cached_data = await cache_manager.get("integration_test")
        assert cached_data == "test_data"
        
        # Test memory operations
        memory_stats = memory_manager.get_memory_stats()
        assert memory_stats is not None
        
        # Test load balancer
        server = ServerNode(id="integration_server", host="localhost", port=9000)
        await load_balancer.add_server(server)
        
        selected_server = await load_balancer.get_next_server()
        assert selected_server is not None
        
        # Test system status
        status = performance_manager.get_system_status()
        assert status['status'] != 'error'
        
    finally:
        await performance_manager.shutdown()
        
if __name__ == "__main__":
    pytest.main([__file__])
```

## Documentation

**File**: `docs/performance-optimization.md`

# Performance Optimization Framework

The SentientCore performance optimization framework provides comprehensive tools for monitoring, analyzing, and optimizing system performance across multiple dimensions.

## Overview

The framework consists of five main components:

1. **Cache Manager** - Intelligent caching with multiple strategies
2. **Database Optimizer** - Query optimization and performance monitoring
3. **Memory Manager** - Memory leak detection and pool management
4. **Load Balancer** - Adaptive load balancing with health monitoring
5. **Metrics Collector** - Real-time performance metrics collection

## Quick Start

```python
from backend.core.performance import performance_manager

# Initialize the performance framework
config = {
    "cache": {"enabled": True, "max_size": 1000},
    "memory": {"monitoring_enabled": True},
    "load_balancer": {"strategy": "adaptive"},
    "metrics": {"collection_interval": 60}
}

await performance_manager.initialize(config)

# Get system status
status = performance_manager.get_system_status()
print(f"System health: {status['status']} ({status['health_score']}%)")
```

## Configuration

The framework can be configured through environment variables or configuration files:

```bash
# Cache configuration
PERFORMANCE_CACHE_ENABLED=true
PERFORMANCE_CACHE_DEFAULT_TTL=3600
PERFORMANCE_CACHE_MAX_SIZE=1000

# Database optimization
PERFORMANCE_DB_OPTIMIZATION_ENABLED=true
PERFORMANCE_DB_SLOW_QUERY_THRESHOLD=1.0

# Memory management
PERFORMANCE_MEMORY_MONITORING_ENABLED=true
PERFORMANCE_MEMORY_GC_THRESHOLD=0.8

# Load balancing
PERFORMANCE_LOAD_BALANCER_STRATEGY=adaptive
PERFORMANCE_LOAD_BALANCER_HEALTH_CHECK_INTERVAL=30
```

## API Endpoints

The framework provides REST API endpoints for monitoring and control:

- `GET /api/performance/metrics` - Get performance metrics
- `GET /api/performance/cache/stats` - Get cache statistics
- `GET /api/performance/database/stats` - Get database performance
- `GET /api/performance/memory/stats` - Get memory usage
- `GET /api/performance/load-balancer/stats` - Get load balancer status
- `GET /api/performance/health` - System health check
- `GET /api/performance/recommendations` - Get optimization recommendations

## Frontend Dashboard

The performance dashboard provides real-time visualization of system metrics:

- System overview with key performance indicators
- Interactive charts for performance trends
- Cache hit/miss analysis
- Database query performance
- Memory usage and leak detection
- Load balancer distribution and health

## Best Practices

1. **Cache Strategy**: Use appropriate TTL values and cache keys
2. **Database Optimization**: Monitor slow queries and add indexes
3. **Memory Management**: Regular garbage collection and leak detection
4. **Load Balancing**: Choose the right strategy for your workload
5. **Monitoring**: Set up alerts for performance thresholds

## Troubleshooting

### High Memory Usage
- Check for memory leaks using the memory manager
- Review object pools and cleanup unused objects
- Adjust garbage collection thresholds

### Low Cache Hit Rate
- Review cache key patterns and TTL settings
- Analyze cache usage patterns
- Consider increasing cache size

### Database Performance Issues
- Identify slow queries using the database optimizer
- Add appropriate indexes
- Optimize query patterns

### Load Balancer Issues
- Check server health status
- Review load balancing strategy
- Monitor request distribution

For more detailed information, see the API documentation and component-specific guides.
```

## Human Testing Scenarios

### Backend Testing

1. **Performance Profiling**
   - Start performance profiling for a specific function
   - Execute the function multiple times with different inputs
   - Verify profiling data collection and bottleneck identification
   - Check performance threshold alerts

2. **Cache Management**
   - Set cache entries with different TTL values
   - Verify cache hit/miss statistics
   - Test cache eviction policies
   - Clear specific cache types and verify cleanup

3. **Database Optimization**
   - Execute slow queries and verify detection
   - Check index recommendations
   - Test connection pool optimization
   - Verify query statistics collection

4. **Memory Management**
   - Monitor memory usage during high-load operations
   - Test memory leak detection
   - Verify garbage collection triggers
   - Check memory pool efficiency

5. **Load Balancing**
   - Add/remove servers dynamically
   - Test different load balancing strategies
   - Verify health check functionality
   - Check request distribution patterns

### Frontend Testing

1. **Performance Dashboard**
   - Verify real-time metrics display
   - Test chart interactions and filtering
   - Check responsive design on different screen sizes
   - Validate data refresh mechanisms

2. **Cache Statistics**
   - View cache hit/miss ratios
   - Test cache clearing functionality
   - Verify cache size and usage displays
   - Check cache type filtering

3. **Database Performance**
   - Review slow query analysis
   - Test database optimization triggers
   - Verify connection pool status
   - Check query performance trends

4. **Memory Monitoring**
   - Monitor memory usage graphs
   - Test memory leak alerts
   - Verify garbage collection indicators
   - Check memory pool utilization

5. **Load Balancer Status**
   - View server health status
   - Test load balancing strategy changes
   - Verify request distribution visualization
   - Check server response time metrics

### Integration Testing

1. **End-to-End Performance Workflow**
   - Initialize performance framework
   - Execute high-load operations
   - Monitor all performance metrics
   - Verify optimization recommendations
   - Test performance alerts and notifications

2. **Multi-Component Optimization**
   - Trigger cache optimization
   - Execute database optimization
   - Perform memory cleanup
   - Adjust load balancing strategy
   - Verify overall system health improvement

3. **Performance Degradation Simulation**
   - Simulate high memory usage
   - Create database bottlenecks
   - Overload cache systems
   - Test system recovery mechanisms
   - Verify alert generation and resolution

## Validation Criteria

### Backend Validation

- [ ] Performance profiling accurately identifies bottlenecks
- [ ] Cache operations maintain >95% reliability
- [ ] Database optimization reduces query times by >20%
- [ ] Memory management prevents memory leaks
- [ ] Load balancer maintains >99% uptime
- [ ] All API endpoints respond within 200ms
- [ ] Performance metrics are collected accurately
- [ ] Optimization recommendations are relevant
- [ ] System health monitoring is real-time
- [ ] Background optimization tasks run reliably

### Frontend Validation

- [ ] Dashboard loads within 2 seconds
- [ ] Real-time metrics update every 30 seconds
- [ ] Charts display data accurately
- [ ] Interactive elements respond immediately
- [ ] Mobile responsiveness works correctly
- [ ] Data filtering functions properly
- [ ] Performance alerts are visible
- [ ] Cache management controls work
- [ ] Database optimization triggers function
- [ ] Load balancer controls are responsive

### Integration Validation

- [ ] Performance framework initializes successfully
- [ ] All components integrate seamlessly
- [ ] Cross-component optimization works
- [ ] Performance alerts trigger correctly
- [ ] System recovery mechanisms function
- [ ] Data consistency across components
- [ ] Real-time updates work end-to-end
- [ ] Performance recommendations are actionable
- [ ] System health reflects actual status
- [ ] Optimization results are measurable

## Success Metrics

### Performance Improvements

- **Response Time**: Reduce average response time by 30%
- **Cache Hit Rate**: Achieve >85% cache hit rate
- **Database Performance**: Reduce slow queries by 50%
- **Memory Efficiency**: Maintain <80% memory usage
- **Load Distribution**: Achieve balanced load across servers
- **System Uptime**: Maintain >99.9% system availability

### Monitoring Effectiveness

- **Alert Accuracy**: >95% of alerts are actionable
- **Detection Speed**: Performance issues detected within 1 minute
- **Resolution Time**: Average issue resolution <5 minutes
- **False Positives**: <5% false positive rate
- **Coverage**: Monitor 100% of critical system components

### User Experience

- **Dashboard Usability**: >90% user satisfaction score
- **Information Clarity**: Performance data easily understood
- **Action Efficiency**: Optimization actions completed quickly
- **System Reliability**: Consistent performance monitoring
- **Feature Adoption**: >80% of optimization features used

---

**Next Steps**: Proceed to `25-final-integration-testing.md` for comprehensive system integration and testing procedures.

**Dependencies**: 
- Monitoring and alerting system (File 23)
- Core agent implementation (Files 8-15)
- Frontend infrastructure (Files 2-7)

**Estimated Implementation Time**: 3-4 weeks

**Priority**: High - Essential for production readiness and system optimization

  const getStatusColor = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return 'destructive';
    if (value >= thresholds.warning) return 'warning';
    return 'success';
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Performance Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor system performance, caching, database, and load balancing metrics
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-muted-foreground">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={fetchPerformanceData}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* System Performance */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {performanceData.length > 0 ? `${performanceData[performanceData.length - 1].cpu_usage.toFixed(1)}%` : 'N/A'}
            </div>
            <Badge variant={getStatusColor(
              performanceData.length > 0 ? performanceData[performanceData.length - 1].cpu_usage : 0,
              { warning: 70, critical: 90 }
            )}>
              {performanceData.length > 0 && performanceData[performanceData.length - 1].cpu_usage > 70 ? 'High' : 'Normal'}
            </Badge>
          </CardContent>
        </Card>

        {/* Memory Usage */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {memoryStats ? `${memoryStats.system_memory.used_percent.toFixed(1)}%` : 'N/A'}
            </div>
            <p className="text-xs text-muted-foreground">
              {memoryStats ? `${memoryStats.system_memory.available_gb.toFixed(1)}GB available` : ''}
            </p>
          </CardContent>
        </Card>

        {/* Cache Hit Rate */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cache Hit Rate</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {cacheStats ? `${(cacheStats.hit_rate * 100).toFixed(1)}%` : 'N/A'}
            </div>
            <Badge variant={getStatusColor(
              cacheStats ? cacheStats.hit_rate * 100 : 0,
              { warning: 70, critical: 50 }
            )}>
              {cacheStats && cacheStats.hit_rate > 0.8 ? 'Excellent' : 'Needs Optimization'}
            </Badge>
          </CardContent>
        </Card>

        {/* Response Time */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Response Time</CardTitle>
            <TrendingDown className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {performanceData.length > 0 ? `${performanceData[performanceData.length - 1].response_time.toFixed(0)}ms` : 'N/A'}
            </div>
            <Badge variant={getStatusColor(
              performanceData.length > 0 ? performanceData[performanceData.length - 1].response_time : 0,
              { warning: 500, critical: 1000 }
            )}>
              {performanceData.length > 0 && performanceData[performanceData.length - 1].response_time < 200 ? 'Fast' : 'Slow'}
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Metrics */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="cache">Cache</TabsTrigger>
          <TabsTrigger value="database">Database</TabsTrigger>
          <TabsTrigger value="memory">Memory</TabsTrigger>
          <TabsTrigger value="load-balancer">Load Balancer</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Performance Trends */}
            <Card>
              <CardHeader>
                <CardTitle>Performance Trends</CardTitle>
                <CardDescription>System performance over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="cpu_usage" stroke="#8884d8" name="CPU %" />
                    <Line type="monotone" dataKey="memory_usage" stroke="#82ca9d" name="Memory %" />
                    <Line type="monotone" dataKey="response_time" stroke="#ffc658" name="Response Time (ms)" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Request Metrics */}
            <Card>
              <CardHeader>
                <CardTitle>Request Metrics</CardTitle>
                <CardDescription>Request volume and error rates</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="requests_per_second" stroke="#8884d8" name="RPS" />
                    <Line type="monotone" dataKey="error_rate" stroke="#ff7300" name="Error Rate %" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Cache Tab */}
        <TabsContent value="cache" className="space-y-4">
          {cacheStats && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Cache Performance</CardTitle>
                  <CardDescription>Cache hit/miss statistics</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'Hits', value: cacheStats.hit_rate * 100 },
                          { name: 'Misses', value: cacheStats.miss_rate * 100 }
                        ]}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {[0, 1].map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Cache Statistics</CardTitle>
                  <CardDescription>Detailed cache metrics</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between">
                    <span>Total Requests:</span>
                    <span className="font-mono">{cacheStats.total_requests.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Cache Size:</span>
                    <span className="font-mono">{formatBytes(cacheStats.cache_size)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Evictions:</span>
                    <span className="font-mono">{cacheStats.evictions.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Hit Rate:</span>
                    <Badge variant={getStatusColor(cacheStats.hit_rate * 100, { warning: 70, critical: 50 })}>
                      {(cacheStats.hit_rate * 100).toFixed(1)}%
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* Database Tab */}
        <TabsContent value="database" className="space-y-4">
          {databaseStats && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Database Performance</CardTitle>
                  <CardDescription>Query performance metrics</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between">
                    <span>Slow Queries:</span>
                    <Badge variant={databaseStats.slow_queries_count > 10 ? 'destructive' : 'success'}>
                      {databaseStats.slow_queries_count}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Avg Query Time:</span>
                    <span className="font-mono">{databaseStats.avg_query_time.toFixed(2)}ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Connection Pool Usage:</span>
                    <Badge variant={getStatusColor(databaseStats.connection_pool_usage, { warning: 70, critical: 90 })}>
                      {databaseStats.connection_pool_usage.toFixed(1)}%
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Index Recommendations:</span>
                    <Badge variant={databaseStats.index_recommendations > 0 ? 'warning' : 'success'}>
                      {databaseStats.index_recommendations}
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              {databaseStats.index_recommendations > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Optimization Recommendations</CardTitle>
                    <CardDescription>Database optimization suggestions</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center space-x-2 text-amber-600">
                      <AlertTriangle className="h-4 w-4" />
                      <span>Index recommendations available</span>
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">
                      {databaseStats.index_recommendations} index recommendations found. 
                      Review slow queries to improve performance.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </TabsContent>

        {/* Memory Tab */}
        <TabsContent value="memory" className="space-y-4">
          {memoryStats && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Memory Usage</CardTitle>
                  <CardDescription>System and process memory statistics</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>System Memory:</span>
                      <span className="font-mono">
                        {memoryStats.system_memory.used_percent.toFixed(1)}% of {memoryStats.system_memory.total_gb.toFixed(1)}GB
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Process RSS:</span>
                      <span className="font-mono">{memoryStats.process_memory.rss_mb.toFixed(1)}MB</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Process VMS:</span>
                      <span className="font-mono">{memoryStats.process_memory.vms_mb.toFixed(1)}MB</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Memory Leaks:</span>
                      <Badge variant={memoryStats.detected_leaks > 0 ? 'destructive' : 'success'}>
                        {memoryStats.detected_leaks}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Memory Pools</CardTitle>
                  <CardDescription>Memory pool efficiency</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {Object.entries(memoryStats.memory_pools).map(([name, pool]) => (
                      <div key={name} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span>{name}:</span>
                          <span>{pool.current_size}/{pool.max_size}</span>
                        </div>
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Hit Rate:</span>
                          <Badge variant={getStatusColor(pool.hit_rate * 100, { warning: 70, critical: 50 })} className="text-xs">
                            {(pool.hit_rate * 100).toFixed(1)}%
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* Load Balancer Tab */}
        <TabsContent value="load-balancer" className="space-y-4">
          {loadBalancerStats && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Load Balancer Status</CardTitle>
                  <CardDescription>Load balancing performance and distribution</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between">
                    <span>Strategy:</span>
                    <Badge variant="outline">{loadBalancerStats.strategy}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Healthy Servers:</span>
                    <span className="font-mono">
                      {loadBalancerStats.healthy_servers}/{loadBalancerStats.total_servers}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Success Rate:</span>
                    <Badge variant={getStatusColor(loadBalancerStats.stats.success_rate, { warning: 95, critical: 90 })}>
                      {loadBalancerStats.stats.success_rate.toFixed(1)}%
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Avg Response Time:</span>
                    <span className="font-mono">{loadBalancerStats.stats.avg_response_time.toFixed(0)}ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Requests/sec:</span>
                    <span className="font-mono">{loadBalancerStats.stats.requests_per_second.toFixed(1)}</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Server Distribution</CardTitle>
                  <CardDescription>Request distribution across servers</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={Object.entries(loadBalancerStats.server_distribution).map(([server, requests]) => ({
                      server,
                      requests
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="server" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="requests" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}