"""
Cache Manager for IAAIR System

This module provides intelligent caching for query embeddings, search results,
and other expensive operations to dramatically reduce latency.
"""

import time
import hashlib
import pickle
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class LRUCache:
    """LRU Cache with TTL (Time To Live) support."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.timestamps:
            return True
        return (datetime.now() - self.timestamps[key]).total_seconds() > self.ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache or self._is_expired(key):
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = datetime.now()
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class CacheManager:
    """Centralized cache manager for the IAAIR system."""
    
    def __init__(self, 
                 embedding_cache_size: int = 5000,
                 search_cache_size: int = 2000,
                 embedding_ttl: int = 7200,  # 2 hours
                 search_ttl: int = 1800,     # 30 minutes
                 persistent_cache_dir: Optional[str] = None):
        """
        Initialize cache manager.
        
        Args:
            embedding_cache_size: Max number of cached embeddings
            search_cache_size: Max number of cached search results
            embedding_ttl: TTL for embeddings in seconds
            search_ttl: TTL for search results in seconds  
            persistent_cache_dir: Directory for persistent cache storage
        """
        self.embedding_cache = LRUCache(embedding_cache_size, embedding_ttl)
        self.search_cache = LRUCache(search_cache_size, search_ttl)
        self.ai_response_cache = LRUCache(1000, 3600)  # 1 hour for AI responses
        
        # Persistent cache directory
        self.persistent_cache_dir = persistent_cache_dir
        if persistent_cache_dir:
            os.makedirs(persistent_cache_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            'embedding_hits': 0,
            'embedding_misses': 0,
            'search_hits': 0,
            'search_misses': 0,
            'ai_response_hits': 0,
            'ai_response_misses': 0
        }
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching."""
        return query.lower().strip()
    
    def _generate_cache_key(self, query: str, **kwargs) -> str:
        """Generate consistent cache key."""
        normalized_query = self._normalize_query(query)
        
        # Include relevant parameters in cache key
        cache_components = [normalized_query]
        for key, value in sorted(kwargs.items()):
            cache_components.append(f"{key}:{value}")
        
        cache_string = "|".join(cache_components)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_embedding(self, query: str) -> Optional[List[float]]:
        """Get cached embedding for query."""
        cache_key = self._generate_cache_key(query)
        embedding = self.embedding_cache.get(cache_key)
        
        if embedding is not None:
            self.stats['embedding_hits'] += 1
            logger.debug(f"Embedding cache HIT for query: {query[:50]}...")
            return embedding
        
        self.stats['embedding_misses'] += 1
        logger.debug(f"Embedding cache MISS for query: {query[:50]}...")
        return None
    
    def cache_embedding(self, query: str, embedding: List[float]):
        """Cache embedding for query."""
        cache_key = self._generate_cache_key(query)
        self.embedding_cache.put(cache_key, embedding)
        logger.debug(f"Cached embedding for query: {query[:50]}...")
    
    def get_search_results(self, query: str, top_k: int, use_hybrid: bool = True,
                          routing_strategy: str = "adaptive") -> Optional[List[Dict]]:
        """Get cached search results."""
        cache_key = self._generate_cache_key(
            query, 
            top_k=top_k, 
            use_hybrid=use_hybrid,
            routing_strategy=routing_strategy
        )
        
        results = self.search_cache.get(cache_key)
        
        if results is not None:
            self.stats['search_hits'] += 1
            logger.debug(f"Search cache HIT for query: {query[:50]}...")
            return results
        
        self.stats['search_misses'] += 1  
        logger.debug(f"Search cache MISS for query: {query[:50]}...")
        return None
    
    def cache_search_results(self, query: str, results: List[Dict], top_k: int, 
                           use_hybrid: bool = True, routing_strategy: str = "adaptive"):
        """Cache search results."""
        cache_key = self._generate_cache_key(
            query,
            top_k=top_k,
            use_hybrid=use_hybrid, 
            routing_strategy=routing_strategy
        )
        
        self.search_cache.put(cache_key, results)
        logger.debug(f"Cached search results for query: {query[:50]}...")
    
    def get_ai_response(self, query: str, results_hash: str) -> Optional[str]:
        """Get cached AI response."""
        cache_key = self._generate_cache_key(query, results_hash=results_hash)
        response = self.ai_response_cache.get(cache_key)
        
        if response is not None:
            self.stats['ai_response_hits'] += 1
            logger.debug(f"AI response cache HIT for query: {query[:50]}...")
            return response
            
        self.stats['ai_response_misses'] += 1
        return None
    
    def cache_ai_response(self, query: str, results_hash: str, response: str):
        """Cache AI response."""
        cache_key = self._generate_cache_key(query, results_hash=results_hash)
        self.ai_response_cache.put(cache_key, response)
        logger.debug(f"Cached AI response for query: {query[:50]}...")
    
    def get_persistent_cache(self, cache_key: str) -> Optional[Any]:
        """Get value from persistent cache."""
        if not self.persistent_cache_dir:
            return None
            
        cache_file = os.path.join(self.persistent_cache_dir, f"{cache_key}.pkl")
        if not os.path.exists(cache_file):
            return None
        
        try:
            # Check if file is too old (older than 24 hours)
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age > 86400:  # 24 hours
                os.remove(cache_file)
                return None
                
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
            try:
                os.remove(cache_file)
            except:
                pass
            return None
    
    def set_persistent_cache(self, cache_key: str, value: Any):
        """Set value in persistent cache."""
        if not self.persistent_cache_dir:
            return
            
        cache_file = os.path.join(self.persistent_cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save persistent cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_embedding_requests = self.stats['embedding_hits'] + self.stats['embedding_misses']
        total_search_requests = self.stats['search_hits'] + self.stats['search_misses']
        total_ai_requests = self.stats['ai_response_hits'] + self.stats['ai_response_misses']
        
        return {
            'embedding_cache': {
                'hits': self.stats['embedding_hits'],
                'misses': self.stats['embedding_misses'],
                'hit_rate': (self.stats['embedding_hits'] / max(1, total_embedding_requests)) * 100,
                'cache_size': self.embedding_cache.size()
            },
            'search_cache': {
                'hits': self.stats['search_hits'], 
                'misses': self.stats['search_misses'],
                'hit_rate': (self.stats['search_hits'] / max(1, total_search_requests)) * 100,
                'cache_size': self.search_cache.size()
            },
            'ai_response_cache': {
                'hits': self.stats['ai_response_hits'],
                'misses': self.stats['ai_response_misses'], 
                'hit_rate': (self.stats['ai_response_hits'] / max(1, total_ai_requests)) * 100,
                'cache_size': self.ai_response_cache.size()
            }
        }
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        self.search_cache.clear() 
        self.ai_response_cache.clear()
        
        # Clear persistent cache
        if self.persistent_cache_dir and os.path.exists(self.persistent_cache_dir):
            for file in os.listdir(self.persistent_cache_dir):
                if file.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(self.persistent_cache_dir, file))
                    except:
                        pass
        
        # Reset stats
        for key in self.stats:
            self.stats[key] = 0
        
        logger.info("Cleared all caches")
    
    def warm_up_cache(self, common_queries: List[str]):
        """Pre-warm cache with common queries."""
        logger.info(f"Warming up cache with {len(common_queries)} common queries")
        
        # This would be implemented with actual search calls
        # For now, just log the intent
        for query in common_queries:
            logger.debug(f"Would warm up cache for: {query}")