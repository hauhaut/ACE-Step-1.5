"""Local cache module to replace Redis

Uses diskcache as backend, provides Redis-compatible API.
Supports persistent storage and TTL expiration.
"""

import json
import os
from typing import Any, Optional
from threading import Lock

try:
    from diskcache import Cache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False


class LocalCache:
    """
    Local cache implementation with Redis-compatible API.
    Uses diskcache as backend, supports persistence and TTL.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, cache_dir: Optional[str] = None):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_dir: Optional[str] = None):
        if getattr(self, '_initialized', False):
            return

        if not HAS_DISKCACHE:
            raise ImportError(
                "diskcache not installed. Run: pip install diskcache"
            )

        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                ".cache",
                "local_redis"
            )

        os.makedirs(cache_dir, exist_ok=True)
        self._cache = Cache(cache_dir)
        self._initialized = True

    def set(self, name: str, value: Any, ex: Optional[int] = None) -> bool:
        """
        Set key-value pair

        Args:
            name: Key name
            value: Value (auto-serialize dict/list)
            ex: Expiration time (seconds)

        Returns:
            bool: Success status
        """
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        self._cache.set(name, value, expire=ex)
        return True

    def get(self, name: str) -> Optional[str]:
        """Get value"""
        return self._cache.get(name)

    def close(self):
        """Close cache connection"""
        if hasattr(self, '_cache'):
            self._cache.close()
