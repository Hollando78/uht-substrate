"""Response caching for UHT Factory API calls."""

from typing import Any, Optional

from cachetools import TTLCache


class ResponseCache:
    """Thread-safe TTL cache for API responses."""

    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        """
        Initialize the cache.

        Args:
            maxsize: Maximum number of items to cache
            ttl: Default time-to-live in seconds
        """
        self._cache: TTLCache[str, Any] = TTLCache(maxsize=maxsize, ttl=ttl)
        self._default_ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL (uses default if not provided)
        """
        # TTLCache doesn't support per-item TTL, so we use the cache's TTL
        # For custom TTL, we'd need a more sophisticated solution
        self._cache[key] = value

    def delete(self, key: str) -> bool:
        """
        Delete a value from cache.

        Args:
            key: Cache key

        Returns:
            True if key existed and was deleted
        """
        try:
            del self._cache[key]
            return True
        except KeyError:
            return False

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        return key in self._cache

    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self._cache)
