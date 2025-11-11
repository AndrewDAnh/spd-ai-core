"""Datetime utilities for RFC3339 formatting consistency."""

from datetime import datetime, UTC
from typing import Optional


def to_rfc3339(dt: datetime) -> str:
    """
    Convert datetime to RFC3339 format string.
    
    RFC3339 format: "2006-01-02T15:04:05Z07:00"
    
    Args:
        dt: Datetime object to convert
        
    Returns:
        RFC3339 formatted string
        
    Examples:
        >>> dt = datetime(2025, 11, 10, 15, 30, 45, tzinfo=UTC)
        >>> to_rfc3339(dt)
        '2025-11-10T15:30:45+00:00'
        
        >>> dt_naive = datetime(2025, 11, 10, 15, 30, 45)
        >>> to_rfc3339(dt_naive)
        '2025-11-10T15:30:45Z'
    """
    if dt.tzinfo is not None:
        # Timezone-aware datetime - use isoformat with timezone
        return dt.isoformat()
    else:
        # Naive datetime - treat as UTC and append 'Z'
        return dt.isoformat() + 'Z'


def now_rfc3339() -> str:
    """
    Get current UTC time in RFC3339 format.
    
    Returns:
        Current UTC time as RFC3339 string
        
    Example:
        >>> now_rfc3339()
        '2025-11-10T15:30:45.123456+00:00'
    """
    return datetime.now(UTC).isoformat()


def from_rfc3339(dt_str: str) -> datetime:
    """
    Parse RFC3339 formatted string to datetime.
    
    Args:
        dt_str: RFC3339 formatted datetime string
        
    Returns:
        Parsed datetime object
        
    Example:
        >>> from_rfc3339('2025-11-10T15:30:45+00:00')
        datetime.datetime(2025, 11, 10, 15, 30, 45, tzinfo=datetime.timezone.utc)
    """
    return datetime.fromisoformat(dt_str)
