"""Private implementation modules for gpype backend core.

This package contains internal helper classes that are not part of the
public API. These classes are used internally by Source and Sink composites
for distributed pipeline support.
"""

from .link import Link
from .oscar import Oscar

__all__ = ["Oscar", "Link"]
