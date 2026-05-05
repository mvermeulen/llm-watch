"""
Shared cache-directory resolution for llm-watch.

All agents that persist data to disk should obtain their cache directory via
:func:`get_cache_dir` rather than hardcoding ``.llmwatch_cache``.

The resolved directory can be overridden at runtime::

    export LLMWATCH_CACHE_DIR=/home/me/.cache/llm-watch

When the variable is unset the default ``.llmwatch_cache`` (relative to the
current working directory) is used, preserving the original behaviour.
"""

from __future__ import annotations

import os

_DEFAULT = ".llmwatch_cache"


def get_cache_dir() -> str:
    """Return the cache directory path, honouring ``LLMWATCH_CACHE_DIR``."""
    return os.getenv("LLMWATCH_CACHE_DIR", _DEFAULT)
