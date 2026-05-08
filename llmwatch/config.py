"""
Configuration management for llm-watch.

Supports loading watcher configurations from YAML/JSON files and merging with CLI options.

Configuration files use the format:

    watchers:
      enabled:
        - huggingface
        - ollama
      disabled:
        - tldr_ai

Or enable most watchers with selective disables:

    watchers:
      disabled:
        - tldr_ai
        - vendor_blogs

The configuration file location can be specified via:
  1. --config-file CLI option
  2. LLMWATCH_CONFIG environment variable
  3. Default: llmwatch.yaml in current directory (if it exists)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_config_path() -> str | None:
    """Get config file path from env var or default."""
    if env_path := os.getenv("LLMWATCH_CONFIG"):
        return env_path
    
    default_path = Path("llmwatch.yaml")
    if default_path.exists():
        return str(default_path)
    
    default_path = Path("llmwatch.yml")
    if default_path.exists():
        return str(default_path)
    
    default_path = Path("llmwatch.json")
    if default_path.exists():
        return str(default_path)
    
    return None


def load_config_file(file_path: str) -> dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Parameters
    ----------
    file_path:
        Path to the configuration file (.yaml, .yml, or .json).
        
    Returns
    -------
    dict
        Parsed configuration as a dictionary.
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is not supported or is malformed.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML support requires PyYAML. Install with: pip install pyyaml"
            ) from exc
        
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    elif path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    
    else:
        raise ValueError(
            f"Unsupported configuration file format: {path.suffix}. "
            "Use .yaml, .yml, or .json"
        )


def parse_watcher_config(config: dict[str, Any]) -> dict[str, list[str]]:
    """
    Parse watcher enable/disable configuration.
    
    Parameters
    ----------
    config:
        Configuration dictionary (typically the top-level dict from load_config_file).
        
    Returns
    -------
    dict
        Dictionary with 'enabled' and 'disabled' keys, each containing a list of watcher names.
        
    Examples
    --------
    >>> config = {
    ...     "watchers": {
    ...         "enabled": ["huggingface", "ollama"],
    ...         "disabled": []
    ...     }
    ... }
    >>> parse_watcher_config(config)
    {'enabled': ['huggingface', 'ollama'], 'disabled': []}
    """
    watchers_config = config.get("watchers", {})
    
    if not isinstance(watchers_config, dict):
        return {"enabled": [], "disabled": []}
    
    enabled = watchers_config.get("enabled", [])
    disabled = watchers_config.get("disabled", [])
    
    # Ensure both are lists
    if not isinstance(enabled, list):
        enabled = []
    if not isinstance(disabled, list):
        disabled = []
    
    return {
        "enabled": [str(w).strip() for w in enabled if w],
        "disabled": [str(w).strip() for w in disabled if w],
    }


def resolve_enabled_watchers(
    all_watchers: list[str],
    config_enabled: list[str] | None = None,
    config_disabled: list[str] | None = None,
    cli_enabled: list[str] | None = None,
    cli_disabled: list[str] | None = None,
) -> set[str]:
    """
    Resolve which watchers should be enabled based on config and CLI options.
    
    Priority (highest to lowest):
    1. CLI --disable-watcher (removes from enabled set)
    2. CLI --enable-watcher (adds to enabled set)
    3. Config file watchers.enabled or watchers.disabled
    4. All available watchers (default)
    
    Note: If config_enabled is specified, it acts as a whitelist and config_disabled
    is ignored. If only config_disabled is specified, it acts as a blacklist.
    
    Parameters
    ----------
    all_watchers:
        List of all available watcher names.
    config_enabled:
        List of watchers enabled in config file.
    config_disabled:
        List of watchers disabled in config file.
    cli_enabled:
        List of watchers enabled via CLI flags.
    cli_disabled:
        List of watchers disabled via CLI flags.
        
    Returns
    -------
    set
        Set of watcher names that should be enabled.
        
    Examples
    --------
    >>> all_watchers = ["a", "b", "c", "d"]
    >>> resolve_enabled_watchers(all_watchers, config_disabled=["d"])
    {'a', 'b', 'c'}
    
    >>> resolve_enabled_watchers(
    ...     all_watchers,
    ...     config_enabled=["a", "b"],
    ...     cli_enabled=["c"]
    ... )
    {'a', 'b', 'c'}
    """
    config_enabled = config_enabled or []
    config_disabled = config_disabled or []
    cli_enabled = cli_enabled or []
    cli_disabled = cli_disabled or []
    
    # Start with all available watchers
    enabled: set[str] = set(all_watchers)
    
    # Apply config file settings
    if config_enabled:
        # If config specifies enabled watchers, use only those
        # (config_disabled is ignored when config_enabled is set)
        enabled = set(config_enabled) & set(all_watchers)
    elif config_disabled:
        # Only apply config_disabled if config_enabled is NOT set
        enabled -= set(config_disabled)
    
    # Apply CLI overrides (highest priority)
    if cli_enabled:
        enabled |= set(cli_enabled) & set(all_watchers)
    
    if cli_disabled:
        enabled -= set(cli_disabled)
    
    return enabled


def validate_watcher_names(
    watcher_names: list[str],
    available_watchers: list[str],
) -> tuple[list[str], list[str]]:
    """
    Validate watcher names against available watchers.
    
    Parameters
    ----------
    watcher_names:
        List of watcher names to validate.
    available_watchers:
        List of available (valid) watcher names.
        
    Returns
    -------
    tuple
        (valid_names, invalid_names) where each is a list.
        
    Examples
    --------
    >>> valid, invalid = validate_watcher_names(
    ...     ["huggingface", "invalid_watcher"],
    ...     ["huggingface", "ollama"]
    ... )
    >>> valid
    ['huggingface']
    >>> invalid
    ['invalid_watcher']
    """
    available_set = set(available_watchers)
    valid = [w for w in watcher_names if w in available_set]
    invalid = [w for w in watcher_names if w not in available_set]
    return valid, invalid
