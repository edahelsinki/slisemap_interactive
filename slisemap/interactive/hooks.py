"""
    Hooks for connecting to xiplot (using entry points in 'pyproject.toml').
"""
from typing import Any, Callable, Dict
from pandas import DataFrame
from slisemap.interactive.load import slisemap_to_dataframe


def load() -> Dict[str, Callable[[Any], DataFrame]]:
    """Register new loading functions.

    Returns:
        Dictionary of fileendings and parsing functions.
    """
    return {".sm": slisemap_to_dataframe}
