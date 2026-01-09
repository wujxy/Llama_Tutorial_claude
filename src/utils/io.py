"""
Utility module for file I/O operations.

Usage:
    from src.utils.io import save_json, load_json, ensure_dir
    ensure_dir("outputs/checkpoint")
    save_json({"loss": 0.5}, "outputs/metrics.json")
"""

import json
from pathlib import Path
from typing import Any, Dict, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Dictionary to save
        path: Output file path
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        path: Input file path

    Returns:
        Dictionary containing the loaded data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_checkpoint(state_dict: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save a model checkpoint.

    Args:
        state_dict: Model state dictionary
        path: Output file path
    """
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state_dict, path)


def load_checkpoint(path: Union[str, Path], map_location: str = 'cpu') -> Dict[str, Any]:
    """
    Load a model checkpoint.

    Args:
        path: Input file path
        map_location: Device to map the checkpoint to

    Returns:
        State dictionary
    """
    import torch
    return torch.load(path, map_location=map_location, weights_only=False)
