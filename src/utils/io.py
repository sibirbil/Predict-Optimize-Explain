"""
Input/Output utilities for JSON, file operations, and hashing.
"""
import json
import hashlib
import shutil
import numpy as np
from pathlib import Path
from typing import Any, List


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(path: Path, obj: Any, convert_np: bool = True) -> None:
    """
    Save a Python object to JSON file with pretty printing.

    Args:
        path: Path to save JSON file
        obj: Python object to serialize (must be JSON-serializable)
        convert_np: Whether to convert numpy types (default: True)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        if convert_np:
            json.dump(obj, f, indent=2, cls=NumpyEncoder)
        else:
            json.dump(obj, f, indent=2)


def load_json(path: Path) -> Any:
    """
    Load a Python object from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Deserialized Python object
    """
    with open(path, 'r') as f:
        return json.load(f)


def sha1_of_list(xs: List[str]) -> str:
    """
    Compute SHA1 hash of a list of strings.

    Useful for generating unique identifiers for feature lists or configurations.

    Args:
        xs: List of strings to hash

    Returns:
        SHA1 hexadecimal digest string
    """
    joined = '|'.join(xs)
    return hashlib.sha1(joined.encode('utf-8')).hexdigest()


def safe_rm_tree(path: Path) -> None:
    """
    Safely remove a directory tree if it exists.

    Args:
        path: Directory path to remove
    """
    path = Path(path)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
