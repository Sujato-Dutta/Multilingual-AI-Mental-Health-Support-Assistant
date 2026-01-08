"""
Seed Utility Module

Provides functions for setting deterministic seeds across all libraries
to ensure reproducible behavior.
"""

import os
import random
import hashlib
from typing import Optional

import numpy as np


def set_all_seeds(seed: int = 42) -> None:
    """
    Set seeds for all random number generators to ensure reproducibility.
    
    Args:
        seed: The seed value to use across all libraries.
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # Transformers (if available)
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass


def compute_config_hash(config_dict: dict) -> str:
    """
    Compute a hash of configuration dictionary for logging.
    
    Args:
        config_dict: Configuration dictionary to hash.
        
    Returns:
        SHA256 hash of the configuration.
    """
    config_str = str(sorted(config_dict.items()))
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_file_hash(filepath: str) -> Optional[str]:
    """
    Compute SHA256 hash of a file.
    
    Args:
        filepath: Path to the file.
        
    Returns:
        SHA256 hash of the file, or None if file doesn't exist.
    """
    if not os.path.exists(filepath):
        return None
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:16]


def get_model_version_info(model_path: str) -> dict:
    """
    Get version information for a model or adapter.
    
    Args:
        model_path: Path to the model or adapter directory.
        
    Returns:
        Dictionary with version information.
    """
    info = {
        "path": model_path,
        "exists": os.path.exists(model_path),
        "hash": None,
        "files": []
    }
    
    if info["exists"] and os.path.isdir(model_path):
        # List relevant files
        for filename in os.listdir(model_path):
            if filename.endswith((".bin", ".safetensors", ".json")):
                info["files"].append(filename)
                
        # Hash the config if available
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            info["hash"] = compute_file_hash(config_path)
    
    return info
