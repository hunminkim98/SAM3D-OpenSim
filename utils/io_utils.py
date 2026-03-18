"""I/O utilities for configuration and data files."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml

from sam3d_opensim.config import default_marker_mapping_path, load_pipeline_config


def load_config(config_path: str = None) -> dict:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to config file. If None, loads default config.

    Returns:
        Configuration dictionary
    """
    return load_pipeline_config(config_path)


def load_marker_mapping(mapping_path: str = None) -> dict:
    """
    Load MHR70 to OpenSim marker mapping from YAML file.

    Args:
        mapping_path: Path to mapping file. If None, loads default mapping.

    Returns:
        Marker mapping dictionary
    """
    if mapping_path is None:
        mapping_path = default_marker_mapping_path()

    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Marker mapping file not found: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)

    return mapping


def save_json(data: Any, output_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file with numpy array support.

    Args:
        data: Data to save
        output_path: Output file path
        indent: JSON indentation level
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return super().default(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)


def load_json(json_path: str) -> Any:
    """
    Load data from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Loaded data
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir(video_path: str, base_output_dir: Optional[str] = None) -> Path:
    """
    Generate output directory name based on video name and timestamp.

    Args:
        video_path: Path to input video
        base_output_dir: Base output directory (default: current directory)

    Returns:
        Output directory path
    """
    from datetime import datetime

    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"output_{timestamp}_{video_name}"

    if base_output_dir:
        return Path(base_output_dir) / output_name
    else:
        return Path(output_name)


def save_pickle(data: Any, output_path: str) -> None:
    """
    Save data to pickle file.

    Args:
        data: Data to save
        output_path: Output file path
    """
    import pickle

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(pickle_path: str) -> Any:
    """
    Load data from pickle file.

    Args:
        pickle_path: Path to pickle file

    Returns:
        Loaded data
    """
    import pickle

    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """
    Recursively merge two configuration dictionaries.
    Values in override_config take precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        elif value is not None:
            result[key] = value

    return result
