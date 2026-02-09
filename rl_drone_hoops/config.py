"""Configuration management for RL Drone Hoops training."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def _get_default_config_path() -> Path:
    """Get path to default config file."""
    return Path(__file__).parent.parent / "config" / "default.yaml"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML config file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML as dictionary

    Raises:
        ImportError: If PyYAML is not installed
        FileNotFoundError: If config file doesn't exist
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required to use YAML config files. "
            "Install with: pip install pyyaml"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loaded config from {path}")
    return cfg if cfg is not None else {}


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load JSON config file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON as dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        cfg = json.load(f)

    logger.info(f"Loaded config from {path}")
    return cfg


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load config from file (YAML or JSON).

    If no path specified, uses default config.

    Args:
        path: Path to config file (auto-detects format from extension)

    Returns:
        Configuration dictionary
    """
    if path is None:
        path = _get_default_config_path()

    path = Path(path)

    if path.suffix.lower() in (".yaml", ".yml"):
        return load_yaml(path)
    elif path.suffix.lower() == ".json":
        return load_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override config into base config.

    Override values take precedence. Handles nested dicts.

    Args:
        base: Base configuration
        overrides: Overrides to apply

    Returns:
        Merged configuration
    """
    result = dict(base)

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def extract_ppo_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract PPO-specific config values.

    Flattens nested config structure for PPOConfig initialization.

    Args:
        cfg: Full configuration dictionary

    Returns:
        Dict with keys matching PPOConfig fields
    """
    ppo_cfg = cfg.get("ppo", {})
    env_cfg = cfg.get("environment", {})
    track_cfg = cfg.get("track", {})
    eval_cfg = cfg.get("evaluation", {})

    return {
        # PPO params
        "seed": ppo_cfg.get("seed", 0),
        "device": ppo_cfg.get("device", "auto"),
        "num_envs": ppo_cfg.get("num_envs", 4),
        "total_steps": ppo_cfg.get("total_steps", 200_000),
        "rollout_steps": ppo_cfg.get("rollout_steps", 128),
        "gamma": ppo_cfg.get("gamma", 0.99),
        "gae_lambda": ppo_cfg.get("gae_lambda", 0.95),
        "clip_coef": ppo_cfg.get("clip_coef", 0.2),
        "vf_coef": ppo_cfg.get("vf_coef", 0.5),
        "ent_coef": ppo_cfg.get("ent_coef", 0.01),
        "max_grad_norm": ppo_cfg.get("max_grad_norm", 0.5),
        "lr": ppo_cfg.get("lr", 3e-4),
        "adam_eps": ppo_cfg.get("adam_eps", 1e-5),
        "update_epochs": ppo_cfg.get("update_epochs", 4),
        "minibatch_envs": ppo_cfg.get("minibatch_envs", 4),

        # Environment params
        "image_size": env_cfg.get("image_size", 96),
        "image_rot90": env_cfg.get("image_rot90", 0),
        "camera_fps": env_cfg.get("camera_fps", 60.0),
        "imu_hz": env_cfg.get("imu_hz", 400.0),
        "control_hz": env_cfg.get("control_hz", 100.0),
        "physics_hz": env_cfg.get("physics_hz", 1000.0),

        # Track params
        "track_type": track_cfg.get("track_type", "straight"),
        "gate_radius": track_cfg.get("gate_radius", 1.25),
        "turn_max_deg": track_cfg.get("turn_max_deg", 20.0),
        "n_gates": track_cfg.get("n_gates", 3),
        "episode_s": track_cfg.get("episode_duration_s", 12.0),

        # Evaluation params
        "eval_every_steps": eval_cfg.get("eval_every_steps", 50_000),
        "eval_episodes": eval_cfg.get("eval_episodes", 3),
    }


def extract_curriculum(cfg: Dict[str, Any]) -> list[tuple[int, Dict[str, Any]]]:
    """Extract curriculum stages from config.

    Args:
        cfg: Full configuration dictionary

    Returns:
        List of (step_threshold, env_overrides) tuples
    """
    curr_cfg = cfg.get("curriculum", {})
    stages_list = curr_cfg.get("stages", [])

    curriculum = []
    for stage in stages_list:
        step = stage.get("step", 0)
        overrides = stage.get("overrides", {})
        curriculum.append((step, overrides))

    # Sort by step number
    curriculum.sort(key=lambda x: x[0])

    return curriculum


def extract_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model architecture config.

    Args:
        cfg: Full configuration dictionary

    Returns:
        Dict with model architecture parameters
    """
    model_cfg = cfg.get("model", {})

    return {
        "cnn_channels": model_cfg.get("cnn_channels", [16, 32, 64]),
        "cnn_kernel": model_cfg.get("cnn_kernel", 3),
        "cnn_stride": model_cfg.get("cnn_stride", 2),
        "imu_hidden": model_cfg.get("imu_hidden", 64),
        "rnn_type": model_cfg.get("rnn_type", "gru"),
        "rnn_hidden": model_cfg.get("rnn_hidden", 256),
        "actor_hidden": model_cfg.get("actor_hidden", 256),
        "critic_hidden": model_cfg.get("critic_hidden", 256),
    }


def extract_video_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract video rendering config.

    Args:
        cfg: Full configuration dictionary

    Returns:
        Dict with video rendering parameters
    """
    video_cfg = cfg.get("video", {})

    return {
        "enabled": video_cfg.get("enabled", True),
        "size": video_cfg.get("size", 256),
        "fps": video_cfg.get("fps", 30),
        "overlay": video_cfg.get("overlay", True),
        "record_first_episode_only": video_cfg.get("record_first_episode_only", True),
        "quality": video_cfg.get("quality", "high"),
        "codec": video_cfg.get("codec", "h264"),
    }


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        cfg: Configuration dictionary
        path: Output path (YAML or JSON based on extension)
    """
    if yaml is None and str(path).endswith((".yaml", ".yml")):
        raise ImportError(
            "PyYAML is required to save YAML config. "
            "Install with: pip install pyyaml"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() in (".yaml", ".yml"):
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    elif path.suffix.lower() == ".json":
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    logger.info(f"Saved config to {path}")
