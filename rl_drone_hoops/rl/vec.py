"""Minimal in-process vector env for MuJoCo-based envs (not pickleable)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    obs: Dict[str, np.ndarray]
    reward: np.ndarray
    done: np.ndarray
    info: List[dict]


class InProcessVecEnv:
    """
    Minimal in-process vector env for MuJoCo-based envs (not pickleable).
    """

    def __init__(self, env_fns: List) -> None:
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self, *, seeds: Optional[List[int]] = None, options: Optional[dict] = None) -> Dict[str, np.ndarray]:
        obs0: Dict[str, List[np.ndarray]] = {}
        for i, env in enumerate(self.envs):
            seed = None if seeds is None else seeds[i]
            obs, _info = env.reset(seed=seed, options=options)
            for k, v in obs.items():
                obs0.setdefault(k, []).append(v)
        return {k: np.stack(vs, axis=0) for k, vs in obs0.items()}

    def step(self, actions: np.ndarray) -> StepResult:
        """Execute one step in all environments in parallel.

        Args:
            actions: Array of shape (num_envs, action_dim)

        Returns:
            StepResult with stacked observations, rewards, dones, and infos
        """
        obs1: Dict[str, List[np.ndarray]] = {}
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=np.bool_)
        infos: List[dict] = []
        for i, env in enumerate(self.envs):
            try:
                obs, r, term, trunc, info = env.step(actions[i])
            except Exception as e:
                logger.error(f"Error in step for env {i}: {e}")
                raise
            done = bool(term or trunc)
            rewards[i] = float(r)
            dones[i] = done
            infos.append(info)
            if done:
                try:
                    obs, _ = env.reset()
                except Exception as e:
                    logger.error(f"Error resetting env {i} after done: {e}")
                    raise
            for k, v in obs.items():
                obs1.setdefault(k, []).append(v)
        return StepResult(
            obs={k: np.stack(vs, axis=0) for k, vs in obs1.items()},
            reward=rewards,
            done=dones,
            info=infos,
        )

    def close(self) -> None:
        """Close all environments."""
        for i, e in enumerate(self.envs):
            try:
                e.close()
            except Exception as e:
                logger.warning(f"Error closing env {i}: {e}")

