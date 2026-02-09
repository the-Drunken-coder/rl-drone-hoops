from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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
        obs1: Dict[str, List[np.ndarray]] = {}
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=np.bool_)
        infos: List[dict] = []
        for i, env in enumerate(self.envs):
            obs, r, term, trunc, info = env.step(actions[i])
            done = bool(term or trunc)
            rewards[i] = float(r)
            dones[i] = done
            infos.append(info)
            if done:
                obs, _ = env.reset()
            for k, v in obs.items():
                obs1.setdefault(k, []).append(v)
        return StepResult(
            obs={k: np.stack(vs, axis=0) for k, vs in obs1.items()},
            reward=rewards,
            done=dones,
            info=infos,
        )

    def close(self) -> None:
        for e in self.envs:
            try:
                e.close()
            except Exception:
                pass

