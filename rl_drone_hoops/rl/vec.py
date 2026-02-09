"""Minimal in-process vector env for MuJoCo-based envs (not pickleable)."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    obs: Dict[str, np.ndarray]
    reward: np.ndarray
    done: np.ndarray
    info: List[dict]


def _subproc_worker(remote, parent_remote, env_cls: Type, env_kwargs: dict):
    parent_remote.close()
    try:
        env = env_cls(**env_kwargs)
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                seed, options = data
                obs, _info = env.reset(seed=seed, options=options)
                remote.send(obs)
            elif cmd == "step":
                action = data
                obs, r, term, trunc, info = env.step(action)
                done = bool(term or trunc)
                if done:
                    # Auto-reset to preserve current InProcessVecEnv semantics.
                    obs, _ = env.reset()
                remote.send((obs, float(r), done, info))
            elif cmd == "close":
                try:
                    env.close()
                except Exception:
                    pass
                remote.close()
                break
            else:
                raise RuntimeError(f"unknown cmd: {cmd}")
    except BaseException as e:
        # Send a simple error payload; exceptions themselves can be non-pickleable.
        try:
            remote.send(("__error__", repr(e)))
        except Exception:
            pass
        try:
            remote.close()
        except Exception:
            pass


class SubprocVecEnv:
    """
    Minimal subprocess-based vec env for CPU throughput.

    Runs each env in its own process and communicates via Pipe.
    This enables true parallel stepping across CPU cores on Windows/Linux.
    """

    def __init__(self, env_cls: Type, env_kwargs_list: List[dict]) -> None:
        self.num_envs = len(env_kwargs_list)
        if self.num_envs < 1:
            raise ValueError("env_kwargs_list must be non-empty")

        # Prefer spawn to avoid MuJoCo + fork hazards.
        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._closed = False

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(self.num_envs)])
        self.procs = []
        for wr, r, kwargs in zip(self.work_remotes, self.remotes, env_kwargs_list):
            p = ctx.Process(target=_subproc_worker, args=(wr, r, env_cls, kwargs), daemon=True)
            p.start()
            wr.close()
            self.procs.append(p)

        self._obs_buf: Optional[Dict[str, np.ndarray]] = None

    def reset(self, *, seeds: Optional[List[int]] = None, options: Optional[dict] = None) -> Dict[str, np.ndarray]:
        if self._closed:
            raise RuntimeError("SubprocVecEnv is closed")
        for i, remote in enumerate(self.remotes):
            seed = None if seeds is None else seeds[i]
            remote.send(("reset", (seed, options)))

        obs_out: Optional[Dict[str, np.ndarray]] = None
        for i, remote in enumerate(self.remotes):
            obs = remote.recv()
            if isinstance(obs, tuple) and len(obs) == 2 and obs[0] == "__error__":
                raise RuntimeError(f"worker reset failed: {obs[1]}")
            if obs_out is None:
                obs_out = {k: np.empty((self.num_envs, *v.shape), dtype=v.dtype) for k, v in obs.items()}
            for k, v in obs.items():
                obs_out[k][i] = v

        assert obs_out is not None
        self._obs_buf = obs_out
        return obs_out

    def step(self, actions: np.ndarray) -> StepResult:
        if self._closed:
            raise RuntimeError("SubprocVecEnv is closed")
        if self._obs_buf is None:
            _ = self.reset()
        assert self._obs_buf is not None

        for i, remote in enumerate(self.remotes):
            remote.send(("step", actions[i]))

        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=np.bool_)
        infos: List[dict] = []
        for i, remote in enumerate(self.remotes):
            res = remote.recv()
            if isinstance(res, tuple) and len(res) == 2 and res[0] == "__error__":
                raise RuntimeError(f"worker step failed: {res[1]}")
            obs, r, done, info = res
            rewards[i] = float(r)
            dones[i] = bool(done)
            infos.append(info)
            for k, v in obs.items():
                self._obs_buf[k][i] = v

        return StepResult(obs=self._obs_buf, reward=rewards, done=dones, info=infos)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            try:
                p.join(timeout=2.0)
            except Exception:
                pass


class InProcessVecEnv:
    """
    Minimal in-process vector env for MuJoCo-based envs (not pickleable).
    """

    def __init__(self, env_fns: List) -> None:
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self._obs_buf: Optional[Dict[str, np.ndarray]] = None

    def reset(self, *, seeds: Optional[List[int]] = None, options: Optional[dict] = None) -> Dict[str, np.ndarray]:
        # Allocate a stable output buffer once and refill it in-place to avoid per-step
        # list building + np.stack overhead.
        obs_out: Optional[Dict[str, np.ndarray]] = None
        for i, env in enumerate(self.envs):
            seed = None if seeds is None else seeds[i]
            obs, _info = env.reset(seed=seed, options=options)
            if obs_out is None:
                obs_out = {k: np.empty((self.num_envs, *v.shape), dtype=v.dtype) for k, v in obs.items()}
            for k, v in obs.items():
                obs_out[k][i] = v

        assert obs_out is not None
        self._obs_buf = obs_out
        return obs_out

    def step(self, actions: np.ndarray) -> StepResult:
        """Execute one step in all environments in parallel.

        Args:
            actions: Array of shape (num_envs, action_dim)

        Returns:
            StepResult with stacked observations, rewards, dones, and infos
        """
        if self._obs_buf is None:
            # Defensive: allow step() without an explicit reset() call.
            _ = self.reset()
        assert self._obs_buf is not None

        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=np.bool_)
        infos: List[dict] = []
        for i, env in enumerate(self.envs):
            try:
                obs, r, term, trunc, info = env.step(actions[i])
            except Exception:
                logger.exception("Error in step for env %s", i)
                raise
            done = bool(term or trunc)
            rewards[i] = float(r)
            dones[i] = done
            infos.append(info)
            if done:
                try:
                    obs, _ = env.reset()
                except Exception:
                    logger.exception("Error resetting env %s after done", i)
                    raise
            for k, v in obs.items():
                self._obs_buf[k][i] = v
        return StepResult(
            # Note: returns a reused buffer that is overwritten on subsequent calls.
            obs=self._obs_buf,
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

