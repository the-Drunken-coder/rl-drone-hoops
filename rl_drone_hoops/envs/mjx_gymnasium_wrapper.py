"""Gymnasium-compatible wrapper around MJX batched physics.

This module wraps :class:`MJXDronePhysics` in a standard Gymnasium ``Env``
interface so that existing training code (PPO, evaluation, video recording) can
use MJX with *zero* changes to the RL loop.

Each wrapper instance manages **one** logical environment backed by a single
slice of the MJX batch.  The :class:`MJXVecAdapter` below manages the full
batch and exposes the same API as :class:`InProcessVecEnv`.

Usage::

    from rl_drone_hoops.envs.mjx_gymnasium_wrapper import MJXDroneHoopsEnv

    env = MJXDroneHoopsEnv(
        image_size=96,
        n_gates=3,
        gate_radius=1.25,
    )
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None  # type: ignore

import gymnasium as gym
from gymnasium import spaces

from rl_drone_hoops.constants import ACTION_DIM
from rl_drone_hoops.envs.mjx_drone_hoops_env import MJXDronePhysics, MJXEnvState


class MJXDroneHoopsEnv(gym.Env):
    """Single-environment Gymnasium wrapper over MJX batched physics.

    Observation and action spaces match :class:`MujocoDroneHoopsEnv` exactly:

    * **Observation** (Dict):
      - ``image``: ``(H, W, 1)`` uint8 – *placeholder black image* (MJX has no
        GPU renderer; use CPU MuJoCo for rendering if needed).
      - ``imu``: ``(T, 6)`` float32.
      - ``last_action``: ``(4,)`` float32.
    * **Action**: ``Box(-1, 1, (4,))`` float32.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        *,
        image_size: int = 96,
        image_rot90: int = 0,
        camera_fps: float = 60.0,
        imu_hz: float = 400.0,
        control_hz: float = 100.0,
        physics_hz: float = 1000.0,
        episode_s: float = 15.0,
        n_gates: int = 5,
        gate_radius: float = 0.75,
        gate_spacing: float = 4.0,
        track_type: str = "straight",
        turn_max_deg: float = 25.0,
        gate_y_range: float = 3.0,
        gate_z_range: Tuple[float, float] = (1.0, 2.0),
        bounds_xyz: Tuple[float, float, float] = (20.0, 20.0, 10.0),
        seed: Optional[int] = None,
        # Ignored MuJoCo-specific kwargs for compatibility
        cam_latency_s: float = 0.03,
        imu_latency_s: float = 0.003,
        act_latency_s: float = 0.01,
        reward_weights: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.image_size = int(image_size)
        self.image_rot90 = int(image_rot90)
        self.n_gates = int(n_gates)
        self.gate_radius = float(gate_radius)
        self._seed_val = seed

        self._physics = MJXDronePhysics(
            num_envs=1,
            image_size=image_size,
            camera_fps=camera_fps,
            imu_hz=imu_hz,
            control_hz=control_hz,
            physics_hz=physics_hz,
            episode_s=episode_s,
            n_gates=n_gates,
            gate_radius=gate_radius,
            gate_spacing=gate_spacing,
            track_type=track_type,
            turn_max_deg=turn_max_deg,
            gate_y_range=gate_y_range,
            gate_z_range=gate_z_range,
            bounds_xyz=bounds_xyz,
            seed=seed,
        )

        imu_window_n = self._physics.imu_window_n

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255,
                    shape=(self.image_size, self.image_size, 1),
                    dtype=np.uint8,
                ),
                "imu": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(imu_window_n, 6),
                    dtype=np.float32,
                ),
                "last_action": spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._state: Optional[MJXEnvState] = None
        self._reset_count = 0

    def _make_obs(self, state: MJXEnvState) -> Dict[str, np.ndarray]:
        """Build observation dict from MJX state (single env, squeeze batch)."""
        raw = self._physics.get_obs_numpy(state)
        # MJX cannot render camera images; provide a black placeholder.
        image = np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
        return {
            "image": image,
            "imu": raw["imu"][0],  # squeeze batch dim
            "last_action": raw["last_action"][0],
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Dict with optional ``fixed_track`` key (ignored in MJX).

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)
        use_seed = seed if seed is not None else (self._seed_val or self._reset_count)
        self._reset_count += 1

        self._state = self._physics.reset(seed=use_seed)
        obs = self._make_obs(self._state)
        info = {"next_gate_idx": int(np.asarray(self._state.next_gate_idx[0]))}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one control step.

        Args:
            action: (4,) normalized action.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Expected action shape ({ACTION_DIM},), got {action.shape}")

        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        # Expand to (1, 4) for batch
        action_batch = jnp.array(action[None, :], dtype=jnp.float32)
        self._state, rewards, dones, terminated_arr, truncated_arr, infos = (
            self._physics.step(self._state, action_batch)
        )

        obs = self._make_obs(self._state)
        reward = float(np.asarray(rewards[0]))
        terminated = bool(np.asarray(terminated_arr[0]))
        truncated = bool(np.asarray(truncated_arr[0]))

        info: Dict[str, Any] = {}
        for k, v in infos.items():
            val = np.asarray(v[0])
            if val.ndim == 0:
                info[k] = val.item()
            else:
                info[k] = val

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Clean up resources."""
        self._state = None

    def render(self) -> np.ndarray:
        """Return placeholder FPV RGB image."""
        return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def render_rgb(self, *, height: int, width: int) -> np.ndarray:
        """Return placeholder RGB image at given resolution."""
        return np.zeros((int(height), int(width), 3), dtype=np.uint8)

    def pose_rpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current drone position and roll/pitch/yaw."""
        if self._state is None:
            return np.zeros(3), np.zeros(3)
        qpos = np.asarray(self._state.mjx_data.qpos[0, :7])
        p = qpos[:3]
        q = qpos[3:7]
        from rl_drone_hoops.utils.math3d import quat_to_mat, mat_to_rpy
        R = quat_to_mat(q)
        rpy = mat_to_rpy(R)
        return p, rpy
