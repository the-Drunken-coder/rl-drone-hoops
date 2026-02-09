"""Vectorized environment adapter for MJX physics.

Provides :class:`MJXVecAdapter` which exposes the same interface as
:class:`InProcessVecEnv` but delegates all ``N`` environments to **one**
MJX batched simulation.  This avoids the overhead of per-env Python loops.

Usage::

    from rl_drone_hoops.envs.mjx_vec_adapter import MJXVecAdapter

    vec = MJXVecAdapter(num_envs=16, n_gates=3, gate_radius=1.25, seed=0)
    obs = vec.reset()          # dict of (16, ...) arrays
    result = vec.step(actions)  # StepResult
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None  # type: ignore

from rl_drone_hoops.envs.mjx_drone_hoops_env import MJXDronePhysics, MJXEnvState

# Re-use StepResult from the rl.vec module if available, otherwise define a
# compatible dataclass to avoid a transitive torch dependency.
try:
    from rl_drone_hoops.rl.vec import StepResult
except ImportError:
    from dataclasses import dataclass as _dc

    @_dc
    class StepResult:  # type: ignore[no-redef]
        obs: Dict[str, np.ndarray]
        reward: np.ndarray
        done: np.ndarray
        info: List[dict]


class MJXVecAdapter:
    """Vectorized env using a single MJX batch (drop-in for InProcessVecEnv).

    All ``num_envs`` environments share one batched MJX simulation.  Auto-reset
    is handled internally: when an environment reaches ``done``, its slice is
    reset before the next observation is returned.
    """

    def __init__(
        self,
        *,
        num_envs: int = 16,
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
        seed: int = 0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.image_size = int(image_size)

        self._physics = MJXDronePhysics(
            num_envs=num_envs,
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

        self._seed = int(seed)
        self._state: Optional[MJXEnvState] = None
        self._reset_counter = 0

    def _make_obs(self, state: MJXEnvState) -> Dict[str, np.ndarray]:
        """Extract stacked observations from batched state."""
        raw = self._physics.get_obs_numpy(state)
        # Provide black placeholder images (MJX has no camera renderer)
        image = np.zeros(
            (self.num_envs, self.image_size, self.image_size, 1), dtype=np.uint8
        )
        return {
            "image": image,
            "imu": raw["imu"],
            "last_action": raw["last_action"],
        }

    def reset(
        self,
        *,
        seeds: Optional[List[int]] = None,
        options: Optional[dict] = None,
    ) -> Dict[str, np.ndarray]:
        """Reset all environments.

        Args:
            seeds: Per-env seeds (only the first is used for track generation;
                   the rest are derived).
            options: Passed through (currently unused for MJX).

        Returns:
            Stacked observation dict ``{key: (num_envs, ...) ndarray}``.
        """
        seed = seeds[0] if seeds else self._seed
        self._reset_counter += 1
        self._state = self._physics.reset(seed=seed)
        return self._make_obs(self._state)

    def step(self, actions: np.ndarray) -> StepResult:
        """Step all environments.

        Args:
            actions: ``(num_envs, 4)`` normalized actions.

        Returns:
            :class:`StepResult` with stacked obs, rewards, dones, infos.
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        jax_actions = jnp.array(actions, dtype=jnp.float32)
        self._state, rewards_j, dones_j, term_j, trunc_j, infos_j = self._physics.step(
            self._state, jax_actions
        )

        rewards = np.asarray(rewards_j).astype(np.float32)
        dones = np.asarray(dones_j)

        # Build per-env info dicts
        infos: List[dict] = []
        for i in range(self.num_envs):
            info: Dict[str, Any] = {}
            for k, v in infos_j.items():
                val = np.asarray(v[i])
                info[k] = val.item() if val.ndim == 0 else val
            infos.append(info)

        # Auto-reset done environments (selective per-env reset)
        done_mask = np.asarray(dones)
        if done_mask.any():
            self._reset_counter += 1
            # Use a distinct seed per reset to avoid repeating the same track.
            reset_seed = self._seed + self._reset_counter * self.num_envs
            new_state = self._physics.reset(seed=reset_seed)

            # Selectively replace only done envs with fresh state
            import jax

            def _select(old, new):
                mask = jnp.array(done_mask)
                for _ in range(old.ndim - 1):
                    mask = mask[..., None]
                return jnp.where(mask, new, old)

            self._state = jax.tree.map(_select, self._state, new_state)

        obs = self._make_obs(self._state)
        return StepResult(obs=obs, reward=rewards, done=dones, info=infos)

    def close(self) -> None:
        """Clean up resources."""
        self._state = None
