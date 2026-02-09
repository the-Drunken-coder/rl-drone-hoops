"""MJX (MuJoCo XLA) batched drone hoops physics engine.

This module provides GPU-accelerated batched physics simulation for the drone
racing environment using MuJoCo's JAX backend (MJX).  It mirrors the dynamics
of :class:`MujocoDroneHoopsEnv` but operates on batched JAX arrays, enabling
1000+ parallel simulations on a single GPU.

Key design choices
------------------
* **Architecture A (bridge at episode boundaries):**  Physics runs in JAX;
  observations are converted to NumPy/PyTorch at each ``step()`` call.
* **Batch organisation A (one JAX batch = one RL env):**  ``num_envs`` parallel
  worlds are vmapped.  The existing PPO training loop is unchanged.
* **Sensor pipeline in PyTorch:**  Camera rendering and IMU encoding remain in
  the PyTorch model (``SmallCNN`` / ``IMUEncoder``).

Usage::

    from rl_drone_hoops.envs.mjx_drone_hoops_env import MJXDronePhysics

    phys = MJXDronePhysics(num_envs=16, n_gates=3, gate_radius=1.25)
    state = phys.reset(seed=42)
    actions = jnp.zeros((16, 4))
    state, obs, rewards, dones, infos = phys.step(state, actions)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from jax import random as jrandom
except ImportError as _e:
    jax = None  # type: ignore
    jnp = None  # type: ignore
    jrandom = None  # type: ignore
    _JAX_IMPORT_ERR = _e

try:
    import mujoco
    import mujoco.mjx as mjx
except ImportError as _e:
    mujoco = None  # type: ignore
    mjx = None  # type: ignore
    _MUJOCO_IMPORT_ERR = _e

from rl_drone_hoops.constants import (
    ACTION_DIM,
    EPSILON,
    MAX_TILT_DEG,
)
from rl_drone_hoops.mujoco_assets.xml_builder import build_drone_hoops_xml
from rl_drone_hoops.utils.math3d import quat_from_two_vectors, unit


# ---------------------------------------------------------------------------
# Gate dataclass (plain NumPy – used only at track-building time)
# ---------------------------------------------------------------------------
@dataclass
class GateSpec:
    """Gate specification for track building (CPU-side, NumPy)."""

    center: np.ndarray  # (3,)
    normal: np.ndarray  # (3,) unit
    radius: float


# ---------------------------------------------------------------------------
# JAX-side state carried between steps
# ---------------------------------------------------------------------------
@dataclass
class MJXEnvState:
    """Per-environment state carried between ``step()`` calls.

    All arrays have a leading batch dimension ``(B, ...)``.
    """

    mjx_data: Any  # batched mjx.Data pytree  (B, ...)
    t: Any  # (B,) float32 – simulation time
    step_i: Any  # (B,) int32 – control step counter
    next_gate_idx: Any  # (B,) int32
    p_prev: Any  # (B, 3) float64 – previous drone position
    v_prev: Any  # (B, 3) float64 – previous drone velocity (for IMU)
    last_action_norm: Any  # (B, 4) float32
    thrust_state: Any  # (B,) float32 – 1st-order lagged thrust
    applied_action: Any  # (B, 4) float64 – currently applied (scaled) action
    imu_history: Any  # (B, imu_window_n, 6) float32
    gate_centers: Any  # (B, n_gates, 3) float64
    gate_normals: Any  # (B, n_gates, 3) float64
    gate_radii: Any  # (B, n_gates) float64
    rng_key: Any  # (B, 2) JAX PRNG key


# Register MJXEnvState as a JAX pytree so it can be used with jit/vmap.
if jax is not None:
    jax.tree_util.register_dataclass(
        MJXEnvState,
        data_fields=[
            "mjx_data", "t", "step_i", "next_gate_idx", "p_prev", "v_prev",
            "last_action_norm", "thrust_state", "applied_action", "imu_history",
            "gate_centers", "gate_normals", "gate_radii", "rng_key",
        ],
        meta_fields=[],
    )


# ---------------------------------------------------------------------------
# MJX batched physics engine
# ---------------------------------------------------------------------------
class MJXDronePhysics:
    """Batched drone-hoops physics using MuJoCo MJX.

    Parameters mirror :class:`MujocoDroneHoopsEnv` for drop-in compatibility.
    """

    def __init__(
        self,
        *,
        num_envs: int = 16,
        image_size: int = 96,
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
    ) -> None:
        if jax is None:
            raise ImportError(
                "JAX is required for MJXDronePhysics. "
                f"Install with `pip install jax jaxlib mujoco-mjx`.\n"
                f"Original error: {_JAX_IMPORT_ERR}"
            )
        if mujoco is None:
            raise ImportError(
                "MuJoCo is required for MJXDronePhysics. "
                f"Install with `pip install mujoco mujoco-mjx`.\n"
                f"Original error: {_MUJOCO_IMPORT_ERR}"
            )

        self.num_envs = int(num_envs)
        self.image_size = int(image_size)
        self.camera_fps = float(camera_fps)
        self.imu_hz = float(imu_hz)
        self.control_hz = float(control_hz)
        self.physics_hz = float(physics_hz)

        self.dt_phys = 1.0 / self.physics_hz
        self.dt_control = 1.0 / self.control_hz
        self.n_substeps = max(1, int(round(self.dt_control / self.dt_phys)))

        self.episode_s = float(episode_s)
        self.max_steps = int(np.ceil(self.episode_s * self.control_hz))

        self.n_gates = int(n_gates)
        self.gate_radius = float(gate_radius)
        self.gate_spacing = float(gate_spacing)
        self.track_type = str(track_type)
        self.turn_max_deg = float(turn_max_deg)
        self.gate_y_range = float(gate_y_range)
        self.gate_z_range = (float(gate_z_range[0]), float(gate_z_range[1]))
        self.bounds_xyz = np.array(bounds_xyz, dtype=np.float64)

        # Dynamics (must match MujocoDroneHoopsEnv)
        self.max_rates = np.array([4.0, 4.0, 6.0], dtype=np.float64)
        self.thrust_min = 0.0
        self.thrust_max = 25.0
        self.rate_kp = np.array([0.025, 0.025, 0.02], dtype=np.float64)
        self.torque_max = np.array([0.35, 0.35, 0.2], dtype=np.float64)
        self.lin_drag = 0.15
        self.quad_drag = 0.04
        self.ang_drag = 0.05
        self.thrust_tau = 0.05

        # Reward weights (must match MujocoDroneHoopsEnv)
        self.r_alive = 0.1
        self.r_gate = 10.0
        self.k_progress = 1.5
        self.k_center = 0.5
        self.k_speed = 0.02
        self.k_smooth = 0.02
        self.k_tilt = 0.05
        self.k_angrate = 0.0002
        self.r_crash = -20.0

        # IMU history window
        self.imu_window_s = 0.10
        self.imu_window_n = int(np.ceil(self.imu_hz * self.imu_window_s))

        # Build MuJoCo model and convert to MJX
        xml_max_gates = max(16, self.n_gates)
        xml = build_drone_hoops_xml(
            max_gates=xml_max_gates,
            ring_radius=self.gate_radius,
            tube_radius=0.05,
            ring_segments=16,
        )
        self._mj_model = mujoco.MjModel.from_xml_string(xml)
        self._mj_data = mujoco.MjData(self._mj_model)
        self._mjx_model = mjx.put_model(self._mj_model)

        # Drone mass for hover thrust computation
        from rl_drone_hoops.constants import XML_DRONE_BODY_NAME

        drone_bid = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, XML_DRONE_BODY_NAME
        )
        self._drone_bid = int(drone_bid)
        self._drone_mass = float(self._mj_model.body_mass[drone_bid])
        self._hover_thrust = float(
            np.clip(self._drone_mass * 9.81, self.thrust_min, self.thrust_max)
        )
        self._hover_thrust_norm = float(
            2.0 * (self._hover_thrust - self.thrust_min)
            / max(self.thrust_max - self.thrust_min, 1e-9)
            - 1.0
        )

        # Pre-compute JAX constants
        self._jax_max_rates = jnp.array(self.max_rates, dtype=jnp.float32)
        self._jax_rate_kp = jnp.array(self.rate_kp, dtype=jnp.float32)
        self._jax_torque_max = jnp.array(self.torque_max, dtype=jnp.float32)
        self._jax_bounds = jnp.array(self.bounds_xyz, dtype=jnp.float32)
        self._jax_max_tilt = float(np.deg2rad(MAX_TILT_DEG))

        self._seed = seed if seed is not None else 0

        # JIT-compile the batched step function
        self._jit_step = jax.jit(self._batched_step)
        self._jit_reset = jax.jit(self._batched_reset)

    # ------------------------------------------------------------------
    # Track generation (CPU-side, then broadcast to JAX)
    # ------------------------------------------------------------------
    def _build_track_np(self, rng: np.random.Generator) -> list[GateSpec]:
        """Build a track using NumPy (CPU), identical logic to MujocoDroneHoopsEnv."""
        gates = []
        if self.track_type == "straight":
            for i in range(self.n_gates):
                x = (i + 1) * self.gate_spacing
                y = rng.uniform(-min(1.0, self.gate_y_range), min(1.0, self.gate_y_range))
                z = rng.uniform(*self.gate_z_range)
                center = np.array([x, y, z], dtype=np.float64)
                normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                gates.append(GateSpec(center=center, normal=unit(normal), radius=self.gate_radius))
        elif self.track_type == "random_turns":
            pos = np.array(
                [self.gate_spacing, 0.0, float(np.mean(self.gate_z_range))],
                dtype=np.float64,
            )
            dir_vec = unit(np.array([1.0, 0.0, 0.0], dtype=np.float64))
            max_turn = np.deg2rad(self.turn_max_deg)
            for i in range(self.n_gates):
                yaw = rng.uniform(-max_turn, max_turn)
                pitch = rng.uniform(-0.35 * max_turn, 0.35 * max_turn)
                cy, sy = np.cos(yaw), np.sin(yaw)
                cp, sp = np.cos(pitch), np.sin(pitch)
                Rz = np.array(
                    [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
                )
                Ry = np.array(
                    [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64
                )
                dir_vec = unit(Rz @ (Ry @ dir_vec))
                candidate = pos + dir_vec * self.gate_spacing
                candidate[1] += rng.uniform(-self.gate_y_range, self.gate_y_range)
                candidate[2] = rng.uniform(*self.gate_z_range)
                candidate[0] = np.clip(
                    candidate[0],
                    -self.bounds_xyz[0] + 1.0,
                    self.bounds_xyz[0] - 1.0,
                )
                candidate[1] = np.clip(
                    candidate[1],
                    -self.bounds_xyz[1] + 1.0,
                    self.bounds_xyz[1] - 1.0,
                )
                candidate[2] = np.clip(candidate[2], 0.5, self.bounds_xyz[2] - 0.5)
                pos = candidate
                gates.append(
                    GateSpec(center=pos.copy(), normal=dir_vec.copy(), radius=self.gate_radius)
                )
        else:
            raise ValueError(
                f"Unknown track_type={self.track_type!r}. Expected 'straight' or 'random_turns'."
            )
        return gates

    def _gates_to_jax(
        self, gates_list: list[list[GateSpec]]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Convert per-env gate lists to batched JAX arrays.

        Args:
            gates_list: List of length ``num_envs``, each containing ``n_gates`` GateSpec.

        Returns:
            (gate_centers, gate_normals, gate_radii) each with leading batch dim.
        """
        B = len(gates_list)
        centers = np.zeros((B, self.n_gates, 3), dtype=np.float64)
        normals = np.zeros((B, self.n_gates, 3), dtype=np.float64)
        radii = np.zeros((B, self.n_gates), dtype=np.float64)
        for b, glist in enumerate(gates_list):
            for g_idx, g in enumerate(glist):
                centers[b, g_idx] = g.center
                normals[b, g_idx] = g.normal
                radii[b, g_idx] = g.radius
        return jnp.array(centers), jnp.array(normals), jnp.array(radii)

    # ------------------------------------------------------------------
    # JAX pure-function physics helpers (vmappable)
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_to_mat_jax(q: jnp.ndarray) -> jnp.ndarray:
        """Quaternion (w,x,y,z) → 3×3 rotation matrix, JAX version."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        return jnp.array(
            [
                [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
            ]
        )

    @staticmethod
    def _mat_to_rpy_jax(R: jnp.ndarray) -> jnp.ndarray:
        """Rotation matrix → roll/pitch/yaw (aerospace convention), JAX version."""
        sy = jnp.clip(-R[2, 0], -1.0, 1.0)
        pitch = jnp.arcsin(sy)
        roll = jnp.arctan2(R[2, 1], R[2, 2])
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        return jnp.array([roll, pitch, yaw])

    def _scale_action_jax(self, a_norm: jnp.ndarray) -> jnp.ndarray:
        """Scale normalized action [-1,1] to physical rates/thrust."""
        a = jnp.clip(a_norm, -1.0, 1.0)
        rates = a[:3] * self._jax_max_rates
        thrust = (a[3] + 1.0) * 0.5 * (self.thrust_max - self.thrust_min) + self.thrust_min
        return jnp.concatenate([rates, jnp.array([thrust])])

    def _apply_forces_single(
        self,
        mjx_data: Any,
        applied_action: jnp.ndarray,
        thrust_state: jnp.ndarray,
        v_prev: jnp.ndarray,
    ) -> Tuple[Any, jnp.ndarray, jnp.ndarray]:
        """Apply forces and step physics for one substep (single env, no batch dim)."""
        qpos = mjx_data.qpos[:7]
        qvel = mjx_data.qvel[:6]
        q = qpos[3:7]
        R = self._quat_to_mat_jax(q)
        v_world = qvel[:3]
        w_world = qvel[3:6]
        w_body = R.T @ w_world

        w_cmd = applied_action[:3]
        thrust_cmd = applied_action[3]

        # Thrust lag
        alpha = jnp.float32(self.dt_phys / max(self.thrust_tau, 1e-6))
        alpha = jnp.clip(alpha, 0.0, 1.0)
        new_thrust_state = (1.0 - alpha) * thrust_state + alpha * thrust_cmd

        # Rate PD control → body torque → world torque
        w_err = w_cmd - w_body
        tau_body = jnp.clip(
            self._jax_rate_kp * w_err, -self._jax_torque_max, self._jax_torque_max
        )
        tau_world = R @ tau_body

        # Thrust along body +Z
        f_world = R @ jnp.array([0.0, 0.0, new_thrust_state])

        # Drag
        speed = jnp.linalg.norm(v_world)
        f_drag = -self.lin_drag * v_world - self.quad_drag * speed * v_world
        tau_drag = -self.ang_drag * w_world

        f_total = f_world + f_drag
        tau_total = tau_world + tau_drag

        # Build qfrc_applied: for a freejoint body, the generalized forces are
        # [fx, fy, fz, tx, ty, tz] in world frame.
        qfrc = jnp.zeros_like(mjx_data.qfrc_applied)
        qfrc = qfrc.at[:3].set(f_total)
        qfrc = qfrc.at[3:6].set(tau_total)
        mjx_data = mjx_data.replace(qfrc_applied=qfrc)

        # Step physics
        mjx_data = mjx.step(self._mjx_model, mjx_data)

        # Update v_prev for IMU
        new_v_prev = mjx_data.qvel[:3]

        return mjx_data, new_thrust_state, new_v_prev

    def _compute_imu_jax(
        self, mjx_data: Any, v_prev: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute one IMU sample (single env, no batch dim)."""
        qpos = mjx_data.qpos[:7]
        qvel = mjx_data.qvel[:6]
        q = qpos[3:7]
        R = self._quat_to_mat_jax(q)
        v_world = qvel[:3]
        w_world = qvel[3:6]
        w_body = R.T @ w_world

        a_world = (v_world - v_prev) / jnp.float32(max(self.dt_phys, 1e-9))
        g = jnp.array([0.0, 0.0, -9.81])
        a_body = R.T @ (a_world - g)

        return jnp.concatenate([w_body, a_body])

    def _gate_crossed_jax(
        self,
        gate_center: jnp.ndarray,
        gate_normal: jnp.ndarray,
        gate_radius: jnp.ndarray,
        p_prev: jnp.ndarray,
        p_curr: jnp.ndarray,
    ) -> jnp.ndarray:
        """Check if drone crossed through a single gate (scalar output, no batch)."""
        s_prev = jnp.dot(gate_normal, p_prev - gate_center)
        s_curr = jnp.dot(gate_normal, p_curr - gate_center)

        crossed = ((s_prev < 0.0) & (s_curr >= 0.0)) | ((s_prev >= 0.0) & (s_curr < 0.0))

        denom = s_curr - s_prev
        safe_denom = jnp.where(jnp.abs(denom) < EPSILON, 1.0, denom)
        t_interp = jnp.clip(-s_prev / safe_denom, 0.0, 1.0)

        p_cross = p_prev + t_interp * (p_curr - p_prev)
        d = p_cross - gate_center
        d_perp = d - jnp.dot(d, gate_normal) * gate_normal
        within_radius = jnp.linalg.norm(d_perp) <= gate_radius

        denom_ok = jnp.abs(denom) >= EPSILON
        return crossed & within_radius & denom_ok

    def _step_single_env(
        self,
        state_tuple: Tuple,
        action: jnp.ndarray,
    ) -> Tuple:
        """Step a single environment (used inside vmap).

        Takes and returns flat tuples for vmappability.
        """
        (
            mjx_data,
            t,
            step_i,
            next_gate_idx,
            p_prev,
            v_prev,
            last_action_norm,
            thrust_state,
            applied_action,
            imu_history,
            gate_centers,
            gate_normals,
            gate_radii,
            rng_key,
        ) = state_tuple

        # Clip and scale action
        a_norm = jnp.clip(action.astype(jnp.float32), -1.0, 1.0)
        a_scaled = self._scale_action_jax(a_norm)

        # Store p_prev for gate crossing
        p_before = mjx_data.qpos[:3]

        # Apply action immediately (simplified: no latency in MJX version for performance)
        cur_applied = a_scaled
        cur_thrust = thrust_state

        # Run n_substeps physics steps
        def substep_fn(carry, _):
            md, ts, vp = carry
            md, ts, vp = self._apply_forces_single(md, cur_applied, ts, vp)
            return (md, ts, vp), None

        (mjx_data, thrust_state, v_prev), _ = jax.lax.scan(
            substep_fn, (mjx_data, cur_thrust, v_prev), None, length=self.n_substeps
        )

        # Update time
        t = t + jnp.float32(self.dt_control)
        step_i = step_i + 1

        # Compute IMU sample and shift history
        imu_sample = self._compute_imu_jax(mjx_data, v_prev)
        imu_history = jnp.roll(imu_history, -1, axis=0)
        imu_history = imu_history.at[-1].set(imu_sample)

        # Current position
        p_curr = mjx_data.qpos[:3]
        q = mjx_data.qpos[3:7]
        R = self._quat_to_mat_jax(q)
        rpy = self._mat_to_rpy_jax(R)
        v = mjx_data.qvel[:3]

        # Gate crossing check (only check the next gate)
        gate_center = gate_centers[next_gate_idx]
        gate_normal = gate_normals[next_gate_idx]
        gate_rad = gate_radii[next_gate_idx]
        passed = self._gate_crossed_jax(gate_center, gate_normal, gate_rad, p_before, p_curr)
        # Only count if we haven't passed all gates yet
        passed = passed & (next_gate_idx < self.n_gates)
        next_gate_idx = jnp.where(passed, next_gate_idx + 1, next_gate_idx)

        # Termination checks
        oob = (
            (jnp.abs(p_curr[0]) > self._jax_bounds[0])
            | (jnp.abs(p_curr[1]) > self._jax_bounds[1])
            | (p_curr[2] < 0.05)
            | (p_curr[2] > self._jax_bounds[2])
        )
        tilt_exceeded = (jnp.abs(rpy[0]) > self._jax_max_tilt) | (
            jnp.abs(rpy[1]) > self._jax_max_tilt
        )
        crashed = oob | tilt_exceeded
        truncated = step_i >= self.max_steps
        done_track = next_gate_idx >= self.n_gates
        terminated = crashed | done_track

        # Reward
        gate_bonus = jnp.where(passed, self.r_gate, 0.0)

        # Shaping (relative to next gate)
        ng_idx = jnp.clip(next_gate_idx, 0, self.n_gates - 1)
        ng_center = gate_centers[ng_idx]
        ng_normal = gate_normals[ng_idx]
        ng_radius = gate_radii[ng_idx]
        has_next = next_gate_idx < self.n_gates

        d_prev_gate = jnp.linalg.norm(p_before - ng_center)
        d_curr_gate = jnp.linalg.norm(p_curr - ng_center)
        progress = self.k_progress * (d_prev_gate - d_curr_gate)

        # Centering penalty
        d_to_gate = p_curr - ng_center
        d_perp = d_to_gate - jnp.dot(d_to_gate, ng_normal) * ng_normal
        radial = jnp.linalg.norm(d_perp)
        centering = -self.k_center * (radial / jnp.maximum(ng_radius, EPSILON))

        # Speed toward gate
        to_gate_dir = ng_center - p_curr
        to_gate_norm = jnp.linalg.norm(to_gate_dir)
        to_gate_unit = to_gate_dir / jnp.maximum(to_gate_norm, EPSILON)
        v_toward = jnp.dot(v, to_gate_unit)
        speed_reward = self.k_speed * jnp.clip(v_toward, 0.0, 20.0)

        shaping = jnp.where(has_next, progress + centering + speed_reward, 0.0)

        # Smoothness penalty
        da = a_norm - last_action_norm
        smooth_pen = -self.k_smooth * jnp.dot(da, da)

        # Stability
        tilt_pen = -self.k_tilt * (jnp.abs(rpy[0]) + jnp.abs(rpy[1]))
        w_world = mjx_data.qvel[3:6]
        w_body = R.T @ w_world
        w2 = jnp.dot(w_body, w_body)
        ang_pen = -self.k_angrate * jnp.minimum(w2, 400.0)

        reward = self.r_alive + gate_bonus + shaping + smooth_pen + tilt_pen + ang_pen
        reward = jnp.where(crashed, reward + self.r_crash, reward)

        done = terminated | truncated

        new_state_tuple = (
            mjx_data,
            t,
            step_i,
            next_gate_idx,
            p_curr,
            v_prev,
            a_norm,
            thrust_state,
            cur_applied,
            imu_history,
            gate_centers,
            gate_normals,
            gate_radii,
            rng_key,
        )

        info = {
            "gate_passed": passed,
            "next_gate_idx": next_gate_idx,
            "crash": crashed,
            "finished": done_track & (~crashed),
            "pos": p_curr,
            "vel": v,
            "reward_alive": jnp.float32(self.r_alive),
            "reward_gate": gate_bonus,
            "reward_shaping": shaping,
            "reward_smooth": smooth_pen,
            "reward_tilt": tilt_pen,
            "reward_angrate": ang_pen,
            "t": t,
        }

        return new_state_tuple, reward, done, terminated, truncated, info

    def _batched_step(
        self, state: MJXEnvState, actions: jnp.ndarray
    ) -> Tuple[MJXEnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
        """Batched step over all envs (JIT-compiled)."""
        state_tuple = (
            state.mjx_data,
            state.t,
            state.step_i,
            state.next_gate_idx,
            state.p_prev,
            state.v_prev,
            state.last_action_norm,
            state.thrust_state,
            state.applied_action,
            state.imu_history,
            state.gate_centers,
            state.gate_normals,
            state.gate_radii,
            state.rng_key,
        )

        vmapped = jax.vmap(self._step_single_env)
        new_state_tuple, rewards, dones, terminated, truncated, infos = vmapped(
            state_tuple, actions
        )

        new_state = MJXEnvState(
            mjx_data=new_state_tuple[0],
            t=new_state_tuple[1],
            step_i=new_state_tuple[2],
            next_gate_idx=new_state_tuple[3],
            p_prev=new_state_tuple[4],
            v_prev=new_state_tuple[5],
            last_action_norm=new_state_tuple[6],
            thrust_state=new_state_tuple[7],
            applied_action=new_state_tuple[8],
            imu_history=new_state_tuple[9],
            gate_centers=new_state_tuple[10],
            gate_normals=new_state_tuple[11],
            gate_radii=new_state_tuple[12],
            rng_key=new_state_tuple[13],
        )

        return new_state, rewards, dones, terminated, truncated, infos

    def _init_single_env_data(self) -> Any:
        """Create initial MJX data for one environment."""
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        return mjx.put_data(self._mj_model, self._mj_data)

    def _batched_reset(
        self,
        rng_key: jnp.ndarray,
        gate_centers: jnp.ndarray,
        gate_normals: jnp.ndarray,
        gate_radii: jnp.ndarray,
        batched_mjx_data: Any,
    ) -> MJXEnvState:
        """Create initial batched state (JIT-compiled)."""
        B = self.num_envs
        keys = jrandom.split(rng_key, B)

        return MJXEnvState(
            mjx_data=batched_mjx_data,
            t=jnp.zeros(B, dtype=jnp.float32),
            step_i=jnp.zeros(B, dtype=jnp.int32),
            next_gate_idx=jnp.zeros(B, dtype=jnp.int32),
            p_prev=jnp.tile(jnp.array([0.0, 0.0, 1.0]), (B, 1)),
            v_prev=jnp.zeros((B, 3), dtype=jnp.float32),
            last_action_norm=jnp.tile(
                jnp.array([0.0, 0.0, 0.0, self._hover_thrust_norm], dtype=jnp.float32),
                (B, 1),
            ),
            thrust_state=jnp.full(B, self._hover_thrust, dtype=jnp.float32),
            applied_action=jnp.tile(
                jnp.array(
                    [0.0, 0.0, 0.0, self._hover_thrust], dtype=jnp.float32
                ),
                (B, 1),
            ),
            imu_history=jnp.zeros(
                (B, self.imu_window_n, 6), dtype=jnp.float32
            ),
            gate_centers=gate_centers,
            gate_normals=gate_normals,
            gate_radii=gate_radii,
            rng_key=keys,
        )

    def reset(self, seed: Optional[int] = None) -> MJXEnvState:
        """Reset all environments.

        Args:
            seed: Random seed (uses constructor seed if None).

        Returns:
            Initial :class:`MJXEnvState`.
        """
        seed = seed if seed is not None else self._seed

        # Build tracks on CPU
        gates_list = []
        for i in range(self.num_envs):
            rng = np.random.default_rng(seed + i)
            gates_list.append(self._build_track_np(rng))

        gate_centers, gate_normals, gate_radii = self._gates_to_jax(gates_list)

        # Create batched MJX data by stacking individual resets
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        single_data = mjx.put_data(self._mj_model, self._mj_data)
        batched_data = jax.tree.map(
            lambda x: jnp.tile(x[None], (self.num_envs,) + (1,) * x.ndim),
            single_data,
        )

        rng_key = jrandom.PRNGKey(seed)
        state = self._jit_reset(rng_key, gate_centers, gate_normals, gate_radii, batched_data)
        return state

    def step(
        self, state: MJXEnvState, actions: jnp.ndarray
    ) -> Tuple[MJXEnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
        """Step all environments.

        Args:
            state: Current batched state.
            actions: (num_envs, 4) normalized actions.

        Returns:
            (new_state, rewards, dones, terminated, truncated, infos)
        """
        return self._jit_step(state, actions)

    def get_obs_numpy(self, state: MJXEnvState) -> Dict[str, np.ndarray]:
        """Extract observations as NumPy arrays (for Gymnasium wrapper).

        Returns dict with keys:
            - ``imu``: (B, imu_window_n, 6) float32
            - ``last_action``: (B, 4) float32

        Note: Camera images are not available from MJX (no GPU rendering).
        The Gymnasium wrapper should use a placeholder or CPU-side renderer.
        """
        imu = np.asarray(state.imu_history).astype(np.float32)
        last_action = np.asarray(state.last_action_norm).astype(np.float32)
        return {
            "imu": imu,
            "last_action": last_action,
        }
