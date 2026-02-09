"""Isaac Gym drone physics simulation module.

Implements vectorized drone dynamics for parallel GPU simulation.
All operations use PyTorch tensors for GPU-native computation.
When Isaac Gym is not available, provides a CPU-based fallback
using the same tensor interface.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore
    logger.warning("PyTorch not available; Isaac Gym physics will not function.")

try:
    from isaacgym import gymapi, gymtorch, gymutil  # type: ignore
    ISAAC_GYM_AVAILABLE = True
except ImportError:
    ISAAC_GYM_AVAILABLE = False
    logger.info("Isaac Gym not available; using CPU tensor fallback for physics.")


@dataclass
class DronePhysicsConfig:
    """Physics configuration for the drone simulation.

    Attributes:
        mass: Drone mass in kg.
        gravity: Gravitational acceleration magnitude (m/s^2).
        max_thrust: Maximum total thrust in Newtons.
        max_rate: Maximum body rate command in rad/s.
        thrust_tau: First-order lag time constant for thrust (seconds).
        rate_tau: First-order lag time constant for rate control (seconds).
        drag_coeff_linear: Linear drag coefficient.
        drag_coeff_quadratic: Quadratic drag coefficient.
        thrust_limits: (min, max) thrust per motor in Newtons.
        rate_limits: (min, max) body rate in rad/s.
        ixx: Moment of inertia about X axis.
        iyy: Moment of inertia about Y axis.
        izz: Moment of inertia about Z axis.
        dt: Physics timestep in seconds.
    """

    mass: float = 0.5
    gravity: float = 9.81
    max_thrust: float = 12.0  # 4 motors * 3.0 N each
    max_rate: float = 8.0
    thrust_tau: float = 0.02  # 20ms first-order lag
    rate_tau: float = 0.01  # 10ms first-order lag
    drag_coeff_linear: float = 0.1
    drag_coeff_quadratic: float = 0.01
    thrust_limits: Tuple[float, float] = (0.0, 12.0)
    rate_limits: Tuple[float, float] = (-8.0, 8.0)
    ixx: float = 0.0023
    iyy: float = 0.0023
    izz: float = 0.004
    dt: float = 0.001  # 1000 Hz physics


class IsaacDronePhysics:
    """Vectorized drone physics for Isaac Gym parallel environments.

    Manages drone state and actuator dynamics for N parallel environments.
    All tensors are batched with shape (num_envs, ...).

    When Isaac Gym is available, physics runs through PhysX on GPU.
    Otherwise, a CPU tensor-based simulation is used with the same interface.

    Args:
        num_envs: Number of parallel environments.
        config: Physics configuration.
        device: PyTorch device string ('cuda:0' or 'cpu').
        sim: Isaac Gym simulation handle (None for CPU fallback).
        env_handles: List of Isaac Gym environment handles.
        actor_handles: List of actor handles (one per env).
    """

    def __init__(
        self,
        num_envs: int,
        config: Optional[DronePhysicsConfig] = None,
        device: str = "cpu",
        sim: Any = None,
        env_handles: Optional[list] = None,
        actor_handles: Optional[list] = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for IsaacDronePhysics.")

        self.num_envs = num_envs
        self.config = config or DronePhysicsConfig()
        self.device = torch.device(device)
        self.sim = sim
        self.env_handles = env_handles
        self.actor_handles = actor_handles
        self.use_isaac = sim is not None and ISAAC_GYM_AVAILABLE

        # Actuator state (first-order lag filters)
        self._current_thrust = torch.zeros(
            num_envs, dtype=torch.float32, device=self.device
        )
        self._current_rates = torch.zeros(
            (num_envs, 3), dtype=torch.float32, device=self.device
        )

        # State buffers for CPU fallback mode
        self._position = torch.zeros(
            (num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._velocity = torch.zeros(
            (num_envs, 3), dtype=torch.float32, device=self.device
        )
        # Quaternion (w, x, y, z)
        self._quaternion = torch.zeros(
            (num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._quaternion[:, 0] = 1.0  # Identity quaternion
        self._angular_velocity = torch.zeros(
            (num_envs, 3), dtype=torch.float32, device=self.device
        )

    def reset(self, env_ids: Optional["torch.Tensor"] = None) -> None:
        """Reset physics state for specified environments.

        Args:
            env_ids: Tensor of environment indices to reset.
                     If None, resets all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._current_thrust[env_ids] = 0.0
        self._current_rates[env_ids] = 0.0
        self._position[env_ids] = 0.0
        self._position[env_ids, 2] = 2.0  # Start at height 2m
        self._velocity[env_ids] = 0.0
        self._quaternion[env_ids] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )
        self._angular_velocity[env_ids] = 0.0

    def apply_actions(self, actions: "torch.Tensor") -> None:
        """Apply normalized action commands to all parallel environments.

        Actions are 4D: [roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd, thrust_cmd]
        All in range [-1, 1], scaled to physical limits internally.

        Args:
            actions: Tensor of shape (num_envs, 4) with normalized commands.
        """
        actions = actions.clamp(-1.0, 1.0)

        # Scale actions to physical limits
        rate_cmd = actions[:, :3] * self.config.max_rate  # (num_envs, 3)
        thrust_cmd = (actions[:, 3] + 1.0) / 2.0 * self.config.max_thrust  # (num_envs,)

        # First-order lag on actuators
        dt = self.config.dt
        alpha_thrust = dt / (self.config.thrust_tau + dt)
        alpha_rate = dt / (self.config.rate_tau + dt)

        self._current_thrust = (
            self._current_thrust + alpha_thrust * (thrust_cmd - self._current_thrust)
        )
        self._current_rates = (
            self._current_rates + alpha_rate * (rate_cmd - self._current_rates)
        )

        # Clamp to limits
        self._current_thrust.clamp_(
            self.config.thrust_limits[0], self.config.thrust_limits[1]
        )
        self._current_rates.clamp_(
            self.config.rate_limits[0], self.config.rate_limits[1]
        )

    def step(self) -> None:
        """Advance the physics simulation by one timestep.

        If Isaac Gym is available and configured, delegates to PhysX.
        Otherwise, performs a simple Euler integration on CPU/GPU tensors.
        """
        if self.use_isaac:
            self._step_isaac()
        else:
            self._step_cpu()

    def _step_isaac(self) -> None:
        """Step physics through Isaac Gym / PhysX."""
        # Apply forces/torques to all environments
        # (This would use gymapi.apply_rigid_body_force_at_pos_tensors
        #  when Isaac Gym is properly initialized)
        raise NotImplementedError(
            "Isaac Gym stepping requires a valid simulation handle. "
            "Use CPU fallback for testing without Isaac Gym SDK."
        )

    def _step_cpu(self) -> None:
        """CPU/GPU tensor-based physics simulation (fallback).

        Simple semi-implicit Euler integration with:
        - Gravity
        - Thrust along body Z-axis
        - Linear and quadratic drag
        - Angular velocity integration
        """
        dt = self.config.dt
        g = self.config.gravity
        mass = self.config.mass

        # Compute body Z-axis from quaternion (thrust direction)
        body_z = _quat_rotate_vector(
            self._quaternion,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(
                self.num_envs, 3
            ),
        )

        # Forces: gravity + thrust + drag
        gravity_force = torch.tensor(
            [0.0, 0.0, -g * mass], device=self.device
        ).expand(self.num_envs, 3)

        thrust_force = body_z * self._current_thrust.unsqueeze(1)

        speed = self._velocity.norm(dim=1, keepdim=True).clamp(min=1e-8)
        drag_linear = -self.config.drag_coeff_linear * self._velocity
        drag_quadratic = (
            -self.config.drag_coeff_quadratic * speed * self._velocity
        )

        total_force = gravity_force + thrust_force + drag_linear + drag_quadratic
        acceleration = total_force / mass

        # Semi-implicit Euler integration
        self._velocity = self._velocity + acceleration * dt
        self._position = self._position + self._velocity * dt

        # Angular velocity update (simplified: direct rate control)
        # In a full model, torques would be computed from rate error
        self._angular_velocity = self._current_rates.clone()

        # Quaternion integration
        self._quaternion = _integrate_quaternion(
            self._quaternion, self._angular_velocity, dt
        )

        # Clamp altitude (ground collision)
        ground_mask = self._position[:, 2] < 0.0
        self._position[ground_mask, 2] = 0.0
        self._velocity[ground_mask, 2] = 0.0

    def get_state(self) -> Dict[str, "torch.Tensor"]:
        """Get the current batched drone state.

        Returns:
            Dictionary with keys:
                - 'position': (num_envs, 3) world-frame position
                - 'velocity': (num_envs, 3) world-frame velocity
                - 'quaternion': (num_envs, 4) orientation (w, x, y, z)
                - 'angular_velocity': (num_envs, 3) body-frame angular velocity
                - 'thrust': (num_envs,) current thrust
                - 'rates': (num_envs, 3) current body rates
        """
        return {
            "position": self._position.clone(),
            "velocity": self._velocity.clone(),
            "quaternion": self._quaternion.clone(),
            "angular_velocity": self._angular_velocity.clone(),
            "thrust": self._current_thrust.clone(),
            "rates": self._current_rates.clone(),
        }

    def set_position(
        self, env_ids: "torch.Tensor", positions: "torch.Tensor"
    ) -> None:
        """Set drone positions for specific environments.

        Args:
            env_ids: Environment indices.
            positions: (len(env_ids), 3) positions.
        """
        self._position[env_ids] = positions


def _quat_rotate_vector(
    q: "torch.Tensor", v: "torch.Tensor"
) -> "torch.Tensor":
    """Rotate vectors by quaternions (batched).

    Args:
        q: Quaternions of shape (..., 4) in (w, x, y, z) format.
        v: Vectors of shape (..., 3).

    Returns:
        Rotated vectors of shape (..., 3).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    # Quaternion rotation: q * v * q^{-1}
    t0 = 2.0 * (x * vx + y * vy + z * vz)
    t1 = w * w - (x * x + y * y + z * z)

    cx = y * vz - z * vy
    cy = z * vx - x * vz
    cz = x * vy - y * vx

    rx = t1 * vx + t0 * x + 2.0 * w * cx
    ry = t1 * vy + t0 * y + 2.0 * w * cy
    rz = t1 * vz + t0 * z + 2.0 * w * cz

    return torch.stack([rx, ry, rz], dim=-1)


def _integrate_quaternion(
    q: "torch.Tensor", omega: "torch.Tensor", dt: float
) -> "torch.Tensor":
    """Integrate quaternion with angular velocity (first-order).

    Args:
        q: Current quaternions (num_envs, 4) in (w, x, y, z) format.
        omega: Angular velocities (num_envs, 3) in body frame.
        dt: Timestep.

    Returns:
        Updated quaternions (num_envs, 4), normalized.
    """
    # dq/dt = 0.5 * q ⊗ [0, omega]
    ox, oy, oz = omega[..., 0], omega[..., 1], omega[..., 2]
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    dw = 0.5 * (-x * ox - y * oy - z * oz)
    dx = 0.5 * (w * ox + y * oz - z * oy)
    dy = 0.5 * (w * oy + z * ox - x * oz)
    dz = 0.5 * (w * oz + x * oy - y * ox)

    q_new = torch.stack([w + dw * dt, x + dx * dt, y + dy * dt, z + dz * dt], dim=-1)

    # Normalize
    q_new = q_new / q_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return q_new
