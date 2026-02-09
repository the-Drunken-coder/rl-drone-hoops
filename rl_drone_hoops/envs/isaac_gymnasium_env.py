"""Gymnasium-compatible Isaac Gym drone hoops environment.

Wraps the Isaac Gym simulation in a standard Gymnasium interface,
managing N parallel environments internally. All observations
are returned as PyTorch tensors.

When Isaac Gym SDK is not installed, falls back to a CPU tensor-based
simulation that provides the same interface for development and testing.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore

import gymnasium as gym
from gymnasium import spaces

from rl_drone_hoops.constants import ACTION_DIM, MAX_TILT_DEG
from rl_drone_hoops.envs.isaac_gym_env import IsaacGymBase, IsaacSimParams
from rl_drone_hoops.envs.isaac_drone_physics import (
    DronePhysicsConfig,
    IsaacDronePhysics,
)
from rl_drone_hoops.envs.isaac_sensors import (
    CameraConfig,
    IMUConfig,
    IsaacSensors,
)
from rl_drone_hoops.assets.isaac_gym_assets import (
    TrackConfig,
    generate_gate_positions,
)


@dataclass
class IsaacEnvConfig:
    """Configuration for the Isaac Gym drone hoops environment.

    Attributes:
        num_envs: Number of parallel environments.
        image_size: FPV camera resolution (square, pixels).
        camera_fps: Camera sampling rate in Hz.
        camera_latency_ms: Camera latency in milliseconds.
        imu_hz: IMU sampling rate in Hz.
        imu_latency_ms: IMU latency in milliseconds.
        imu_window_size: Number of IMU samples in observation window.
        control_hz: Control loop rate in Hz.
        physics_hz: Physics simulation rate in Hz.
        n_gates: Number of gates per episode.
        gate_radius: Gate radius in meters.
        gate_spacing: Distance between gates in meters.
        track_type: 'straight' or 'random_turns'.
        turn_max_deg: Maximum turn angle between gates.
        episode_duration_s: Maximum episode duration in seconds.
        max_tilt_deg: Maximum tilt before termination.
        device: PyTorch device ('cpu' or 'cuda:N').
        seed: Random seed for reproducibility.
    """

    num_envs: int = 256
    image_size: int = 96
    camera_fps: float = 60.0
    camera_latency_ms: float = 20.0
    imu_hz: float = 400.0
    imu_latency_ms: float = 2.0
    imu_window_size: int = 8
    control_hz: float = 100.0
    physics_hz: float = 1000.0
    n_gates: int = 3
    gate_radius: float = 1.25
    gate_spacing: float = 5.0
    track_type: str = "straight"
    turn_max_deg: float = 20.0
    episode_duration_s: float = 12.0
    max_tilt_deg: float = 75.0
    device: str = "cpu"
    seed: int = 0


# Reward constants (matching the MuJoCo environment)
REWARD_GATE_PASSED = 10.0
REWARD_SURVIVAL = 0.1
REWARD_CRASH = -20.0
REWARD_PROGRESS_SCALE = 1.0
REWARD_CENTERING_SCALE = 0.5
REWARD_SPEED_SCALE = 0.1
REWARD_SMOOTHNESS_SCALE = 0.05


class IsaacDroneHoopsEnv(gym.Env):
    """Gymnasium-compatible Isaac Gym drone hoops environment.

    Manages N parallel drone simulations internally. Each environment
    has its own track, gates, and episode state. When an episode ends
    (crash, timeout, or all gates passed), that environment is
    automatically reset.

    Observations:
        Dictionary with:
        - 'image': (num_envs, H, W, 1) uint8 grayscale FPV camera
        - 'imu': (num_envs, window_size, 6) float32 IMU history
        - 'last_action': (num_envs, 4) float32 previous actions

    Actions:
        (num_envs, 4) float32 in [-1, 1]:
        [roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd, thrust_cmd]

    Rewards:
        (num_envs,) float32 per-step rewards including:
        - +10 per gate passed
        - +0.1 survival bonus per step
        - Progress toward next gate (shaping)
        - -20 on crash

    Args:
        config: Environment configuration. Uses defaults if None.
        num_envs: Number of parallel environments (overrides config).
        sim_params: Isaac Gym simulation parameters.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: Optional[IsaacEnvConfig] = None,
        num_envs: Optional[int] = None,
        sim_params: Optional[IsaacSimParams] = None,
    ) -> None:
        super().__init__()

        if torch is None:
            raise ImportError("PyTorch is required for IsaacDroneHoopsEnv.")

        self.config = config or IsaacEnvConfig()
        if num_envs is not None:
            self.config.num_envs = num_envs

        self.num_envs = self.config.num_envs
        self.device = torch.device(self.config.device)

        # Timing
        self._physics_dt = 1.0 / self.config.physics_hz
        self._control_dt = 1.0 / self.config.control_hz
        self._physics_steps_per_control = max(
            1, int(self.config.physics_hz / self.config.control_hz)
        )
        self._max_episode_steps = int(
            self.config.episode_duration_s * self.config.control_hz
        )
        self._max_tilt_rad = math.radians(self.config.max_tilt_deg)

        # Define observation and action spaces
        img_size = self.config.image_size
        imu_window = self.config.imu_window_size

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(self.num_envs, img_size, img_size, 1),
                dtype=np.uint8,
            ),
            "imu": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_envs, imu_window, 6),
                dtype=np.float32,
            ),
            "last_action": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.num_envs, ACTION_DIM),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_envs, ACTION_DIM),
            dtype=np.float32,
        )

        # Initialize physics
        physics_config = DronePhysicsConfig(dt=self._physics_dt)
        self.physics = IsaacDronePhysics(
            num_envs=self.num_envs,
            config=physics_config,
            device=str(self.device),
        )

        # Initialize sensors
        camera_config = CameraConfig(
            image_size=self.config.image_size,
            fps=self.config.camera_fps,
            latency_ms=self.config.camera_latency_ms,
        )
        imu_config = IMUConfig(
            hz=self.config.imu_hz,
            latency_ms=self.config.imu_latency_ms,
            window_size=self.config.imu_window_size,
        )
        self.sensors = IsaacSensors(
            num_envs=self.num_envs,
            camera_config=camera_config,
            imu_config=imu_config,
            device=str(self.device),
            physics_dt=self._physics_dt,
        )

        # Track and gate state
        self._rng = np.random.default_rng(self.config.seed)
        self._gates: List[List[Dict[str, Any]]] = []
        self._next_gate_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Episode state
        self._step_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._sim_time = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._last_action = torch.zeros(
            (self.num_envs, ACTION_DIM), dtype=torch.float32, device=self.device
        )
        self._prev_gate_dist = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float32, device=self.device
        )
        self._episode_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # Pre-compute gate positions as tensors
        self._gate_centers: Optional[torch.Tensor] = None  # (num_envs, n_gates, 3)
        self._gate_normals: Optional[torch.Tensor] = None  # (num_envs, n_gates, 3)

        logger.info(
            "IsaacDroneHoopsEnv created: %d envs, device=%s, "
            "physics=%gHz, control=%gHz",
            self.num_envs,
            self.device,
            self.config.physics_hz,
            self.config.control_hz,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, Any]]:
        """Reset all environments to initial state.

        Args:
            seed: Random seed (updates internal RNG).
            options: Optional dict; supports 'env_ids' to reset subset.

        Returns:
            Tuple of (observations, info_dict).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        env_ids = None
        if options and "env_ids" in options:
            env_ids = torch.as_tensor(options["env_ids"], device=self.device)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._reset_envs(env_ids)

        obs = self._get_observations()
        info = {
            "next_gate_idx": self._next_gate_idx.clone(),
            "step_count": self._step_count.clone(),
        }
        return obs, info

    def _reset_envs(self, env_ids: "torch.Tensor") -> None:
        """Reset specified environments."""
        # Generate new tracks
        track_config = TrackConfig(
            n_gates=self.config.n_gates,
            gate_spacing=self.config.gate_spacing,
            gate_radius=self.config.gate_radius,
            track_type=self.config.track_type,
            turn_max_deg=self.config.turn_max_deg,
        )

        # Ensure gate list is populated
        while len(self._gates) < self.num_envs:
            self._gates.append([])

        for idx in env_ids.cpu().tolist():
            gates = generate_gate_positions(track_config, self._rng)
            self._gates[idx] = gates

        # Build gate center/normal tensors
        self._rebuild_gate_tensors()

        # Reset physics and sensors
        self.physics.reset(env_ids)
        self.sensors.reset(env_ids)

        # Reset episode state
        self._step_count[env_ids] = 0
        self._sim_time[env_ids] = 0.0
        self._last_action[env_ids] = 0.0
        self._next_gate_idx[env_ids] = 0
        self._prev_gate_dist[env_ids] = float("inf")
        self._episode_reward[env_ids] = 0.0

    def _rebuild_gate_tensors(self) -> None:
        """Rebuild batched gate center/normal tensors from gate lists."""
        n_gates = self.config.n_gates
        centers = torch.zeros(
            (self.num_envs, n_gates, 3),
            dtype=torch.float32,
            device=self.device,
        )
        normals = torch.zeros(
            (self.num_envs, n_gates, 3),
            dtype=torch.float32,
            device=self.device,
        )

        for i, gate_list in enumerate(self._gates):
            for j, gate in enumerate(gate_list):
                centers[i, j] = torch.tensor(
                    gate["center"], dtype=torch.float32, device=self.device
                )
                normals[i, j] = torch.tensor(
                    gate["normal"], dtype=torch.float32, device=self.device
                )

        self._gate_centers = centers
        self._gate_normals = normals

    def step(
        self, actions: Union[np.ndarray, "torch.Tensor"]
    ) -> Tuple[
        Dict[str, "torch.Tensor"],
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        Dict[str, Any],
    ]:
        """Execute one control step in all parallel environments.

        Args:
            actions: (num_envs, 4) actions in [-1, 1].

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

        actions = actions.clamp(-1.0, 1.0)

        # Run physics substeps
        self.physics.apply_actions(actions)
        for _ in range(self._physics_steps_per_control):
            self.physics.step()
            state = self.physics.get_state()
            sim_time = (self._step_count.float() * self._control_dt).mean().item()
            self.sensors.update(state, sim_time)

        self._step_count += 1
        self._last_action = actions.clone()

        # Get state for reward/termination computation
        state = self.physics.get_state()

        # Compute rewards
        reward = self._compute_reward(state, actions)
        self._episode_reward += reward

        # Check termination conditions
        terminated, truncated, crash_mask, oob_mask = self._check_termination(state)

        # Build info
        info: Dict[str, Any] = {
            "next_gate_idx": self._next_gate_idx.clone(),
            "step_count": self._step_count.clone(),
            "crash": crash_mask.clone(),
            "out_of_bounds": oob_mask.clone(),
            "episode_reward": self._episode_reward.clone(),
        }

        # Auto-reset terminated/truncated environments
        done_mask = terminated | truncated
        if done_mask.any():
            done_ids = torch.where(done_mask)[0]
            self._reset_envs(done_ids)

        obs = self._get_observations()
        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self,
        state: Dict[str, "torch.Tensor"],
        actions: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute vectorized rewards for all environments.

        Args:
            state: Current drone state from physics.
            actions: Current actions.

        Returns:
            (num_envs,) reward tensor.
        """
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        if self._gate_centers is None:
            return reward

        # Survival bonus
        reward += REWARD_SURVIVAL

        pos = state["position"]  # (num_envs, 3)

        # Gate progress and crossing detection
        gate_passed = self._check_gate_crossing(state)
        reward += gate_passed.float() * REWARD_GATE_PASSED

        # Progress toward next gate (shaping)
        curr_gate_dist = self._distance_to_next_gate(pos)
        progress = (self._prev_gate_dist - curr_gate_dist).clamp(-2.0, 2.0)
        reward += progress * REWARD_PROGRESS_SCALE
        self._prev_gate_dist = curr_gate_dist

        # Centering reward (closer to gate center axis)
        centering = self._gate_centering_reward(pos)
        reward += centering * REWARD_CENTERING_SCALE

        # Smoothness penalty (penalize large action changes)
        if self._step_count.min() > 1:
            action_diff = (actions - self._last_action).norm(dim=1)
            reward -= action_diff * REWARD_SMOOTHNESS_SCALE

        return reward

    def _check_gate_crossing(
        self, state: Dict[str, "torch.Tensor"]
    ) -> "torch.Tensor":
        """Detect gate crossing for all environments.

        Uses plane-crossing logic: checks if the drone has crossed
        the gate plane in the forward direction.

        Returns:
            (num_envs,) boolean tensor, True where a gate was passed.
        """
        if self._gate_centers is None or self._gate_normals is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        pos = state["position"]  # (num_envs, 3)
        passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Gather current target gate center and normal
        gate_idx = self._next_gate_idx.clone()
        valid = gate_idx < self.config.n_gates

        if not valid.any():
            return passed

        # Get target gate for each env
        batch_idx = torch.arange(self.num_envs, device=self.device)
        clamped_idx = gate_idx.clamp(max=self.config.n_gates - 1)
        target_center = self._gate_centers[batch_idx, clamped_idx]  # (num_envs, 3)
        target_normal = self._gate_normals[batch_idx, clamped_idx]  # (num_envs, 3)

        # Vector from gate center to drone
        to_drone = pos - target_center  # (num_envs, 3)

        # Signed distance along gate normal
        signed_dist = (to_drone * target_normal).sum(dim=1)  # (num_envs,)

        # Gate is passed when drone crosses the plane (signed_dist > 0)
        # and is within the gate radius
        lateral_dist = (to_drone - signed_dist.unsqueeze(1) * target_normal).norm(dim=1)

        crossing = (signed_dist > 0) & (lateral_dist < self.config.gate_radius) & valid
        passed = crossing

        # Advance gate index for environments that passed
        self._next_gate_idx[passed] += 1

        return passed

    def _distance_to_next_gate(self, pos: "torch.Tensor") -> "torch.Tensor":
        """Compute distance from drone to next gate center.

        Args:
            pos: (num_envs, 3) drone positions.

        Returns:
            (num_envs,) distances.
        """
        if self._gate_centers is None:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        batch_idx = torch.arange(self.num_envs, device=self.device)
        clamped_idx = self._next_gate_idx.clamp(max=self.config.n_gates - 1)
        target_center = self._gate_centers[batch_idx, clamped_idx]
        return (pos - target_center).norm(dim=1)

    def _gate_centering_reward(self, pos: "torch.Tensor") -> "torch.Tensor":
        """Compute centering reward based on proximity to gate axis.

        Args:
            pos: (num_envs, 3) drone positions.

        Returns:
            (num_envs,) centering rewards (higher = more centered).
        """
        if self._gate_centers is None or self._gate_normals is None:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        batch_idx = torch.arange(self.num_envs, device=self.device)
        clamped_idx = self._next_gate_idx.clamp(max=self.config.n_gates - 1)
        target_center = self._gate_centers[batch_idx, clamped_idx]
        target_normal = self._gate_normals[batch_idx, clamped_idx]

        to_drone = pos - target_center
        signed_dist = (to_drone * target_normal).sum(dim=1)
        lateral = to_drone - signed_dist.unsqueeze(1) * target_normal
        lateral_dist = lateral.norm(dim=1)

        # Reward: 1.0 when perfectly centered, decaying with distance
        radius = self.config.gate_radius
        centering = (1.0 - (lateral_dist / radius).clamp(max=2.0)).clamp(min=0.0)
        return centering

    def _check_termination(
        self, state: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Check termination and truncation conditions.

        Returns:
            Tuple of (terminated, truncated, crash_mask, oob_mask).
        """
        pos = state["position"]
        quat = state["quaternion"]

        # Crash detection: ground collision
        ground_crash = pos[:, 2] <= 0.01

        # Crash detection: excessive tilt
        # Compute tilt angle from quaternion
        # For quaternion (w, x, y, z), the body Z-axis in world frame is:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        body_z_world = torch.stack([
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ], dim=1)
        cos_tilt = body_z_world[:, 2].clamp(-1.0, 1.0)
        tilt_angle = torch.acos(cos_tilt)
        tilt_crash = tilt_angle > self._max_tilt_rad

        crash_mask = ground_crash | tilt_crash

        # Out of bounds
        oob_mask = (
            (pos[:, 0].abs() > 50.0)
            | (pos[:, 1].abs() > 50.0)
            | (pos[:, 2] > 20.0)
        )

        terminated = crash_mask | oob_mask

        # Apply crash penalty
        reward_adjustment = crash_mask.float() * REWARD_CRASH
        self._episode_reward += reward_adjustment

        # Truncation: max episode steps
        truncated = self._step_count >= self._max_episode_steps

        return terminated, truncated, crash_mask, oob_mask

    def _get_observations(self) -> Dict[str, "torch.Tensor"]:
        """Assemble observation dictionary from sensors and state.

        Returns:
            Dict with 'image', 'imu', and 'last_action' tensors.
        """
        sensor_obs = self.sensors.get_observations()
        return {
            "image": sensor_obs["image"],
            "imu": sensor_obs["imu"],
            "last_action": self._last_action.clone(),
        }

    def close(self) -> None:
        """Clean up environment resources."""
        logger.info("IsaacDroneHoopsEnv closed.")

    def __repr__(self) -> str:
        return (
            f"IsaacDroneHoopsEnv(num_envs={self.num_envs}, "
            f"device={self.device}, n_gates={self.config.n_gates})"
        )
