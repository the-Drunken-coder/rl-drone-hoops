"""Isaac Gym sensor simulation module.

Provides batched camera and IMU sensor simulation for parallel
drone environments. Handles sensor latency, buffering, and
data formatting for the RL observation pipeline.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore
    logger.warning("PyTorch not available; Isaac Gym sensors will not function.")


@dataclass
class CameraConfig:
    """Configuration for FPV camera sensor.

    Attributes:
        image_size: Resolution (square) in pixels.
        fps: Sampling rate in Hz.
        latency_ms: Camera latency in milliseconds.
        fov_deg: Horizontal field of view in degrees.
    """

    image_size: int = 96
    fps: float = 60.0
    latency_ms: float = 20.0
    fov_deg: float = 90.0


@dataclass
class IMUConfig:
    """Configuration for IMU sensor.

    Attributes:
        hz: Sampling rate in Hz.
        latency_ms: IMU latency in milliseconds.
        window_size: Number of samples to keep in history.
        noise_std_gyro: Standard deviation of gyroscope noise (rad/s).
        noise_std_accel: Standard deviation of accelerometer noise (m/s^2).
    """

    hz: float = 400.0
    latency_ms: float = 2.0
    window_size: int = 8
    noise_std_gyro: float = 0.0  # Start without noise
    noise_std_accel: float = 0.0


class IsaacSensors:
    """Batched sensor simulation for Isaac Gym environments.

    Manages camera images and IMU readings for N parallel environments.
    Handles sensor latency through internal ring buffers and delivers
    observations at the correct simulated time.

    Args:
        num_envs: Number of parallel environments.
        camera_config: Camera sensor configuration.
        imu_config: IMU sensor configuration.
        device: PyTorch device string.
        physics_dt: Physics simulation timestep in seconds.
    """

    def __init__(
        self,
        num_envs: int,
        camera_config: Optional[CameraConfig] = None,
        imu_config: Optional[IMUConfig] = None,
        device: str = "cpu",
        physics_dt: float = 0.001,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for IsaacSensors.")

        self.num_envs = num_envs
        self.camera_config = camera_config or CameraConfig()
        self.imu_config = imu_config or IMUConfig()
        self.device = torch.device(device)
        self.physics_dt = physics_dt

        # Camera state
        self._camera_period = 1.0 / self.camera_config.fps
        self._camera_latency_steps = max(
            1, int(self.camera_config.latency_ms / 1000.0 / physics_dt)
        )

        # IMU state
        self._imu_period = 1.0 / self.imu_config.hz
        self._imu_latency_steps = max(
            1, int(self.imu_config.latency_ms / 1000.0 / physics_dt)
        )

        # Initialize buffers
        self._sim_time = 0.0
        self._last_camera_time = -self._camera_period
        self._last_imu_time = -self._imu_period

        # Image buffer: ring buffer for latency simulation
        img_size = self.camera_config.image_size
        self._image_buffer: Deque["torch.Tensor"] = deque(
            maxlen=self._camera_latency_steps + 1
        )
        self._current_image = torch.zeros(
            (num_envs, img_size, img_size, 1),
            dtype=torch.uint8,
            device=self.device,
        )

        # IMU buffer: ring buffer for windowed history
        self._imu_window = torch.zeros(
            (num_envs, self.imu_config.window_size, 6),
            dtype=torch.float32,
            device=self.device,
        )
        self._imu_sample_buffer: Deque["torch.Tensor"] = deque(
            maxlen=self._imu_latency_steps + 1
        )

    def reset(self, env_ids: Optional["torch.Tensor"] = None) -> None:
        """Reset sensor state for specified environments.

        Args:
            env_ids: Environment indices to reset. Resets all if None.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._current_image[env_ids] = 0
        self._imu_window[env_ids] = 0.0

        if len(env_ids) == self.num_envs:
            self._image_buffer.clear()
            self._imu_sample_buffer.clear()
            self._sim_time = 0.0
            self._last_camera_time = -self._camera_period
            self._last_imu_time = -self._imu_period

    def update(
        self,
        state: Dict[str, "torch.Tensor"],
        sim_time: float,
    ) -> None:
        """Update sensors with current physics state.

        Should be called every physics step.

        Args:
            state: Drone state dict from IsaacDronePhysics.get_state().
            sim_time: Current simulation time in seconds.
        """
        self._sim_time = sim_time
        self._update_imu(state, sim_time)
        self._update_camera(state, sim_time)

    def _update_imu(
        self, state: Dict[str, "torch.Tensor"], sim_time: float
    ) -> None:
        """Update IMU sensor with new physics state."""
        if sim_time - self._last_imu_time >= self._imu_period:
            self._last_imu_time = sim_time

            # Extract angular velocity (gyro) and compute acceleration (accel)
            angular_vel = state["angular_velocity"]  # (num_envs, 3)
            velocity = state["velocity"]  # (num_envs, 3)

            # Simple finite-difference acceleration estimate
            # In a full implementation, this would use proper body-frame
            # acceleration from the physics engine
            accel = torch.zeros_like(velocity)
            accel[:, 2] = 9.81  # Gravity-compensated (at hover)

            # Combine into 6-DOF IMU reading: [gx, gy, gz, ax, ay, az]
            imu_sample = torch.cat([angular_vel, accel], dim=1)  # (num_envs, 6)

            # Add noise if configured
            if self.imu_config.noise_std_gyro > 0:
                imu_sample[:, :3] += torch.randn_like(
                    imu_sample[:, :3]
                ) * self.imu_config.noise_std_gyro
            if self.imu_config.noise_std_accel > 0:
                imu_sample[:, 3:] += torch.randn_like(
                    imu_sample[:, 3:]
                ) * self.imu_config.noise_std_accel

            # Push to latency buffer
            self._imu_sample_buffer.append(imu_sample.clone())

            # Deliver oldest sample (simulating latency)
            if len(self._imu_sample_buffer) > self._imu_latency_steps:
                delivered = self._imu_sample_buffer[0]
                # Shift window and append new sample
                self._imu_window = torch.roll(self._imu_window, -1, dims=1)
                self._imu_window[:, -1, :] = delivered

    def _update_camera(
        self, state: Dict[str, "torch.Tensor"], sim_time: float
    ) -> None:
        """Update camera sensor with new rendering.

        In a full Isaac Gym implementation, this would render from the
        GPU camera sensor. For CPU fallback, generates a synthetic
        grayscale image based on drone position.
        """
        if sim_time - self._last_camera_time >= self._camera_period:
            self._last_camera_time = sim_time

            # In CPU fallback mode, generate a synthetic image
            # Real Isaac Gym would use gymapi camera sensors
            img = self._render_synthetic(state)

            self._image_buffer.append(img.clone())

            # Deliver oldest frame (simulating latency)
            if len(self._image_buffer) > self._camera_latency_steps:
                self._current_image = self._image_buffer[0]

    def _render_synthetic(
        self, state: Dict[str, "torch.Tensor"]
    ) -> "torch.Tensor":
        """Generate synthetic camera image for testing.

        Creates a simple grayscale gradient based on drone height,
        suitable for testing the observation pipeline without
        a full rendering engine.

        Args:
            state: Current drone state.

        Returns:
            Tensor of shape (num_envs, H, W, 1) dtype uint8.
        """
        h = w = self.camera_config.image_size
        # Create a height-dependent gradient image
        heights = state["position"][:, 2].clamp(0.0, 10.0) / 10.0  # Normalize
        base = (heights * 200.0).to(torch.uint8)  # 0-200 intensity

        img = base.view(-1, 1, 1, 1).expand(-1, h, w, 1).clone()
        return img

    def get_observations(self) -> Dict[str, "torch.Tensor"]:
        """Get current sensor observations for all environments.

        Returns:
            Dictionary with:
                - 'image': (num_envs, H, W, 1) uint8 grayscale camera frame
                - 'imu': (num_envs, window_size, 6) float32 IMU history
        """
        return {
            "image": self._current_image.clone(),
            "imu": self._imu_window.clone(),
        }
