"""Base Isaac Gym environment for drone hoops simulation.

Provides the core simulation setup, environment creation, and
stepping logic for running parallel drone simulations on GPU
via NVIDIA Isaac Gym. Falls back to CPU tensor-based simulation
when Isaac Gym SDK is not available.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    from isaacgym import gymapi, gymtorch, gymutil  # type: ignore
    ISAAC_GYM_AVAILABLE = True
except ImportError:
    ISAAC_GYM_AVAILABLE = False


@dataclass
class IsaacSimParams:
    """Parameters for Isaac Gym simulation.

    Attributes:
        num_envs: Number of parallel environments.
        spacing: Spacing between environments in meters.
        physics_dt: Physics simulation timestep in seconds.
        substeps: Number of physics substeps per step.
        gravity: Gravity vector (x, y, z).
        use_gpu_pipeline: Whether to use GPU pipeline for tensor access.
        compute_device_id: CUDA device ID for compute.
        graphics_device_id: CUDA device ID for rendering (-1 for headless).
    """

    num_envs: int = 256
    spacing: float = 10.0
    physics_dt: float = 0.001
    substeps: int = 2
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    use_gpu_pipeline: bool = True
    compute_device_id: int = 0
    graphics_device_id: int = -1  # -1 = headless


class IsaacGymBase:
    """Base class for Isaac Gym parallel environments.

    Manages the Isaac Gym simulation instance, environment creation,
    and provides the stepping interface. When Isaac Gym is not available,
    provides stub methods for testing.

    Args:
        sim_params: Simulation parameters.
    """

    def __init__(self, sim_params: Optional[IsaacSimParams] = None) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for Isaac Gym environments.")

        self.sim_params = sim_params or IsaacSimParams()
        self.num_envs = self.sim_params.num_envs
        self.device = "cpu"

        self.gym = None
        self.sim = None
        self.env_handles: List[Any] = []
        self.actor_handles: List[Any] = []

        if ISAAC_GYM_AVAILABLE:
            self._init_isaac_gym()
        else:
            logger.info(
                "Isaac Gym not available. Environment will use CPU tensor "
                "fallback for physics simulation."
            )
            self.device = "cpu"

    def _init_isaac_gym(self) -> None:
        """Initialize Isaac Gym simulation.

        Creates the Gym instance, configures PhysX, and sets up the
        simulation with the configured parameters.
        """
        self.gym = gymapi.acquire_gym()

        # Configure simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.dt = self.sim_params.physics_dt
        sim_params.substeps = self.sim_params.substeps
        sim_params.gravity = gymapi.Vec3(*self.sim_params.gravity)

        # PhysX parameters
        sim_params.physx.solver_type = 1  # TGS solver
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = self.sim_params.use_gpu_pipeline

        self.sim = self.gym.create_sim(
            self.sim_params.compute_device_id,
            self.sim_params.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params,
        )

        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation.")

        self.device = (
            f"cuda:{self.sim_params.compute_device_id}"
            if self.sim_params.use_gpu_pipeline
            else "cpu"
        )

        logger.info(
            "Isaac Gym simulation created: %d envs on device %s",
            self.num_envs,
            self.device,
        )

    def create_envs(self) -> None:
        """Create parallel environments in Isaac Gym.

        Must be implemented by subclasses to load assets and
        create actors for each environment.
        """
        if not ISAAC_GYM_AVAILABLE:
            logger.info("Skipping env creation (Isaac Gym not available).")
            return

        raise NotImplementedError("Subclasses must implement create_envs().")

    def step_physics(self) -> None:
        """Step the Isaac Gym physics simulation."""
        if self.sim is not None:
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        # CPU fallback: physics stepped externally by IsaacDronePhysics

    def render(self) -> None:
        """Render the simulation (if viewer is attached)."""
        if self.sim is not None and self.gym is not None:
            self.gym.step_graphics(self.sim)

    def destroy(self) -> None:
        """Clean up Isaac Gym resources."""
        if self.sim is not None and self.gym is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None
            logger.info("Isaac Gym simulation destroyed.")
