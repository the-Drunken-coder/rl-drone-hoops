"""Isaac Gym asset creation and management for drone hoops environment.

Provides URDF generation and asset loading utilities for creating
quadrotor drones and gate obstacles in Isaac Gym simulations.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DroneAssetConfig:
    """Configuration for the quadrotor drone asset.

    Attributes:
        mass: Total drone mass in kg.
        arm_length: Distance from center to motor in meters.
        body_radius: Radius of central body sphere in meters.
        body_height: Height of central body cylinder in meters.
        motor_radius: Radius of motor spheres in meters.
        ixx: Moment of inertia about X axis (kg*m^2).
        iyy: Moment of inertia about Y axis (kg*m^2).
        izz: Moment of inertia about Z axis (kg*m^2).
        max_thrust_per_motor: Maximum thrust per motor in Newtons.
        max_rate: Maximum body rate in rad/s.
    """

    mass: float = 0.5
    arm_length: float = 0.15
    body_radius: float = 0.05
    body_height: float = 0.03
    motor_radius: float = 0.01
    ixx: float = 0.0023
    iyy: float = 0.0023
    izz: float = 0.004
    max_thrust_per_motor: float = 3.0
    max_rate: float = 8.0  # rad/s


@dataclass
class GateAssetConfig:
    """Configuration for a gate (hoop) obstacle.

    Attributes:
        radius: Gate inner radius in meters.
        tube_radius: Radius of the gate tube in meters.
        n_segments: Number of segments used to approximate the ring.
        color: RGBA color of the gate.
    """

    radius: float = 1.25
    tube_radius: float = 0.05
    n_segments: int = 24
    color: Tuple[float, float, float, float] = (1.0, 0.5, 0.0, 1.0)


@dataclass
class TrackConfig:
    """Configuration for the racing track.

    Attributes:
        n_gates: Number of gates on the track.
        gate_spacing: Distance between gates in meters.
        gate_radius: Gate radius in meters.
        track_type: 'straight' or 'random_turns'.
        turn_max_deg: Maximum turn angle between consecutive gates.
        arena_size: Half-extent of the arena boundary.
        arena_height: Height of the arena boundary.
    """

    n_gates: int = 3
    gate_spacing: float = 5.0
    gate_radius: float = 1.25
    track_type: str = "straight"
    turn_max_deg: float = 20.0
    arena_size: float = 50.0
    arena_height: float = 20.0


def generate_drone_urdf(config: Optional[DroneAssetConfig] = None) -> str:
    """Generate a URDF string for the quadrotor drone.

    The drone consists of a central body with four motor arms arranged
    in an X-configuration. Each arm terminates in a small sphere
    representing the motor/propeller assembly.

    Args:
        config: Drone asset configuration. Uses defaults if None.

    Returns:
        URDF XML string describing the drone.
    """
    if config is None:
        config = DroneAssetConfig()

    arm = config.arm_length
    # Motor positions in X-config (45° offsets)
    motor_positions = [
        (arm * math.cos(math.radians(a)), arm * math.sin(math.radians(a)), 0.0)
        for a in [45, 135, 225, 315]
    ]

    motor_links = ""
    motor_joints = ""
    for i, (mx, my, mz) in enumerate(motor_positions):
        motor_links += f"""
  <link name="motor_{i}">
    <inertial>
      <mass value="0.01"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
    <visual>
      <geometry><sphere radius="{config.motor_radius}"/></geometry>
      <material name="motor_mat"><color rgba="0.2 0.2 0.2 1.0"/></material>
    </visual>
    <collision>
      <geometry><sphere radius="{config.motor_radius}"/></geometry>
    </collision>
  </link>"""

        motor_joints += f"""
  <joint name="motor_joint_{i}" type="fixed">
    <parent link="base_link"/>
    <child link="motor_{i}"/>
    <origin xyz="{mx:.6f} {my:.6f} {mz:.6f}" rpy="0 0 0"/>
  </joint>"""

    urdf = f"""<?xml version="1.0" encoding="utf-8"?>
<robot name="quadrotor">
  <link name="base_link">
    <inertial>
      <mass value="{config.mass}"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="{config.ixx}" ixy="0" ixz="0"
               iyy="{config.iyy}" iyz="0" izz="{config.izz}"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="{config.body_radius}" length="{config.body_height}"/>
      </geometry>
      <material name="body_mat"><color rgba="0.3 0.3 0.8 1.0"/></material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="{config.body_radius}" length="{config.body_height}"/>
      </geometry>
    </collision>
  </link>
{motor_links}
{motor_joints}
</robot>
"""
    return urdf


def write_urdf_to_temp(urdf_str: str, filename: str = "drone.urdf") -> str:
    """Write a URDF string to a temporary file.

    Args:
        urdf_str: URDF XML string.
        filename: Name for the temporary file.

    Returns:
        Absolute path to the temporary URDF file.
    """
    tmp_dir = tempfile.mkdtemp(prefix="isaac_gym_assets_")
    path = os.path.join(tmp_dir, filename)
    with open(path, "w") as f:
        f.write(urdf_str)
    return path


def generate_gate_positions(
    config: Optional[TrackConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """Generate gate positions and orientations for a racing track.

    Args:
        config: Track configuration. Uses defaults if None.
        rng: NumPy random generator for reproducibility.

    Returns:
        List of dicts with keys: 'center' (3D position), 'normal' (3D unit vector),
        'radius' (float).
    """
    if config is None:
        config = TrackConfig()
    if rng is None:
        rng = np.random.default_rng()

    gates = []
    pos = np.array([0.0, 0.0, 2.0])  # Start position
    heading = 0.0  # Heading angle in radians

    for i in range(config.n_gates):
        if config.track_type == "random_turns" and i > 0:
            turn = rng.uniform(
                -math.radians(config.turn_max_deg),
                math.radians(config.turn_max_deg),
            )
            heading += turn

        # Move forward by gate_spacing
        dx = config.gate_spacing * math.cos(heading)
        dy = config.gate_spacing * math.sin(heading)
        pos = pos + np.array([dx, dy, 0.0])

        # Gate normal points along the heading direction
        normal = np.array([math.cos(heading), math.sin(heading), 0.0])

        gates.append({
            "center": pos.copy(),
            "normal": normal.copy(),
            "radius": config.gate_radius,
        })

    return gates


def generate_gate_urdf(config: Optional[GateAssetConfig] = None) -> str:
    """Generate a URDF string for a gate (ring) obstacle.

    The gate is approximated as a ring of small cylinders.

    Args:
        config: Gate asset configuration. Uses defaults if None.

    Returns:
        URDF XML string describing the gate.
    """
    if config is None:
        config = GateAssetConfig()

    segments = ""
    joints = ""
    for i in range(config.n_segments):
        angle = 2.0 * math.pi * i / config.n_segments
        x = config.radius * math.cos(angle)
        z = config.radius * math.sin(angle)
        # Each segment is a small cylinder oriented tangent to the ring
        seg_length = 2.0 * math.pi * config.radius / config.n_segments
        # Rotation to align the cylinder along the ring tangent
        tangent_angle = angle + math.pi / 2.0

        segments += f"""
  <link name="seg_{i}">
    <inertial>
      <mass value="0.001"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1e-8" ixy="0" ixz="0" iyy="1e-8" iyz="0" izz="1e-8"/>
    </inertial>
    <visual>
      <geometry><cylinder radius="{config.tube_radius}" length="{seg_length:.6f}"/></geometry>
      <material name="gate_mat"><color rgba="{config.color[0]} {config.color[1]} {config.color[2]} {config.color[3]}"/></material>
    </visual>
    <collision>
      <geometry><cylinder radius="{config.tube_radius}" length="{seg_length:.6f}"/></geometry>
    </collision>
  </link>"""

        joints += f"""
  <joint name="seg_joint_{i}" type="fixed">
    <parent link="gate_base"/>
    <child link="seg_{i}"/>
    <origin xyz="{x:.6f} 0 {z:.6f}" rpy="0 {tangent_angle:.6f} 0"/>
  </joint>"""

    urdf = f"""<?xml version="1.0" encoding="utf-8"?>
<robot name="gate">
  <link name="gate_base">
    <inertial>
      <mass value="0.001"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1e-8" ixy="0" ixz="0" iyy="1e-8" iyz="0" izz="1e-8"/>
    </inertial>
  </link>
{segments}
{joints}
</robot>
"""
    return urdf
