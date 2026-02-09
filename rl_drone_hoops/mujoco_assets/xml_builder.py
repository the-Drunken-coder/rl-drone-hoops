from __future__ import annotations

import math
from typing import Iterable


def _ring_capsule_geoms_xml(
    *,
    base_name: str,
    ring_radius: float,
    tube_radius: float,
    segments: int,
    rgba: str,
) -> str:
    """
    Build a pass-through "hoop" approximation as a set of capsule segments.

    The hoop lies in the YZ plane (normal +X) in the local frame of the parent body.
    """
    segs = []
    for j in range(segments):
        t0 = 2.0 * math.pi * (j / segments)
        t1 = 2.0 * math.pi * ((j + 1) / segments)
        y0 = ring_radius * math.cos(t0)
        z0 = ring_radius * math.sin(t0)
        y1 = ring_radius * math.cos(t1)
        z1 = ring_radius * math.sin(t1)
        segs.append(
            f'<geom name="{base_name}_seg{j}" type="capsule" '
            f'fromto="0 {y0:.6f} {z0:.6f} 0 {y1:.6f} {z1:.6f}" '
            f'size="{tube_radius:.4f}" rgba="{rgba}" contype="0" conaffinity="0"/>'
        )
    return "\n      ".join(segs)


def build_drone_hoops_xml(
    *,
    max_gates: int,
    ring_radius: float,
    tube_radius: float = 0.05,
    ring_segments: int = 16,
) -> str:
    """
    Build a MuJoCo XML model string with:
    - ground plane
    - single rigid-body drone (freejoint) + fpv camera
    - preallocated mocap gate bodies (visual-only hoops)
    """
    gates_xml = []
    for i in range(max_gates):
        hoop = _ring_capsule_geoms_xml(
            base_name=f"gate{i}",
            ring_radius=ring_radius,
            tube_radius=tube_radius,
            segments=ring_segments,
            rgba="1 0.2 0.2 1",
        )
        gates_xml.append(
            f"""
    <body name="gate{i}" mocap="true" pos="0 0 -100">
      {hoop}
    </body>""".rstrip()
        )

    gates_block = "\n".join(gates_xml)

    # FPV camera quat:
    # This sets the camera to look forward along +X, but depending on the renderer/driver,
    # the rendered image may still come out rolled. The environment applies an optional
    # 90-degree rotation to keep the image upright.
    return f"""<mujoco model="drone_hoops">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="RK4"/>

  <visual>
    <quality shadowsize="2048"/>
    <map znear="0.01"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="256" height="256"/>
    <material name="gridmat" texture="grid" texrepeat="10 10" reflectance="0.0"/>
  </asset>

  <worldbody>
    <light name="sun" directional="true" pos="0 0 10" dir="0 0 -1" diffuse="1 1 1" specular="0.1 0.1 0.1"/>
    <geom name="ground" type="plane" size="50 50 0.1" material="gridmat" rgba="0.9 0.9 0.9 1" contype="1" conaffinity="1"/>

    <body name="drone" pos="0 0 1">
      <freejoint/>
      <geom name="drone_geom" type="box" size="0.08 0.08 0.02" rgba="0.1 0.1 0.1 1" contype="1" conaffinity="1"/>
      <camera name="fpv" pos="0.10 0 0.02" quat="-0.5 -0.5 0.5 0.5" fovy="140"/>
    </body>

{gates_block}
  </worldbody>
</mujoco>
"""
