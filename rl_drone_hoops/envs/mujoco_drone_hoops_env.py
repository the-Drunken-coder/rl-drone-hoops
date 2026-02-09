"""MuJoCo-based racing drone through hoops environment with realistic sensor/action latency."""
from __future__ import annotations

import logging
import os

# Headless default: if there's no DISPLAY and MUJOCO_GL isn't set, prefer EGL.
# (Can be overridden by explicitly setting MUJOCO_GL=osmesa/glfw/egl.)
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TypedDict

import numpy as np

from rl_drone_hoops.constants import (
    ACTION_DIM,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CONTROL_HZ,
    DEFAULT_IMU_HZ,
    DEFAULT_PHYSICS_HZ,
    EPSILON,
    MAX_CACHED_RENDERERS,
    MAX_TILT_DEG,
    MAX_REWARD_WEIGHT,
    MIN_REWARD_WEIGHT,
    XML_CAMERA_NAME,
    XML_DRONE_BODY_NAME,
    XML_GATE_BODY_PREFIX,
    XML_GATE_SEG_PREFIX,
)

logger = logging.getLogger(__name__)

try:
    import mujoco
except Exception as e:  # pragma: no cover
    mujoco = None  # type: ignore
    _MUJOCO_IMPORT_ERR = e

import gymnasium as gym
from gymnasium import spaces

from rl_drone_hoops.utils.math3d import mat_to_rpy, quat_from_two_vectors, quat_to_mat, unit
from rl_drone_hoops.utils.timed_buffer import TimedDeliveryBuffer
from rl_drone_hoops.mujoco_assets.xml_builder import build_drone_hoops_xml


class ObservationDict(TypedDict):
    """Observation structure for the environment."""

    image: np.ndarray  # (H, W, 1) uint8 grayscale
    imu: np.ndarray  # (T, 6) float32, columns are [gx, gy, gz, ax, ay, az] in body frame
    last_action: np.ndarray  # (4,) float32, normalized action from previous step


@dataclass
class Gate:
    """Gate (hoop) specification.

    Attributes:
        center: 3D position of gate center
        normal: Unit normal vector pointing through the gate
        radius: Gate radius (must be > 0)
    """

    center: np.ndarray  # (3,)
    normal: np.ndarray  # (3,) unit
    radius: float

    def __post_init__(self) -> None:
        """Validate gate parameters."""
        if self.radius <= 0:
            raise ValueError(f"Gate radius must be positive, got {self.radius}")
        if not (0.99 <= np.linalg.norm(self.normal) <= 1.01):
            raise ValueError(f"Gate normal must be unit length, got norm {np.linalg.norm(self.normal)}")


def _asset_path(rel: str) -> str:
    """Get absolute path to an asset relative to this module."""
    here = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(here, "..", "mujoco_assets", rel))


class MujocoDroneHoopsEnv(gym.Env):
    """MuJoCo racing drone through hoops with realistic sensor/action latency.

    Observations:
        - Grayscale FPV camera (96x96)
        - IMU (gyro + accel) window history
        - Last normalized action

    Actions:
        - 4D normalized action: [roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd, thrust_cmd]
        - All values in [-1, 1], scaled to physical limits

    Rewards:
        - +10 per gate passed
        - +0.1 per timestep (survival bonus)
        - Shaping: progress toward gate, centering, speed, smoothness, stability
        - -20 on crash

    Notes:
        - Forces/torques applied via mj_applyFT each physics step
        - Realistic latencies: 30ms camera, 3ms IMU, 10ms actuator
        - Thrust and angular rate limits to avoid instant flips
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
        cam_latency_s: float = 0.03,
        imu_latency_s: float = 0.003,
        act_latency_s: float = 0.01,
        episode_s: float = 15.0,
        bounds_xyz: Tuple[float, float, float] = (20.0, 20.0, 10.0),
        n_gates: int = 5,
        gate_radius: float = 0.75,
        gate_spacing: float = 4.0,
        track_type: str = "straight",
        turn_max_deg: float = 25.0,
        gate_y_range: float = 3.0,
        gate_z_range: Tuple[float, float] = (1.0, 2.0),
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if mujoco is None:  # pragma: no cover
            raise ImportError(
                "MuJoCo python package is not available. Install with `pip install mujoco`.\n"
                f"Original import error: {_MUJOCO_IMPORT_ERR}"
            )

        self.rng = np.random.default_rng(seed)

        self.image_size = int(image_size)
        # Some platforms/driver combos produce a 90deg-rolled FPV image. Keep this configurable.
        # Override with RL_DRONE_HOOPS_IMAGE_ROT90=0..3.
        env_rot = os.environ.get("RL_DRONE_HOOPS_IMAGE_ROT90")
        self.image_rot90 = (int(env_rot) if env_rot is not None else int(image_rot90)) % 4
        self.camera_fps = float(camera_fps)
        self.imu_hz = float(imu_hz)
        self.control_hz = float(control_hz)
        self.physics_hz = float(physics_hz)

        self.dt_phys = 1.0 / self.physics_hz
        self.dt_control = 1.0 / self.control_hz

        self.cam_latency_s = float(cam_latency_s)
        self.imu_latency_s = float(imu_latency_s)
        self.act_latency_s = float(act_latency_s)

        self.episode_s = float(episode_s)
        self.max_steps = int(np.ceil(self.episode_s * self.control_hz))
        self.bounds_xyz = np.array(bounds_xyz, dtype=np.float64)

        self.n_gates = int(n_gates)
        self.gate_radius = float(gate_radius)
        self.gate_spacing = float(gate_spacing)
        self.track_type = str(track_type)
        self.turn_max_deg = float(turn_max_deg)
        self.gate_y_range = float(gate_y_range)
        self.gate_z_range = (float(gate_z_range[0]), float(gate_z_range[1]))
        self._xml_max_gates = max(16, self.n_gates)
        self._xml_ring_segments = 16

        # Dynamics / control parameters (MVP; will be tuned).
        # Conservative limits to avoid instant flips early in training.
        self.max_rates = np.array([4.0, 4.0, 6.0], dtype=np.float64)  # rad/s
        self.thrust_min = 0.0
        self.thrust_max = 25.0  # N (roughly for a ~1 kg class drone)
        self.rate_kp = np.array([0.025, 0.025, 0.02], dtype=np.float64)  # torque per (rad/s) error
        self.torque_max = np.array([0.35, 0.35, 0.2], dtype=np.float64)  # N*m
        self.lin_drag = 0.15
        self.quad_drag = 0.04
        self.ang_drag = 0.05
        self.thrust_tau = 0.05  # seconds, 1st-order thrust lag

        # Reward weights (MVP).
        # Add a small survival/stability reward to avoid the "crash immediately" local optimum.
        self.r_alive = 0.1
        self.r_gate = 10.0
        self.k_progress = 1.5
        self.k_center = 0.5
        self.k_speed = 0.02
        self.k_smooth = 0.02
        self.k_tilt = 0.05  # penalize roll/pitch magnitude (rad)
        # Keep this bounded to avoid huge negative returns when the drone is already unstable.
        self.k_angrate = 0.0002  # penalize body angular rate magnitude
        self.r_crash = -20.0

        # Validate reward weights
        self._validate_reward_weights()

        # Observation: camera + IMU window + last action.
        # IMU window spans last 0.1s of delivered samples, padded if needed.
        self.imu_window_s = 0.10
        self.imu_window_n = int(np.ceil(self.imu_hz * self.imu_window_s))

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(self.image_size, self.image_size, 1), dtype=np.uint8),
                "imu": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.imu_window_n, 6), dtype=np.float32
                ),  # [gx,gy,gz, ax,ay,az] in body
                "last_action": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )

        # Action: normalized in [-1,1], then scaled to rates/thrust.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        xml = build_drone_hoops_xml(
            max_gates=self._xml_max_gates,
            ring_radius=self.gate_radius,
            tube_radius=0.05,
            ring_segments=self._xml_ring_segments,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.renderer = mujoco.Renderer(self.model, height=self.image_size, width=self.image_size)
        # Optional renderers for higher-res debug/eval videos (LRU cache to avoid VRAM leak).
        self._extra_renderers: Dict[Tuple[int, int], mujoco.Renderer] = {}

        # Camera and drone body IDs (validate they exist)
        self._fpv_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, XML_CAMERA_NAME)
        if self._fpv_cam_id < 0:
            raise RuntimeError(f"Camera '{XML_CAMERA_NAME}' not found in MuJoCo model")

        self._drone_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, XML_DRONE_BODY_NAME)
        if self._drone_bid < 0:
            raise RuntimeError(f"Drone body '{XML_DRONE_BODY_NAME}' not found in MuJoCo model")

        self._drone_mass = float(self.model.body_mass[self._drone_bid])
        # Nominal hover thrust for warm-start at reset.
        self._hover_thrust = float(np.clip(self._drone_mass * 9.81, self.thrust_min, self.thrust_max))
        self._hover_thrust_norm = (
            2.0 * (self._hover_thrust - self.thrust_min) / max(self.thrust_max - self.thrust_min, 1e-9) - 1.0
        )

        # Gate body/mocap + geom ids for visualization.
        self._gate_body_ids: list[int] = []
        self._gate_mocap_ids: list[int] = []
        self._gate_geom_ids: list[list[int]] = []
        for i in range(self._xml_max_gates):
            gate_name = f"{XML_GATE_BODY_PREFIX}{i}"
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, gate_name)
            if bid < 0:
                raise RuntimeError(f"Gate body '{gate_name}' not found in MuJoCo model")
            self._gate_body_ids.append(int(bid))
            mid = int(self.model.body_mocapid[bid])
            if mid < 0:
                raise RuntimeError(f"Gate body '{gate_name}' is not mocap-enabled (body_mocapid={mid})")
            self._gate_mocap_ids.append(mid)
            seg_ids: list[int] = []
            for j in range(self._xml_ring_segments):
                seg_name = f"{gate_name}_{XML_GATE_SEG_PREFIX}{j}"
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, seg_name)
                if gid < 0:
                    raise RuntimeError(f"Gate segment geom '{seg_name}' not found in MuJoCo model")
                seg_ids.append(int(gid))
            self._gate_geom_ids.append(seg_ids)

        # Sensor timing.
        self._next_cam_capture = 0.0
        self._next_imu_capture = 0.0

        self._cam_buf: TimedDeliveryBuffer[np.ndarray] = TimedDeliveryBuffer()
        self._imu_buf: TimedDeliveryBuffer[np.ndarray] = TimedDeliveryBuffer()

        self._last_image = np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
        self._imu_history: list[np.ndarray] = []

        # Action latency.
        self._act_queue: list[tuple[float, np.ndarray]] = []  # (apply_ts, action_scaled)
        self._applied_action = np.zeros(4, dtype=np.float64)
        self._thrust_state = 0.0  # lagged thrust (N)

        # Episode state.
        self._t = 0.0
        self._step_i = 0
        self._last_action_norm = np.zeros(4, dtype=np.float32)

        self.gates: list[Gate] = []
        self._next_gate_idx = 0

        self._p_prev = np.zeros(3, dtype=np.float64)
        self._v_prev = np.zeros(3, dtype=np.float64)

        self._build_track()
        self._sync_gate_sites()

    # ---------------------------
    # Validation
    # ---------------------------
    def _validate_reward_weights(self) -> None:
        """Validate reward weights are within reasonable bounds."""
        weights = {
            "r_alive": self.r_alive,
            "r_gate": self.r_gate,
            "k_progress": self.k_progress,
            "k_center": self.k_center,
            "k_speed": self.k_speed,
            "k_smooth": self.k_smooth,
            "k_tilt": self.k_tilt,
            "k_angrate": self.k_angrate,
        }
        for name, val in weights.items():
            if not MIN_REWARD_WEIGHT <= val <= MAX_REWARD_WEIGHT:
                raise ValueError(
                    f"Reward weight '{name}' = {val} outside [{MIN_REWARD_WEIGHT}, {MAX_REWARD_WEIGHT}]"
                )

    # ---------------------------
    # Track + visualization sites
    # ---------------------------
    def _build_track(self) -> None:
        self.gates = []

        if self.track_type == "straight":
            # Straight gates in +X with random Y/Z offsets.
            for i in range(self.n_gates):
                x = (i + 1) * self.gate_spacing
                y = self.rng.uniform(-min(1.0, self.gate_y_range), min(1.0, self.gate_y_range))
                z = self.rng.uniform(*self.gate_z_range)
                center = np.array([x, y, z], dtype=np.float64)
                normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                self.gates.append(Gate(center=center, normal=unit(normal), radius=self.gate_radius))
            return

        if self.track_type == "random_turns":
            # Sequentially sample gate centers with bounded turn angle and keep within bounds.
            # Gate normal points along the local segment direction.
            pos = np.array([self.gate_spacing, 0.0, float(np.mean(self.gate_z_range))], dtype=np.float64)
            dir_vec = unit(np.array([1.0, 0.0, 0.0], dtype=np.float64))

            max_turn = np.deg2rad(self.turn_max_deg)
            for i in range(self.n_gates):
                # Sample yaw/pitch change within bounds.
                yaw = self.rng.uniform(-max_turn, max_turn)
                pitch = self.rng.uniform(-0.35 * max_turn, 0.35 * max_turn)
                cy, sy = np.cos(yaw), np.sin(yaw)
                cp, sp = np.cos(pitch), np.sin(pitch)
                # Rotate dir_vec by yaw about Z then pitch about Y (approx in world frame).
                Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
                dir_vec = unit(Rz @ (Ry @ dir_vec))

                # Step forward and add bounded lateral/vertical variation.
                candidate = pos + dir_vec * self.gate_spacing
                candidate[1] += self.rng.uniform(-self.gate_y_range, self.gate_y_range)
                candidate[2] = self.rng.uniform(*self.gate_z_range)

                # Clamp to bounds for stability.
                candidate[0] = np.clip(candidate[0], -self.bounds_xyz[0] + 1.0, self.bounds_xyz[0] - 1.0)
                candidate[1] = np.clip(candidate[1], -self.bounds_xyz[1] + 1.0, self.bounds_xyz[1] - 1.0)
                candidate[2] = np.clip(candidate[2], 0.5, self.bounds_xyz[2] - 0.5)

                pos = candidate
                self.gates.append(Gate(center=pos.copy(), normal=dir_vec.copy(), radius=self.gate_radius))
            return

        raise ValueError(f"Unknown track_type={self.track_type!r}. Expected 'straight' or 'random_turns'.")

    def _sync_gate_sites(self) -> None:
        # Place mocap gate bodies and update their alpha to highlight the next gate.
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        for i in range(self._xml_max_gates):
            mid = self._gate_mocap_ids[i]
            # Hide gates that are not used in this episode.
            if i >= len(self.gates):
                self.data.mocap_pos[mid] = np.array([0.0, 0.0, -100.0], dtype=np.float64)
                self.data.mocap_quat[mid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                alpha = 0.0
            else:
                gate = self.gates[i]
                self.data.mocap_pos[mid] = gate.center.astype(np.float64)
                self.data.mocap_quat[mid] = quat_from_two_vectors(x_axis, gate.normal.astype(np.float64))
                alpha = 1.0 if i == self._next_gate_idx else 0.25

            for gid in self._gate_geom_ids[i]:
                self.model.geom_rgba[gid, 3] = alpha

    # ---------------------------
    # Sensors
    # ---------------------------
    def _render_fpv_gray(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self._fpv_cam_id)
        rgb = self.renderer.render()
        # High-contrast grayscale: use the red channel so the red gate is bright in obs.
        # (Luma-weighted grayscale makes red relatively dark.)
        gray = rgb[..., 0].astype(np.uint8)
        if self.image_rot90:
            gray = np.rot90(gray, k=self.image_rot90)
        return gray[..., None]

    def _capture_sensors_if_due(self) -> None:
        # Camera capture at fixed FPS.
        cam_dt = 1.0 / self.camera_fps
        while self._t + 1e-12 >= self._next_cam_capture:
            img = self._render_fpv_gray()
            cap = self._next_cam_capture
            self._cam_buf.push(deliver_ts=cap + self.cam_latency_s, capture_ts=cap, payload=img)
            self._next_cam_capture += cam_dt

        # IMU capture.
        imu_dt = 1.0 / self.imu_hz
        while self._t + 1e-12 >= self._next_imu_capture:
            cap = self._next_imu_capture
            imu = self._compute_imu_sample()
            self._imu_buf.push(deliver_ts=cap + self.imu_latency_s, capture_ts=cap, payload=imu)
            self._next_imu_capture += imu_dt

    def _deliver_sensors(self) -> None:
        for it in self._cam_buf.pop_ready(self._t):
            self._last_image = it.payload
        for it in self._imu_buf.pop_ready(self._t):
            self._imu_history.append(it.payload)
        # Keep enough history for the window, plus a little slack.
        max_keep = int(self.imu_window_n * 2)
        if len(self._imu_history) > max_keep:
            self._imu_history = self._imu_history[-max_keep:]

    def _compute_imu_sample(self) -> np.ndarray:
        # Drone pose/vel from freejoint (qpos: 7, qvel: 6).
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:6].copy()

        p = qpos[0:3]
        q = qpos[3:7]  # (w,x,y,z)
        R = quat_to_mat(q)

        v_world = qvel[0:3]
        w_world = qvel[3:6]
        w_body = R.T @ w_world

        # Linear acceleration (finite-diff).
        a_world = (v_world - self._v_prev) / max(self.dt_phys, 1e-9)
        self._v_prev = v_world.copy()

        g = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        a_body = R.T @ (a_world - g)

        return np.concatenate([w_body, a_body], axis=0).astype(np.float32)

    def _obs(self) -> ObservationDict:
        """Construct observation dict from current sensor state."""
        imu = np.zeros((self.imu_window_n, 6), dtype=np.float32)
        if self._imu_history:
            take = min(self.imu_window_n, len(self._imu_history))
            imu[-take:, :] = np.stack(self._imu_history[-take:], axis=0)
        return {
            "image": self._last_image.copy(),
            "imu": imu,
            "last_action": self._last_action_norm.copy(),
        }

    # ---------------------------
    # Control / dynamics injection
    # ---------------------------
    def _scale_action(self, a_norm: np.ndarray) -> np.ndarray:
        a = np.clip(a_norm.astype(np.float64), -1.0, 1.0)
        rates = a[0:3] * self.max_rates
        thrust = (a[3] + 1.0) * 0.5 * (self.thrust_max - self.thrust_min) + self.thrust_min
        return np.array([rates[0], rates[1], rates[2], thrust], dtype=np.float64)

    def _update_applied_action(self) -> None:
        # Apply the newest queued action whose apply_ts has passed.
        if not self._act_queue:
            return
        i = 0
        while i < len(self._act_queue) and self._act_queue[i][0] <= self._t + 1e-12:
            self._applied_action = self._act_queue[i][1]
            i += 1
        if i:
            self._act_queue = self._act_queue[i:]

    def _apply_forces_and_step(self) -> None:
        # Update action if delayed.
        self._update_applied_action()

        # External generalized forces persist in `data.qfrc_applied` unless cleared.
        # We use `mj_applyFT` (additive), so reset to avoid accumulating forces across timesteps.
        self.data.qfrc_applied[:] = 0.0

        # Read current state.
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:6].copy()
        q = qpos[3:7]
        R = quat_to_mat(q)
        v_world = qvel[0:3]
        w_world = qvel[3:6]
        w_body = R.T @ w_world

        # Desired rates in body, thrust in N.
        w_cmd = self._applied_action[0:3]
        thrust_cmd = float(self._applied_action[3])

        # Thrust lag.
        alpha = self.dt_phys / max(self.thrust_tau, 1e-6)
        alpha = np.clip(alpha, 0.0, 1.0)
        self._thrust_state = (1.0 - alpha) * self._thrust_state + alpha * thrust_cmd

        # Rate P control -> body torque, then map to world.
        w_err = w_cmd - w_body
        tau_body = np.clip(self.rate_kp * w_err, -self.torque_max, self.torque_max)
        tau_world = R @ tau_body

        # Thrust in world along body +Z.
        f_world = R @ np.array([0.0, 0.0, self._thrust_state], dtype=np.float64)

        # Simple drag in world.
        v = v_world
        speed = float(np.linalg.norm(v))
        f_drag = -self.lin_drag * v - self.quad_drag * speed * v
        tau_drag_world = -self.ang_drag * w_world

        f_total = f_world + f_drag
        tau_total = tau_world + tau_drag_world

        # Apply at COM.
        # mujoco.mj_applyFT expects (3,1) float64 arrays and writes into qfrc_target.
        point = self.data.xipos[self._drone_bid].copy()
        mujoco.mj_applyFT(
            self.model,
            self.data,
            np.asarray(f_total, dtype=np.float64).reshape(3, 1),
            np.asarray(tau_total, dtype=np.float64).reshape(3, 1),
            np.asarray(point, dtype=np.float64).reshape(3, 1),
            int(self._drone_bid),
            self.data.qfrc_applied,
        )

        mujoco.mj_step(self.model, self.data)
        self._t += self.dt_phys

    # ---------------------------
    # Reward + termination
    # ---------------------------
    def _gate_passed(self, gate: Gate, p_prev: np.ndarray, p_curr: np.ndarray) -> bool:
        """Check if drone passed through gate between two positions.

        Handles crossing in either direction (forward or backward).

        Args:
            gate: Gate to check crossing
            p_prev: Previous position
            p_curr: Current position

        Returns:
            True if drone crossed through the gate plane within gate.radius
        """
        g = gate.center
        n = gate.normal
        s_prev = float(np.dot(n, p_prev - g))
        s_curr = float(np.dot(n, p_curr - g))

        # Detect crossing in either direction
        crossed = (s_prev < 0.0 and s_curr >= 0.0) or (s_prev >= 0.0 and s_curr < 0.0)
        if not crossed:
            return False

        # Find crossing interpolation factor.
        denom = s_curr - s_prev
        if abs(denom) < EPSILON:
            return False
        t = -s_prev / denom
        t = float(np.clip(t, 0.0, 1.0))

        p_cross = p_prev + t * (p_curr - p_prev)
        # Radial distance from center in plane: remove normal component.
        d = p_cross - g
        d_perp = d - np.dot(d, n) * n
        return float(np.linalg.norm(d_perp)) <= gate.radius

    def _compute_reward_and_done(
        self, a_norm: np.ndarray, a_prev_norm: np.ndarray, p_prev: np.ndarray
    ) -> Tuple[float, bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}

        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:6].copy()
        p = qpos[0:3]
        v = qvel[0:3]
        q = qpos[3:7]
        R = quat_to_mat(q)
        rpy = mat_to_rpy(R)

        terminated = False
        truncated = False

        # Bounds / crash heuristics (MVP).
        if np.any(np.abs(p[0:2]) > self.bounds_xyz[0:2]) or p[2] < 0.05 or p[2] > self.bounds_xyz[2]:
            terminated = True
            info["crash"] = True

        max_tilt = np.deg2rad(MAX_TILT_DEG)
        if abs(float(rpy[0])) > max_tilt or abs(float(rpy[1])) > max_tilt:
            terminated = True
            info["crash"] = True

        if self._step_i >= self.max_steps:
            truncated = True

        # Gate logic.
        gate_bonus = 0.0
        passed = False
        if not (terminated or truncated) and self._next_gate_idx < len(self.gates):
            gate = self.gates[self._next_gate_idx]
            if self._gate_passed(gate, p_prev, p):
                self._next_gate_idx += 1
                gate_bonus = self.r_gate
                passed = True
                self._sync_gate_sites()

        done_track = self._next_gate_idx >= len(self.gates)
        if done_track:
            terminated = True
            info["finished"] = True
            self._sync_gate_sites()

        # Shaping relative to next gate (if any).
        shaping = 0.0
        if self._next_gate_idx < len(self.gates):
            gate = self.gates[self._next_gate_idx]
            d_prev = float(np.linalg.norm(p_prev - gate.center))
            d_curr = float(np.linalg.norm(p - gate.center))
            shaping += self.k_progress * (d_prev - d_curr)

            # Centering penalty based on distance to gate axis at current position.
            d = p - gate.center
            d_perp = d - np.dot(d, gate.normal) * gate.normal
            radial = float(np.linalg.norm(d_perp))
            # Gate radius is validated to be > 0 in Gate.__post_init__, but use EPSILON as safeguard
            shaping += -self.k_center * (radial / max(gate.radius, EPSILON))

            # Speed toward gate.
            to_gate = unit(gate.center - p)
            v_toward = float(np.dot(v, to_gate))
            shaping += self.k_speed * np.clip(v_toward, 0.0, 20.0)

        # Smoothness penalty.
        da = a_norm.astype(np.float64) - a_prev_norm.astype(np.float64)
        smooth_pen = -self.k_smooth * float(np.dot(da, da))

        # Stability shaping.
        tilt_pen = -self.k_tilt * (abs(float(rpy[0])) + abs(float(rpy[1])))
        w_world = qvel[3:6]
        w_body = R.T @ w_world
        w2 = float(np.dot(w_body, w_body))
        ang_pen = -self.k_angrate * min(w2, 400.0)

        reward = self.r_alive + gate_bonus + shaping + smooth_pen + tilt_pen + ang_pen

        if terminated and info.get("crash", False):
            reward += self.r_crash

        info.update(
            {
                "t": self._t,
                "gate_passed": passed,
                "next_gate_idx": self._next_gate_idx,
                "reward_alive": float(self.r_alive),
                "reward_gate": float(gate_bonus),
                "reward_shaping": float(shaping),
                "reward_smooth": float(smooth_pen),
                "reward_tilt": float(tilt_pen),
                "reward_angrate": float(ang_pen),
                "pos": p.copy(),
                "vel": v.copy(),
            }
        )
        return float(reward), terminated, truncated, info

    # ---------------------------
    # Gym API
    # ---------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state.

        Args:
            seed: Random seed for episode randomization
            options: Dict with optional keys:
                - fixed_track (bool): If True, reuse the same track (default False)

        Returns:
            observation: Initial observation (dict with keys image, imu, last_action)
            info: Dict with next_gate_idx
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._t = 0.0
        self._step_i = 0

        # Rebuild track each episode unless options specify fixed.
        if options is None:
            options = {}
        fixed = bool(options.get("fixed_track", False))
        if not fixed:
            self._build_track()
        self._next_gate_idx = 0
        # mj_resetData restores mocap bodies to their XML defaults (gates are hidden at z=-100).
        # Sync gate visuals every reset so FPV images/videos actually show the hoops.
        self._sync_gate_sites()

        # Reset timing.
        self._next_cam_capture = 0.0
        self._next_imu_capture = 0.0
        self._cam_buf = TimedDeliveryBuffer()
        self._imu_buf = TimedDeliveryBuffer()
        self._last_image = np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
        self._imu_history = []

        self._act_queue = []
        self._applied_action = np.array([0.0, 0.0, 0.0, self._hover_thrust], dtype=np.float64)
        self._thrust_state = float(self._hover_thrust)
        self._last_action_norm = np.array([0.0, 0.0, 0.0, self._hover_thrust_norm], dtype=np.float32)

        # Initialize prev state.
        qpos = self.data.qpos[:7].copy()
        self._p_prev = qpos[0:3].copy()
        self._v_prev = self.data.qvel[:3].copy()

        # Warm up sensors so we return a non-empty obs.
        for _ in range(int(self.physics_hz * 0.05)):  # 50 ms
            self._apply_forces_and_step()
            self._capture_sensors_if_due()
            self._deliver_sensors()

        obs = self._obs()
        info = {"next_gate_idx": self._next_gate_idx}
        return obs, info

    def step(self, action: np.ndarray):
        """Execute one control step in the environment.

        Args:
            action: 4D action array [roll_rate, pitch_rate, yaw_rate, thrust]
                    values should be in [-1, 1]

        Returns:
            observation: Dict with keys image, imu, last_action
            reward: Scalar reward
            terminated: Boolean (episode ended due to termination condition)
            truncated: Boolean (episode ended due to max steps)
            info: Dict with diagnostic info
        """
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Expected action shape ({ACTION_DIM},), got {action.shape}")

        a_norm = np.clip(action.astype(np.float32), -1.0, 1.0)
        a_prev = self._last_action_norm.copy()

        # Queue scaled action with action latency.
        a_scaled = self._scale_action(a_norm)
        self._act_queue.append((self._t + self.act_latency_s, a_scaled))

        # Store p_prev for gate crossing and progress shaping.
        p_prev = self.data.qpos[:3].copy()

        # Simulate for one control interval.
        n_phys = int(np.round(self.dt_control / self.dt_phys))
        n_phys = max(1, n_phys)
        for _ in range(n_phys):
            self._apply_forces_and_step()
            self._capture_sensors_if_due()
            self._deliver_sensors()

        reward, terminated, truncated, info = self._compute_reward_and_done(a_norm, a_prev, p_prev)
        self._step_i += 1
        self._last_action_norm = a_norm.copy()

        obs = self._obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        # Return current FPV RGB for debugging if needed.
        self.renderer.update_scene(self.data, camera=self._fpv_cam_id)
        rgb = self.renderer.render()
        if self.image_rot90:
            rgb = np.rot90(rgb, k=self.image_rot90)
        return rgb

    def render_rgb(self, *, height: int, width: int) -> np.ndarray:
        """Render FPV RGB at an explicit resolution (used for high-res videos).

        Uses an LRU renderer cache to avoid VRAM exhaustion. Caches up to
        MAX_CACHED_RENDERERS different resolutions.

        Args:
            height: Image height in pixels
            width: Image width in pixels

        Returns:
            RGB image as (height, width, 3) uint8 array
        """
        h = int(height)
        w = int(width)
        key = (h, w)
        r = self._extra_renderers.get(key)
        if r is None:
            # Evict oldest renderer if cache is full (simple LRU)
            if len(self._extra_renderers) >= MAX_CACHED_RENDERERS:
                oldest_key = next(iter(self._extra_renderers))
                self._extra_renderers.pop(oldest_key)
            r = mujoco.Renderer(self.model, height=h, width=w)
            self._extra_renderers[key] = r
        r.update_scene(self.data, camera=self._fpv_cam_id)
        rgb = r.render()
        if self.image_rot90:
            rgb = np.rot90(rgb, k=self.image_rot90)
        return rgb

    def pose_rpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Current drone position and roll/pitch/yaw (radians) in world frame.
        """
        qpos = self.data.qpos[:7].copy()
        p = qpos[0:3]
        q = qpos[3:7]  # (w,x,y,z)
        R = quat_to_mat(q)
        rpy = mat_to_rpy(R)
        return p, rpy

    def close(self):
        """Clean up resources."""
        try:
            self.renderer.close()
        except Exception as e:
            logger.warning(f"Error closing main renderer: {e}")
        for key, r in list(self._extra_renderers.items()):
            try:
                r.close()
            except Exception as e:
                logger.warning(f"Error closing renderer at {key}: {e}")
        self._extra_renderers = {}
