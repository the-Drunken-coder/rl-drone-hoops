# Full System Spec: Pure-RL Racing Drone Through Hoops (MuJoCo)

This document is the authoritative spec for the end-to-end system. It is intentionally implementation-oriented (interfaces, timing, reward definitions), while leaving room for iterative realism improvements.

## Goals
- Train a policy to fly a quadrotor through ordered hoops/gates as fast as possible.
- Observations are sensor-based (camera + IMU), not ground-truth state (except for reward/termination computations).
- “Works in sim first” is the priority; sim-to-real comes later.

## Practical environment notes
- Python 3.9 is supported by pinning MuJoCo to `mujoco==2.3.7` (wheels available for cp39).
- Headless rendering on servers typically uses `MUJOCO_GL=egl` (preferred) or `MUJOCO_GL=osmesa` (fallback).
- Long-running training jobs should be launched under `tmux` so they survive SSH disconnects.
- Before starting a new run, operators/agents should check `tmux ls` to see if a run is already in progress.

## Non-goals (initially)
- Photorealistic rendering.
- Perfect aero/motor modeling.
- Real flight controller stack integration (PX4/Betaflight) in the first iteration.

## Top-Level Architecture
Six major subsystems:
1. Simulator + course generator (MuJoCo world, drone dynamics, gates, collisions)
2. Sensor pipeline (camera + IMU) with realistic rates + latency
3. Action interface (policy control signals + actuator dynamics + delay)
4. Reward + termination logic (gate passing, progress shaping, crash/out-of-bounds)
5. RL training loop (PPO or SAC; recurrent policy)
6. Logging + evaluation (metrics, videos, checkpoints, deterministic eval tracks)

## Training / Evaluation Artifacts (MVP implementation)
This repo includes a minimal in-house recurrent PPO baseline (PyTorch) to avoid external RL framework constraints:
- Training script: `scripts/train_recurrent_ppo.py`
- Checkpoint evaluator: `scripts/eval_checkpoint.py`
- Training deps: `requirements-train.txt` (kept separate to avoid pinning torch+CUDA)
- Checkpoints: `runs/<run>/checkpoints/stepXXXXXXXXX.pt`
- TensorBoard logs: `runs/<run>/tb/`
- Eval videos: `videos/` (MP4, first eval episode)
  - Default location is `./videos` relative to where you launch training.
  - Override with `RL_DRONE_HOOPS_VIDEO_DIR=/abs/path/to/videos`.
  - If images appear rotated, set `RL_DRONE_HOOPS_IMAGE_ROT90=0..3` (CCW 90deg steps).
  - Calibration helper: `scripts/generate_calibration_videos.py --all` writes `calib_rot{0,1,2,3}.mp4`.
  - Quality/overlay env vars:
    - `RL_DRONE_HOOPS_VIDEO_SIZE=256` (square MP4 resolution)
    - `RL_DRONE_HOOPS_VIDEO_FPS=30`
    - `RL_DRONE_HOOPS_VIDEO_OVERLAY=1`

## Environment API (Gymnasium-style)
The environment exposes:
- `reset(seed=None) -> obs, info`
- `step(action) -> obs, reward, terminated, truncated, info`

### Observation (sensor-only)
Observation is built at the control rate (e.g., 100 Hz) from buffered sensors:
- `camera`: latest *delivered* grayscale image (default 96x96; configurable), dtype `uint8` or normalized `float32`
  - Implementation note: grayscale is derived from the FPV render using a high-contrast mapping (red channel) so the red hoop is visible at low resolution.
- `imu`: fixed window of IMU samples (gyro + accel), delivered with latency
- Optional but recommended:
  - `last_action`
  - `dt_control` or timestamps deltas (to make rate jitter survivable)

Ground-truth state (pose/vel/etc.) must not be included in `obs` for the main training run.

### Action (body rates + thrust)
Action is continuous:
- `[roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd, thrust_cmd]`

The simulator applies:
- Saturation/limits on rates and thrust
- Actuation dynamics (e.g., first-order lag on thrust and/or body-rate response)
- Optional action delay via FIFO queue (recommended)

## Timing Model (authoritative)
We explicitly model multi-rate simulation.

### Rates
- Physics integration: 500–1000 Hz (MuJoCo internal timestep)
- Control/action update: 100 Hz (default target)
- IMU: 400 Hz (target range 200–500 Hz)
- Camera: 60 FPS (given)

### Latency
Latencies are modeled as delivery delay relative to capture time:
- Camera latency: `L_cam` (configurable; initial default 20–40 ms)
- IMU latency: `L_imu` (small; initial default 2–5 ms)
- Action latency: `L_act` (configurable; initial default 0–20 ms)

### Implementation concept (buffers)
- Maintain a truth-state ring buffer keyed by physics timestamps.
- Sensor buffers store `(capture_ts, payload)` and deliver at `capture_ts + L_*`.
- `step()` assembles an observation using the most recent delivered camera frame and a fixed-length IMU window aligned to the control step.

## Simulator + Drone Model
### Minimal viable drone dynamics
Phase 1 (stable learning):
- Rigid body with thrust along body z-axis
- Body torque inputs from a simple rate controller tracking commanded body rates
- Drag term (linear + quadratic optional) to prevent unrealistic perpetual acceleration
- Thrust and rate limits + lag

Phase 2 (racing feel improvements):
- Motor saturation more explicit (e.g., thrust-to-torque mapping)
- Better drag model (body-axis dependent)
- Ground effect optional (later)

### Collisions / bounds
- Collide with ground, gate frames (if modeled), optional obstacles.
- Out-of-bounds volume termination to keep training stable.

## Course / Gate Representation
Each gate `i` has:
- Center position `g_i`
- Plane normal `n_i` (unit vector)
- Radius `r_i` (clear radius for passing)
- Optional thickness / frame geometry for collisions
- Ordering index `i` (the “next gate” defines progress)

### Gate passing detection (event)
Gate `i` is counted as passed if:
1. The drone crosses the gate plane in the correct direction:
   - `s_prev = dot(n_i, p_prev - g_i)`
   - `s_curr = dot(n_i, p_curr - g_i)`
   - crossing if `s_prev < 0` and `s_curr >= 0`
2. The crossing point is within radius:
   - Compute linear interpolation `p_cross` between `p_prev` and `p_curr` where plane is crossed.
   - Pass if `||p_cross - g_i||_perp <= r_i`

This avoids false positives from orbiting near the hoop.

### Track generation
Training uses a curriculum:
1. Fixed, simple track (1–3 gates, large radius, gentle yaw)
2. More gates, mild turns and altitude changes
3. Randomized tracks with constraints:
   - Min spacing between gates
   - Max turn angle between consecutive gate normals/positions
   - Keep within bounds

Evaluation uses fixed seeded tracks for stable comparisons.

## Reward Design (shaped, with sparse event bonuses)
Without shaping, policies commonly learn “do nothing to avoid crashing.” Reward is computed from ground-truth sim state and gate geometry.

### Components
- Gate pass bonus:
  - Large positive reward when passing the *next* gate.
- Progress shaping (dense):
  - Reward for reducing distance to next gate center, or increasing projection toward it.
- Centering shaping:
  - Penalty proportional to radial error relative to gate radius.
- Speed shaping:
  - Small reward for velocity component toward next gate (capped).
- Smoothness penalty:
  - Penalty for action changes `||a_t - a_{t-1}||` and/or control energy.
- Crash/out-of-bounds penalty:
  - Large negative terminal reward.

### Termination / truncation
Terminate if:
- Collision (ground/obstacle/gate frame)
- Out of bounds
- Excessive tilt (optional early curriculum constraint)
- Completed track (all gates passed)

Truncate if:
- Timeout (max steps / max episode seconds)

## Policy / RL Training
### Algorithm
Default: PPO with recurrent policy (good stability for partial observability).
Alternative: SAC (sample efficiency), but recurrent SAC is more finicky.

### Model architecture
- Camera encoder: small CNN -> feature vector
- IMU encoder: MLP over fixed IMU window (or 1D conv) -> feature vector
- Fusion: concat + MLP
- Memory: GRU/LSTM (recommended)
- Heads: action mean (and log-std) + value

### Curriculum knobs
- Gate size (radius)
- Gate spacing
- Turn sharpness
- Max action limits (rates/thrust)
- Latency/noise (start small/none, increase gradually)

## Logging + Evaluation
Log per episode:
- Return
- Gates passed
- Crash rate
- Completion rate
- Time-to-finish (lap time) on fixed tracks
- Mean/peak speed
- Control smoothness metrics

Artifacts:
- Checkpoints (best + periodic)
- Rollout videos (FPV camera; optional third-person)
- Deterministic evaluation runs on fixed seeds

## Sim-to-real later (placeholder)
When ready:
- Domain randomization: camera exposure, noise, blur, FOV jitter; IMU bias/drift; latency jitter; mass/inertia/drag variations.
- Replace/augment inner-loop with a controller closer to real stack.
