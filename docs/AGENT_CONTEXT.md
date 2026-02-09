# Agent Context: RL Racing Drone Through Hoops (MuJoCo)

This file is for uninformed AI agents joining the project. It answers “what are we building, what decisions are already made, what should you not change, and what are the open questions.”

## What we are building
A Gymnasium-style RL environment + training stack to learn a racing quadrotor policy that flies through ordered hoops using only:
- 60 FPS grayscale camera (default 96x96; configurable)
- IMU (gyro + accel) at high rate

The policy outputs body rates + thrust. Reward and termination use sim ground truth, but observations do not.

## Decisions already made (do not renegotiate without asking)
- Engine: MuJoCo on Linux.
- Camera: 60 FPS, grayscale, 96x96 by default (configurable).
- Priority: works in sim first; sim-to-real later.
- Action space: `[roll_rate, pitch_rate, yaw_rate, thrust]` (continuous).
- Partial observability is real: sensor latency + rate mismatch are modeled.
- Recurrent policy is expected (GRU/LSTM).

## Key technical constraints
- Multi-rate timing:
  - Physics: 500–1000 Hz
  - Control: ~100 Hz
  - IMU: ~400 Hz (200–500 Hz OK)
  - Camera: 60 Hz
- Latencies are explicit and implemented via sensor delivery buffers.
- Gate passing uses plane-crossing + within-radius at crossing point.
- Camera grayscale uses a high-contrast mapping so the red gate is visible at low resolution.

## Files to read first
- Full spec: `docs/SYSTEM.md`

## Running headless (common on servers)
- If there is no X11 display, set `MUJOCO_GL=egl` (defaulted automatically when `DISPLAY` is missing).
- If EGL is unavailable on a machine, try `MUJOCO_GL=osmesa` (slower, CPU-based).

## Quick smoke test
- `python3 scripts/smoke_env.py --steps 200`

## Training (baseline)
- `pip install -r requirements-train.txt`
- `python3 scripts/train_recurrent_ppo.py --total-steps 200000 --num-envs 4`
- TensorBoard: logs are written under `runs/<run>/tb/`
- Training eval videos: written under `./videos/` by default (override with `RL_DRONE_HOOPS_VIDEO_DIR=...`).
  - If images appear rotated, set `RL_DRONE_HOOPS_IMAGE_ROT90=0..3` (CCW 90deg steps).
  - Quick orientation check: `python3 scripts/generate_calibration_videos.py --all` (writes `calib_rot*.mp4`).
  - Video quality/overlay env vars: `RL_DRONE_HOOPS_VIDEO_SIZE`, `RL_DRONE_HOOPS_VIDEO_FPS`, `RL_DRONE_HOOPS_VIDEO_OVERLAY`.

## Operational note: tmux
Long training jobs should be run inside `tmux` so they survive SSH disconnects.

When joining an existing machine/session:
- Check whether training is already running: `tmux ls`
- Attach to the training session: `tmux attach -t <session>`
- Detach without stopping training: `Ctrl-b` then `d`
- Inspect logs: `tail -n 50 runs/<run>/train.log`
- Inspect checkpoints: `ls -la runs/<run>/checkpoints/`

## Evaluate a checkpoint
- `python3 scripts/eval_checkpoint.py runs/.../checkpoints/stepXXXXXXXXX.pt --episodes 3 --video`
  - Videos are written under `./videos/` by default (override with `--video-dir ...` or `RL_DRONE_HOOPS_VIDEO_DIR=...`).
  - You can force rotation with `--image-rot90 0..3`.

## What “done” looks like for MVP
- MuJoCo environment runs headless.
- `reset()/step()` works.
- Gates exist; passing gates increments progress.
- Camera observations are produced at 60 FPS with latency.
- IMU observations are produced at high rate with latency and aggregated into a fixed window per control step.
- Reward shaping yields learning signal (agent doesn’t just hover).
- Basic PPO recurrent training can improve completion rate on simple tracks.
- Logging includes: return, gates passed, crash rate, videos, checkpoints.

## Non-negotiables for training integrity
- No privileged state in observations for the main run.
- Deterministic eval tracks (fixed seeds) separate from training randomization.
- Reward should be stable under timestep/rate changes (watch for dt bugs).

## Open questions (ask user only if blocked)
- Action/control rate target (default 100 Hz unless specified otherwise).
- Initial latency values to use for camera/action/IMU (defaults can be set, but should be configurable).
- Whether gates have physical frames (collidable) or are just “pass-through rings” initially.
- Whether to include a simple inner-loop rate PID in sim (recommended for early stability).

## Recommended development order
1. Implement minimal MuJoCo world + drone rigid body + gates + collision/bounds.
2. Implement gate passing detection + reward + termination.
3. Implement timing buffers + IMU stream + action delay.
4. Implement camera rendering at 60 FPS grayscale with latency.
5. Add recurrent PPO training harness and evaluation + video logging.
