# RL Racing Drone Through Hoops (MuJoCo)

Primary docs:
- Full system spec: `docs/SYSTEM.md`
- Agent context (for new AI agents): `docs/AGENT_CONTEXT.md`

Initial scope:
- Works in sim first (MuJoCo on Linux)
- Sensors: 60 FPS grayscale camera at 96x96 (configurable) + IMU
- Control: body rates + thrust

Quick start:
```bash
cd rl-drone-hoops
pip install -r requirements.txt
python3 scripts/smoke_env.py --steps 200
```

Train (recurrent PPO, minimal baseline):
```bash
cd rl-drone-hoops
pip install -r requirements-train.txt
python3 scripts/train_recurrent_ppo.py --total-steps 200000 --num-envs 4
```

Videos:
- Training eval videos are written to `./videos/` by default (relative to where you launch training).
- Override with `RL_DRONE_HOOPS_VIDEO_DIR=/abs/path/to/videos`.
- If the camera image appears rotated, set `RL_DRONE_HOOPS_IMAGE_ROT90=0..3` (CCW 90deg steps).
  - Quick calibration: `python3 scripts/generate_calibration_videos.py --all` writes `calib_rot{0,1,2,3}.mp4`.
- Eval video quality / overlay (env vars):
  - `RL_DRONE_HOOPS_VIDEO_SIZE=256` (square MP4 resolution)
  - `RL_DRONE_HOOPS_VIDEO_FPS=30`
  - `RL_DRONE_HOOPS_VIDEO_OVERLAY=1` (set `0` to disable)

Resume a disconnected run (loads latest checkpoint in `--run-dir`):
```bash
cd rl-drone-hoops
python3 scripts/train_recurrent_ppo.py --run-dir runs/<run_name> --resume
```

Keep Training Running (tmux):
```bash
cd rl-drone-hoops
SESSION=ppo_rnn_$(date +%Y%m%d_%H%M%S)
RUN_DIR=runs/$SESSION
mkdir -p "$RUN_DIR"
export RL_DRONE_HOOPS_VIDEO_DIR="$PWD/videos"
tmux new-session -d -s "$SESSION" bash -lc \
  "python3 -u scripts/train_recurrent_ppo.py --run-dir $RUN_DIR --total-steps 200000 --num-envs 4 2>&1 | tee -a $RUN_DIR/train.log"
```

Check/attach to a run in progress:
```bash
tmux ls
tmux attach -t <session_name>
# Detach: Ctrl-b then d
tail -n 50 runs/<run_name>/train.log
```

Fast sanity training (CPU-friendly):
```bash
python3 scripts/train_recurrent_ppo.py --total-steps 4096 --num-envs 1 --rollout-steps 64 --eval-every-steps 2048 --eval-episodes 1 --camera-fps 20 --physics-hz 500 --control-hz 50 --imu-hz 200
```

Evaluate a checkpoint:
```bash
python3 scripts/eval_checkpoint.py runs/.../checkpoints/step000200000.pt --episodes 3 --video
# Optional:
#   --video-dir ./videos
#   --image-rot90 0..3
# Or set:
#   RL_DRONE_HOOPS_VIDEO_DIR=./videos
```
