"""Training script for recurrent PPO."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import torch

# Allow running from a source checkout without installing the package.
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Headless default.
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from rl_drone_hoops.rl.ppo_recurrent import PPOConfig, train_ppo_recurrent  # noqa: E402
from rl_drone_hoops.utils.checkpoint import latest_checkpoint_path  # noqa: E402


_DEFAULTS = dict(
    seed=0,
    device="auto",
    num_envs=4,
    total_steps=200_000,
    rollout_steps=128,
    image_size=96,
    image_rot90=0,
    camera_fps=60.0,
    imu_hz=400.0,
    control_hz=100.0,
    physics_hz=1000.0,
    track_type="straight",
    gate_radius=1.25,
    turn_max_deg=20.0,
    n_gates=3,
    episode_s=12.0,
    eval_every_steps=50_000,
    eval_episodes=3,
)


def _latest_run_dir(base_dir: str = "runs") -> str:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"'{base_dir}' not found; pass --run-dir explicitly.")
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        raise FileNotFoundError(f"No run directories found under '{base_dir}'.")
    # Name includes sortable timestamp: ppo_rnn_YYYYMMDD_HHMMSS
    dirs.sort(reverse=True)
    return os.path.join(base_dir, dirs[0])


def _latest_checkpoint(run_dir: str) -> str:
    """Find latest checkpoint in run directory using shared utility."""
    ckpt_path = latest_checkpoint_path(run_dir)
    if ckpt_path is None:
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        raise FileNotFoundError(f"No checkpoints found under '{ckpt_dir}'.")
    return ckpt_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in the run dir.")
    ap.add_argument("--checkpoint", type=str, default="", help="Resume from an explicit checkpoint path.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--rollout-steps", type=int, default=None)

    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--image-rot90", type=int, default=None, help="Rotate FPV image CCW by 90deg k times (0..3).")
    ap.add_argument("--camera-fps", type=float, default=None)
    ap.add_argument("--imu-hz", type=float, default=None)
    ap.add_argument("--control-hz", type=float, default=None)
    ap.add_argument("--physics-hz", type=float, default=None)
    ap.add_argument("--track-type", type=str, default=None, choices=["straight", "random_turns"])
    ap.add_argument("--gate-radius", type=float, default=None)
    ap.add_argument("--turn-max-deg", type=float, default=None)
    ap.add_argument("--n-gates", type=int, default=None)
    ap.add_argument("--episode-s", type=float, default=None)

    ap.add_argument("--eval-every-steps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    args = ap.parse_args()

    resume_from = ""
    if args.checkpoint:
        resume_from = args.checkpoint
        if not args.run_dir:
            # .../runs/<run>/checkpoints/step....pt -> .../runs/<run>
            run_dir = os.path.dirname(os.path.dirname(os.path.normpath(resume_from)))
        else:
            run_dir = args.run_dir
    elif args.resume:
        run_dir = args.run_dir if args.run_dir else _latest_run_dir("runs")
        resume_from = _latest_checkpoint(run_dir)
    else:
        if args.run_dir:
            run_dir = args.run_dir
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join("runs", f"ppo_rnn_{ts}")

    ckpt_cfg = {}
    if resume_from:
        ckpt = torch.load(resume_from, map_location="cpu")
        ckpt_cfg = dict(ckpt.get("cfg", {}) or {})
        if not args.run_dir and isinstance(ckpt_cfg.get("run_dir"), str) and ckpt_cfg["run_dir"]:
            run_dir = ckpt_cfg["run_dir"]

    def pick(name: str):
        v = getattr(args, name)
        if v is not None:
            return v
        if name in ckpt_cfg:
            return ckpt_cfg[name]
        return _DEFAULTS[name]

    cfg = PPOConfig(
        run_dir=run_dir,
        seed=int(pick("seed")),
        device=str(pick("device")),
        num_envs=int(pick("num_envs")),
        total_steps=int(pick("total_steps")),
        rollout_steps=int(pick("rollout_steps")),
        image_size=int(pick("image_size")),
        image_rot90=int(pick("image_rot90")),
        camera_fps=float(pick("camera_fps")),
        imu_hz=float(pick("imu_hz")),
        control_hz=float(pick("control_hz")),
        physics_hz=float(pick("physics_hz")),
        track_type=str(pick("track_type")),
        gate_radius=float(pick("gate_radius")),
        turn_max_deg=float(pick("turn_max_deg")),
        n_gates=int(pick("n_gates")),
        episode_s=float(pick("episode_s")),
        eval_every_steps=int(pick("eval_every_steps")),
        eval_episodes=int(pick("eval_episodes")),
    )

    # Simple curriculum (optional): thresholds in env-steps.
    curriculum = [
        (0, {"gate_radius": max(cfg.gate_radius, 1.25), "n_gates": min(cfg.n_gates, 3), "track_type": "straight"}),
        (200_000, {"gate_radius": 1.0, "n_gates": 5, "track_type": cfg.track_type}),
        (600_000, {"gate_radius": 0.8, "n_gates": 7, "track_type": "random_turns", "turn_max_deg": max(25.0, cfg.turn_max_deg)}),
    ]

    train_ppo_recurrent(cfg, curriculum=curriculum, resume_from=resume_from or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
