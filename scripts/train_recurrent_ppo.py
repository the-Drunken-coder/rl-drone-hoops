"""Training script for recurrent PPO."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any

import torch

# Allow running from a source checkout without installing the package.
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Headless default.
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from rl_drone_hoops.config import load_config, extract_ppo_config, extract_curriculum  # noqa: E402
from rl_drone_hoops.rl.ppo_recurrent import PPOConfig, train_ppo_recurrent  # noqa: E402
from rl_drone_hoops.utils.checkpoint import latest_checkpoint_path  # noqa: E402


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
    ap = argparse.ArgumentParser(
        description="Train recurrent PPO policy for drone racing through hoops.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python3 train_recurrent_ppo.py

  # Train with custom config file
  python3 train_recurrent_ppo.py --config config/fast-train.yaml

  # Override config values via CLI (takes precedence over config file)
  python3 train_recurrent_ppo.py --eval-every-steps 5000 --num-envs 8

  # Resume from latest checkpoint
  python3 train_recurrent_ppo.py --run-dir runs/my_run --resume

  # Resume and change eval frequency
  python3 train_recurrent_ppo.py --run-dir runs/my_run --resume --eval-every-steps 5000
        """,
    )

    # Config file
    ap.add_argument("--config", type=str, default="", help="Path to YAML/JSON config file (uses default if not specified).")

    # Run management
    ap.add_argument("--run-dir", type=str, default="", help="Directory for checkpoints and logs.")
    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in --run-dir.")
    ap.add_argument("--checkpoint", type=str, default="", help="Resume from explicit checkpoint path.")

    # PPO hyperparameters (override config file)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None, help="Discount factor")
    ap.add_argument("--gae-lambda", type=float, default=None, help="GAE lambda")
    ap.add_argument("--lr", type=float, default=None, help="Learning rate")
    ap.add_argument("--clip-coef", type=float, default=None, help="PPO clipping coefficient")
    ap.add_argument("--vf-coef", type=float, default=None, help="Value function loss weight")
    ap.add_argument("--ent-coef", type=float, default=None, help="Entropy regularization weight")
    ap.add_argument("--max-grad-norm", type=float, default=None, help="Gradient clipping norm")
    ap.add_argument("--update-epochs", type=int, default=None, help="PPO update epochs per rollout")

    # Environment parameters
    ap.add_argument("--image-size", type=int, default=None, help="FPV camera resolution (square)")
    ap.add_argument("--image-rot90", type=int, default=None, help="Rotate FPV image CCW by 90deg k times (0..3).")
    ap.add_argument("--camera-fps", type=float, default=None)
    ap.add_argument("--imu-hz", type=float, default=None)
    ap.add_argument("--control-hz", type=float, default=None)
    ap.add_argument("--physics-hz", type=float, default=None)

    # Track parameters
    ap.add_argument("--track-type", type=str, default=None, choices=["straight", "random_turns"])
    ap.add_argument("--gate-radius", type=float, default=None)
    ap.add_argument("--turn-max-deg", type=float, default=None)
    ap.add_argument("--n-gates", type=int, default=None)
    ap.add_argument("--episode-s", type=float, default=None)

    # Evaluation
    ap.add_argument("--eval-every-steps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)

    args = ap.parse_args()

    # Load base config from file
    try:
        base_config = load_config(args.config if args.config else None)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    # Extract PPO config from file
    ppo_config_dict = extract_ppo_config(base_config)

    # Determine run directory and check for checkpoint config
    resume_from = ""
    if args.checkpoint:
        resume_from = args.checkpoint
        if not args.run_dir:
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

    # Load config from checkpoint if resuming
    ckpt_config = {}
    if resume_from:
        ckpt = torch.load(resume_from, map_location="cpu")
        ckpt_config = dict(ckpt.get("cfg", {}) or {})
        if not args.run_dir and isinstance(ckpt_config.get("run_dir"), str) and ckpt_config["run_dir"]:
            run_dir = ckpt_config["run_dir"]

    # Priority: CLI args > checkpoint config > file config
    def pick(name: str, ppo_name: str | None = None) -> Any:
        """Pick value with priority: CLI > checkpoint > config file."""
        cli_name = ppo_name or name.replace("_", "-")
        cli_value = getattr(args, cli_name.replace("-", "_"), None)
        if cli_value is not None:
            return cli_value
        if name in ckpt_config:
            return ckpt_config[name]
        return ppo_config_dict[name]

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
        gamma=float(pick("gamma")),
        gae_lambda=float(pick("gae_lambda", "gae-lambda")),
        clip_coef=float(pick("clip_coef", "clip-coef")),
        vf_coef=float(pick("vf_coef", "vf-coef")),
        ent_coef=float(pick("ent_coef", "ent-coef")),
        max_grad_norm=float(pick("max_grad_norm", "max-grad-norm")),
        lr=float(pick("lr")),
        adam_eps=float(pick("adam_eps")),
        update_epochs=int(pick("update_epochs", "update-epochs")),
        minibatch_envs=int(pick("minibatch_envs")),
    )

    # Load curriculum from config
    curriculum = extract_curriculum(base_config)

    train_ppo_recurrent(cfg, curriculum=curriculum, resume_from=resume_from or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
