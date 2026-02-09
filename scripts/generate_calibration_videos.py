#!/usr/bin/env python3
"""
Generate short MP4 calibration clips for FPV camera orientation.

Usage:
  python scripts/generate_calibration_videos.py --all
  python scripts/generate_calibration_videos.py --image-rot90 0

Videos go to:
  --out-dir, else $RL_DRONE_HOOPS_VIDEO_DIR, else ./videos
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _default_video_dir() -> str:
    d = os.environ.get("RL_DRONE_HOOPS_VIDEO_DIR", os.path.join(os.getcwd(), "videos"))
    return os.path.abspath(d)


def main() -> None:
    # Allow running from a source checkout without installing the package.
    _ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    # Headless default.
    if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="Output directory for MP4s.")
    ap.add_argument("--seconds", type=float, default=3.0, help="Clip length in seconds.")
    ap.add_argument("--fps", type=int, default=20, help="Video FPS.")
    ap.add_argument("--seed", type=int, default=0, help="Env seed.")
    ap.add_argument("--image-rot90", type=int, default=0, help="Rotate CCW by 90deg k times (0..3).")
    ap.add_argument("--all", action="store_true", help="Generate for k in {0,1,2,3}.")
    args = ap.parse_args()

    # Ensure per-process env override doesn't silently force a different k.
    os.environ.pop("RL_DRONE_HOOPS_IMAGE_ROT90", None)

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else _default_video_dir()
    os.makedirs(out_dir, exist_ok=True)

    # Import lazily so this script can be inspected without MuJoCo installed.
    import imageio.v2 as imageio

    from rl_drone_hoops.envs import MujocoDroneHoopsEnv

    ks = [0, 1, 2, 3] if args.all else [int(args.image_rot90) % 4]

    for k in ks:
        env = MujocoDroneHoopsEnv(
            seed=int(args.seed),
            image_size=256,  # higher-res makes orientation obvious
            image_rot90=int(k),
            camera_fps=float(args.fps),
            physics_hz=500.0,
            control_hz=50.0,
            imu_hz=200.0,
            n_gates=1,
            gate_radius=2.0,
            episode_s=float(args.seconds),
        )
        env.reset(options={"fixed_track": True})

        # Hover-ish command: zero body rates, near-hover thrust.
        hover_thrust_norm = float(getattr(env, "_hover_thrust_norm", 0.0))
        a = np.array([0.0, 0.0, 0.0, hover_thrust_norm], dtype=np.float32)

        stride = max(1, int(round(50.0 / float(args.fps))))  # record roughly at fps
        frames = []
        steps = int(round(float(args.seconds) * 50.0))
        for i in range(steps):
            env.step(a)
            if i % stride == 0:
                frames.append(env.render())

        env.close()

        path = os.path.join(out_dir, f"calib_rot{k}.mp4")
        with imageio.get_writer(path, fps=int(args.fps), macro_block_size=1) as w:
            for fr in frames:
                w.append_data(fr)

        print(f"wrote {path}")


if __name__ == "__main__":
    main()
