from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Headless default.
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

# Allow running from a source checkout without installing the package.
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_drone_hoops.envs import MujocoDroneHoopsEnv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--image-size", type=int, default=96)
    args = ap.parse_args()

    env = MujocoDroneHoopsEnv(image_size=args.image_size)
    obs, info = env.reset()
    print("reset info:", info)
    print("obs keys:", list(obs.keys()))
    print("image:", obs["image"].shape, obs["image"].dtype)
    print("imu:", obs["imu"].shape, obs["imu"].dtype)

    ep_ret = 0.0
    for i in range(args.steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        ep_ret += r
        if info.get("gate_passed", False):
            print(f"step {i}: gate passed -> next_gate_idx={info['next_gate_idx']}")
        if term or trunc:
            print("done:", {"terminated": term, "truncated": trunc, **{k: info.get(k) for k in ['crash','finished','next_gate_idx']}})
            print("ep_return:", ep_ret)
            obs, info = env.reset()
            ep_ret = 0.0
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
