from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Allow running from a source checkout without installing the package.
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Headless default.
if sys.platform != "win32" and "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from rl_drone_hoops.envs import MujocoDroneHoopsEnv  # noqa: E402
from rl_drone_hoops.rl.model import RecurrentActorCritic  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--image-rot90", type=int, default=None, help="Rotate FPV image CCW by 90deg k times (0..3).")
    ap.add_argument("--video-size", type=int, default=256, help="MP4 frame size (square).")
    ap.add_argument("--video-fps", type=float, default=30.0, help="MP4 FPS (frames are strided from control_hz).")
    ap.add_argument("--video-overlay", action="store_true", help="Overlay debug text on the MP4.")
    ap.add_argument("--no-video-overlay", action="store_true", help="Disable overlay text.")
    ap.add_argument(
        "--video-dir",
        type=str,
        default="",
        help="Directory for MP4 output. Defaults to $RL_DRONE_HOOPS_VIDEO_DIR or ./videos",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("cfg", {})

    image_size = int(cfg.get("image_size", 96))
    image_rot90 = int(args.image_rot90) if args.image_rot90 is not None else int(cfg.get("image_rot90", 1))
    env_kwargs = dict(
        image_size=image_size,
        image_rot90=image_rot90,
        camera_fps=float(cfg.get("camera_fps", 60.0)),
        imu_hz=float(cfg.get("imu_hz", 400.0)),
        control_hz=float(cfg.get("control_hz", 100.0)),
        physics_hz=float(cfg.get("physics_hz", 1000.0)),
        n_gates=int(cfg.get("n_gates", 5)),
        gate_radius=float(cfg.get("gate_radius", 0.75)),
        track_type=str(cfg.get("track_type", "straight")),
        turn_max_deg=float(cfg.get("turn_max_deg", 25.0)),
        episode_s=float(cfg.get("episode_s", 12.0)),
    )

    env = MujocoDroneHoopsEnv(seed=args.seed, **env_kwargs)
    obs, _ = env.reset(options={"fixed_track": True})
    imu_window_n = obs["imu"].shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentActorCritic(image_size=image_size, imu_window_n=imu_window_n).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    from rl_drone_hoops.utils.video_overlay import overlay_text_topleft

    frames = []
    returns = []
    ctrl_hz = float(env_kwargs.get("control_hz", 100.0))
    frame_stride = max(1, int(round(ctrl_hz / max(float(args.video_fps), 1e-6))))
    use_overlay = bool(args.video_overlay) and not bool(args.no_video_overlay)
    for ep in range(args.episodes):
        obs, _ = env.reset(options={"fixed_track": True})
        h = model.initial_hidden(1, device)
        done = False
        ep_ret = 0.0
        if args.video and ep == 0:
            fr = env.render_rgb(height=int(args.video_size), width=int(args.video_size))
            if use_overlay:
                p, rpy = env.pose_rpy()
                step = int(ckpt.get("global_step", -1))
                lines = [
                    f"ckpt={os.path.basename(args.checkpoint)} step={step}",
                    f"t={env._t:.2f}s gate={env._next_gate_idx}/{len(env.gates)} ret={ep_ret:.1f}",
                    f"pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})",
                    f"rpy=({np.rad2deg(rpy[0]):.1f},{np.rad2deg(rpy[1]):.1f},{np.rad2deg(rpy[2]):.1f}) deg",
                ]
                fr = overlay_text_topleft(fr, lines)
            frames.append(fr)
        frame_i = 0
        while not done:
            obs_t = {k: torch.as_tensor(v[None, ...], device=device) for k, v in obs.items()}
            out = model.act(obs_t, h, deterministic=True)
            h = out.h
            a = out.action.squeeze(0).cpu().numpy()
            obs, r, term, trunc, info = env.step(a)
            ep_ret += float(r)
            done = bool(term or trunc)
            if args.video and ep == 0 and (frame_i % frame_stride == 0):
                fr = env.render_rgb(height=int(args.video_size), width=int(args.video_size))
                if use_overlay:
                    p, rpy = env.pose_rpy()
                    step = int(ckpt.get("global_step", -1))
                    lines = [
                        f"ckpt={os.path.basename(args.checkpoint)} step={step}",
                        f"t={float(info.get('t', env._t)):.2f}s gate={int(info.get('next_gate_idx', env._next_gate_idx))}/{len(env.gates)} ret={ep_ret:.1f}",
                        f"pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})",
                        f"rpy=({np.rad2deg(rpy[0]):.1f},{np.rad2deg(rpy[1]):.1f},{np.rad2deg(rpy[2]):.1f}) deg",
                        f"a=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[3]:+.2f})",
                    ]
                    fr = overlay_text_topleft(fr, lines)
                frames.append(fr)
            frame_i += 1
        returns.append(ep_ret)
        print(f"ep {ep}: return={ep_ret:.3f} gates={info.get('next_gate_idx')} finished={bool(info.get('finished', False))}")

    if args.video and frames:
        import imageio.v2 as imageio

        video_root = args.video_dir or os.environ.get("RL_DRONE_HOOPS_VIDEO_DIR", os.path.join(os.getcwd(), "videos"))
        video_root = os.path.abspath(video_root)
        os.makedirs(video_root, exist_ok=True)

        out_name = os.path.splitext(os.path.basename(args.checkpoint))[0] + ".mp4"
        out = os.path.join(video_root, out_name)
        with imageio.get_writer(out, fps=float(args.video_fps), macro_block_size=1) as w:
            for fr in frames:
                w.append_data(fr)
        print("wrote:", out)

    print("return_mean:", float(np.mean(returns)))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
