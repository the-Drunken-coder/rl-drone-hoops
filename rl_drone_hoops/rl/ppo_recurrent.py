"""Recurrent PPO implementation for training the drone policy."""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from rl_drone_hoops.constants import EPSILON, EPS_GRAD_UNDERFLOW
from rl_drone_hoops.envs import MujocoDroneHoopsEnv
from rl_drone_hoops.rl.distributions import SquashedDiagGaussian
from rl_drone_hoops.rl.model import RecurrentActorCritic
from rl_drone_hoops.rl.vec import InProcessVecEnv, SubprocVecEnv
from rl_drone_hoops.utils.best_model_tracker import BestModelTracker

logger = logging.getLogger(__name__)

def _auto_num_envs() -> int:
    # Use logical CPU count.
    # Note: This is a heuristic. MuJoCo env stepping is CPU-heavy, so matching envs to
    # logical cores is a reasonable default for throughput benchmarks.
    n = os.cpu_count() or 1
    # Avoid spawning a silly number of MuJoCo processes by default.
    return min(n, 32)


class AsyncCheckpointSaver:
    """(Optimization 4.1: Async checkpoint saving to eliminate I/O blocking)

    Saves checkpoints asynchronously in a background thread to avoid blocking training.
    Implements a simple queue-based design with deep copying to avoid race conditions.
    """

    def __init__(self, max_queue_size: int = 2):
        """Initialize async saver with background worker thread.

        Args:
            max_queue_size: Maximum items in queue before blocking
        """
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        """Background worker thread that processes save tasks."""
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.1)
                if item is None:
                    # Sentinel value to stop
                    break
                ckpt_dict, path = item
                try:
                    torch.save(ckpt_dict, path)
                except Exception:
                    logger.exception(f"Failed to save checkpoint to {path}")
                    # Continue processing other saves even if one fails
            except queue.Empty:
                continue

    def _deep_copy_opt_state(self, opt_state: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy optimizer state, moving tensors to CPU.
        
        Args:
            opt_state: Optimizer state dict from opt.state_dict()
            
        Returns:
            Deep copy with all tensors on CPU
        """
        result = {}
        for key, value in opt_state.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu().clone()
            elif isinstance(value, dict):
                result[key] = self._deep_copy_opt_state(value)
            elif isinstance(value, (list, tuple)):
                result[key] = type(value)(
                    item.detach().cpu().clone() if isinstance(item, torch.Tensor)
                    else self._deep_copy_opt_state(item) if isinstance(item, dict)
                    else item
                    for item in value
                )
            else:
                result[key] = value
        return result

    def save(self, ckpt_dict: Dict[str, Any], path: str) -> None:
        """Queue a checkpoint for async saving.

        Args:
            ckpt_dict: Checkpoint dictionary with model/optimizer state
            path: Path to save to
        """
        # Deep copy state dicts to avoid mutations during save
        ckpt_copy = {
            "global_step": ckpt_dict["global_step"],
            "global_flight": ckpt_dict["global_flight"],
            "model_state": {k: v.cpu().clone() for k, v in ckpt_dict["model_state"].items()},
            "opt_state": self._deep_copy_opt_state(ckpt_dict["opt_state"]),
            "cfg": ckpt_dict["cfg"],
        }
        self.queue.put((ckpt_copy, path))

    def shutdown(self) -> None:
        """Request the worker to stop and wait briefly for termination.

        Uses non-blocking queue operations to avoid hanging shutdown if the
        queue is full or the worker is stuck in I/O.
        """
        # Signal the worker to stop as soon as possible.
        self._stop_event.set()

        # Best-effort enqueue of sentinel without blocking.
        try:
            self.queue.put_nowait(None)  # Sentinel to stop worker
        except queue.Full:
            # Queue is full; drop one pending item to make room, then retry once.
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                logger.warning(
                    "AsyncCheckpointSaver queue full during shutdown; "
                    "skipping sentinel put."
                )

        # Wait briefly for the worker thread to exit.
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            logger.warning(
                "AsyncCheckpointSaver worker thread did not terminate within timeout."
            )


@dataclass
class PPOConfig:
    """Configuration for recurrent PPO training.

    Attributes:
        run_dir: Directory to save checkpoints and logs
        seed: Random seed for reproducibility
        device: "auto", "cpu", or "cuda"
        num_envs: Number of parallel environments
        total_steps: Total training steps (in environment steps, not updates)
        rollout_steps: Rollout length per PPO update
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        clip_coef: PPO clipping coefficient
        vf_coef: Value function loss weight
        ent_coef: Entropy regularization weight
        max_grad_norm: Gradient clipping norm
        lr: Optimizer learning rate
        update_epochs: Number of passes over rollout data
        minibatch_envs: Number of environments per minibatch (preserves RNN sequences)
        eval_every_steps: Evaluation frequency
        eval_episodes: Number of episodes per evaluation
        image_size: FPV camera resolution (square)
        image_rot90: Rotate FPV image CCW by 90deg this many times
        camera_fps: FPV camera sampling rate
        imu_hz: IMU sampling rate
        control_hz: Control loop rate
        physics_hz: Physics simulation rate
        track_type: "straight" or "random_turns"
        gate_radius: Radius of gates in meters
        turn_max_deg: Max turn angle in track generation
        n_gates: Number of gates per episode
        episode_s: Episode duration in seconds
    """

    run_dir: str
    seed: int = 0
    device: str = "auto"

    num_envs: int = 4
    total_steps: int = 200_000
    rollout_steps: int = 128

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    adam_eps: float = 1e-5

    update_epochs: int = 4
    minibatch_envs: int = 4  # recurrent PPO minibatches by env sequences

    # Vec env mode: inproc (single-process), subproc (multi-process), auto
    vec_mode: str = "auto"

    eval_every_steps: int = 50_000
    eval_episodes: int = 3

    # Env params (can be overridden per curriculum stage by caller)
    image_size: int = 96
    image_rot90: int = 0
    camera_fps: float = 60.0
    imu_hz: float = 400.0
    control_hz: float = 100.0
    physics_hz: float = 1000.0
    track_type: str = "straight"
    gate_radius: float = 1.25
    turn_max_deg: float = 20.0
    n_gates: int = 3
    episode_s: float = 12.0


def _move_opt_state_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    # Optimizer state can contain tensors that need to live on the same device as params.
    for st in opt.state.values():
        if not isinstance(st, dict):
            continue
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device=device)


def _device(cfg: PPOConfig) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_torch_obs(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in obs.items()}


def _compute_gae(
    rewards: torch.Tensor,  # (T,N)
    dones: torch.Tensor,  # (T,N) bool
    values: torch.Tensor,  # (T,N)
    last_value: torch.Tensor,  # (N,)
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T, N = rewards.shape
    adv = torch.zeros((T, N), device=rewards.device, dtype=torch.float32)
    last_gae = torch.zeros((N,), device=rewards.device, dtype=torch.float32)
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t].float()
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y < 1e-8:
        return float("nan")
    return 1.0 - float(np.var(y_true - y_pred) / var_y)


def evaluate(
    *,
    model: RecurrentActorCritic,
    device: torch.device,
    seed: int,
    episodes: int,
    run_dir: str,
    step: int,
    flight: int = 0,
    env_kwargs: dict,
    record_video: bool = True,
) -> Dict[str, float]:
    import imageio.v2 as imageio

    # Centralized video output (all runs).
    # Override with RL_DRONE_HOOPS_VIDEO_DIR=/some/path
    video_root = os.environ.get("RL_DRONE_HOOPS_VIDEO_DIR", os.path.join(os.getcwd(), "videos"))
    video_root = os.path.abspath(video_root)

    # Video options (eval-only).
    video_size = int(os.environ.get("RL_DRONE_HOOPS_EVAL_VIDEO_SIZE", os.environ.get("RL_DRONE_HOOPS_VIDEO_SIZE", "256")))
    video_fps = float(os.environ.get("RL_DRONE_HOOPS_EVAL_VIDEO_FPS", os.environ.get("RL_DRONE_HOOPS_VIDEO_FPS", "30")))
    video_overlay = os.environ.get("RL_DRONE_HOOPS_EVAL_VIDEO_OVERLAY", os.environ.get("RL_DRONE_HOOPS_VIDEO_OVERLAY", "1")) not in ("0", "false", "False")
    # Render at ~video_fps by skipping frames relative to control_hz.
    ctrl_hz = float(env_kwargs.get("control_hz", 100.0))
    frame_stride = max(1, int(round(ctrl_hz / max(video_fps, 1e-6))))

    from rl_drone_hoops.utils.video_overlay import overlay_text_topleft

    model.eval()
    rets: List[float] = []
    gates: List[int] = []
    finished: List[int] = []

    # Deterministic eval: create fresh env with fixed seed in constructor, fixed track thereafter.
    env = MujocoDroneHoopsEnv(seed=seed, **env_kwargs)
    for ep in range(episodes):
        obs, _ = env.reset(options={"fixed_track": True})
        h = model.initial_hidden(1, device)
        done = False
        ep_ret = 0.0
        frames = []
        frame_i = 0
        if record_video and ep == 0:
            fr = env.render_rgb(height=video_size, width=video_size)
            if video_overlay:
                p, rpy = env.pose_rpy()
                lines = [
                    f"run={os.path.basename(os.path.normpath(run_dir))} step={step}",
                    f"t={env._t:.2f}s gate={env._next_gate_idx}/{len(env.gates)} ret={ep_ret:.1f}",
                    f"pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})",
                    f"rpy=({np.rad2deg(rpy[0]):.1f},{np.rad2deg(rpy[1]):.1f},{np.rad2deg(rpy[2]):.1f}) deg",
                ]
                fr = overlay_text_topleft(fr, lines)
            frames.append(fr)
        while not done:
            obs_t = _to_torch_obs({k: v[None, ...] for k, v in obs.items()}, device)
            out = model.act(obs_t, h, deterministic=True)
            h = out.h
            a = out.action.squeeze(0).cpu().numpy()
            obs, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            ep_ret += float(r)
            if record_video and ep == 0 and (frame_i % frame_stride == 0):
                fr = env.render_rgb(height=video_size, width=video_size)
                if video_overlay:
                    p, rpy = env.pose_rpy()
                    lines = [
                        f"run={os.path.basename(os.path.normpath(run_dir))} step={step}",
                        f"t={float(info.get('t', env._t)):.2f}s gate={int(info.get('next_gate_idx', env._next_gate_idx))}/{len(env.gates)} ret={ep_ret:.1f}",
                        f"pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})",
                        f"rpy=({np.rad2deg(rpy[0]):.1f},{np.rad2deg(rpy[1]):.1f},{np.rad2deg(rpy[2]):.1f}) deg",
                        f"a=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[3]:+.2f})",
                    ]
                    fr = overlay_text_topleft(fr, lines)
                frames.append(fr)
            frame_i += 1
        rets.append(ep_ret)
        gates.append(int(info.get("next_gate_idx", 0)))
        finished.append(1 if info.get("finished", False) else 0)

        if record_video and ep == 0 and frames:
            os.makedirs(video_root, exist_ok=True)
            run_name = os.path.basename(os.path.normpath(run_dir))
            path = os.path.join(video_root, f"eval_{run_name}_flight{flight:09d}_step{step:09d}.mp4")
            with imageio.get_writer(path, fps=video_fps, macro_block_size=1) as w:
                for fr in frames:
                    w.append_data(fr)

    env.close()
    model.train()
    return {
        "eval/return_mean": float(np.mean(rets)),
        "eval/return_std": float(np.std(rets)),
        "eval/gates_mean": float(np.mean(gates)),
        "eval/finished_rate": float(np.mean(finished)),
    }


def train_ppo_recurrent(
    cfg: PPOConfig,
    *,
    curriculum: Optional[List[Tuple[int, dict]]] = None,
    resume_from: Optional[str] = None,
) -> None:
    """
    Train recurrent PPO.

    curriculum: optional list of (step_threshold, env_kwargs_update).
      The last threshold <= current_step applies.
    """
    os.makedirs(cfg.run_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.run_dir, "checkpoints"), exist_ok=True)

    _set_seed(cfg.seed)
    device = _device(cfg)

    # Build vector envs.
    if cfg.num_envs <= 0:
        cfg.num_envs = _auto_num_envs()

    env_kwargs_list = []
    for rank in range(cfg.num_envs):
        env_kwargs_list.append(
            dict(
                seed=cfg.seed + 1000 * rank,
                image_size=cfg.image_size,
                image_rot90=cfg.image_rot90,
                camera_fps=cfg.camera_fps,
                imu_hz=cfg.imu_hz,
                control_hz=cfg.control_hz,
                physics_hz=cfg.physics_hz,
                n_gates=cfg.n_gates,
                gate_radius=cfg.gate_radius,
                track_type=cfg.track_type,
                turn_max_deg=cfg.turn_max_deg,
                episode_s=cfg.episode_s,
            )
        )

    vec_mode = (cfg.vec_mode or "auto").lower()
    if vec_mode == "auto":
        # Subprocess stepping is mainly beneficial for CPU training with multiple envs.
        vec_mode = "subproc" if (device.type == "cpu" and cfg.num_envs > 1) else "inproc"

    if vec_mode == "subproc":
        venv = SubprocVecEnv(MujocoDroneHoopsEnv, env_kwargs_list)
    elif vec_mode == "inproc":
        venv = InProcessVecEnv([lambda kw=kw: MujocoDroneHoopsEnv(**kw) for kw in env_kwargs_list])
    else:
        raise ValueError(f"Unknown vec_mode={cfg.vec_mode!r}; expected auto|inproc|subproc")
    obs = venv.reset(seeds=[cfg.seed + i for i in range(cfg.num_envs)])

    imu_window_n = obs["imu"].shape[1]
    model = RecurrentActorCritic(image_size=cfg.image_size, imu_window_n=imu_window_n).to(device)
    opt = Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    writer = SummaryWriter(log_dir=os.path.join(cfg.run_dir, "tb"))

    # Best model tracking (auto-saves to models/ folder)
    best_model_tracker = BestModelTracker(models_dir="models")
    writer.add_text("config", str(cfg))

    # Rollout storage.
    T = cfg.rollout_steps
    N = cfg.num_envs

    # (Optimization 3.5: Lazy observation buffer allocation after first reset)
    # Allocate now that we know obs shapes
    obs_buf: Dict[str, torch.Tensor] = {
        "image": torch.zeros((T, N, *obs["image"].shape[1:]), device=device, dtype=torch.uint8),
        "imu": torch.zeros((T, N, *obs["imu"].shape[1:]), device=device, dtype=torch.float32),
        "last_action": torch.zeros((T, N, *obs["last_action"].shape[1:]), device=device, dtype=torch.float32),
    }
    act_buf = torch.zeros((T, N, 4), device=device, dtype=torch.float32)
    logp_buf = torch.zeros((T, N), device=device, dtype=torch.float32)
    val_buf = torch.zeros((T, N), device=device, dtype=torch.float32)
    rew_buf = torch.zeros((T, N), device=device, dtype=torch.float32)
    done_buf = torch.zeros((T, N), device=device, dtype=torch.bool)
    ent_buf = torch.zeros((T, N), device=device, dtype=torch.float32)

    # Episode tracking.
    ep_ret = np.zeros((N,), dtype=np.float64)
    ep_gates = np.zeros((N,), dtype=np.int32)
    ep_len = np.zeros((N,), dtype=np.int32)

    h = model.initial_hidden(N, device)

    # (Optimization 4.1: Async checkpoint saving)
    ckpt_saver = AsyncCheckpointSaver()

    start_time = time.time()
    global_step = 0
    global_flight = 0  # Track number of completed episodes/flights
    next_eval = cfg.eval_every_steps

    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        try:
            model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["opt_state"])
            _move_opt_state_to_device(opt, device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint '{resume_from}': {e}") from e
        global_step = int(ckpt.get("global_step", global_step))
        global_flight = int(ckpt.get("global_flight", global_flight))
        next_eval = ((global_step // cfg.eval_every_steps) + 1) * cfg.eval_every_steps
        writer.add_text("resume/from", str(resume_from))
        print(f"resumed from {resume_from} at step={global_step} flight={global_flight}", flush=True)

    def current_env_kwargs() -> dict:
        base = dict(
            image_size=cfg.image_size,
            image_rot90=cfg.image_rot90,
            camera_fps=cfg.camera_fps,
            imu_hz=cfg.imu_hz,
            control_hz=cfg.control_hz,
            physics_hz=cfg.physics_hz,
            n_gates=cfg.n_gates,
            gate_radius=cfg.gate_radius,
            track_type=cfg.track_type,
            turn_max_deg=cfg.turn_max_deg,
            episode_s=cfg.episode_s,
        )
        if not curriculum:
            return base
        applied = {}
        for thresh, upd in curriculum:
            if global_step >= thresh:
                applied = upd
        base.update(applied)
        return base

    try:
        while global_step < cfg.total_steps:
            # Allow curriculum to affect only eval env kwargs (training envs remain as created).
            # For full dynamic curriculum, we'd rebuild envs; keep it simple for now.

            # Store initial hidden per rollout for recomputation.
            h0 = h.detach()
            for t in range(T):
                # Save obs. (Optimization 3.5: obs_buf already allocated after first reset)
                obs_t = _to_torch_obs(obs, device)
                obs_buf["image"][t] = obs_t["image"].to(torch.uint8)
                obs_buf["imu"][t] = obs_t["imu"].to(torch.float32)
                obs_buf["last_action"][t] = obs_t["last_action"].to(torch.float32)

                out = model.act(obs_t, h, deterministic=False)
                h = out.h

                act = out.action
                act_buf[t] = act
                logp_buf[t] = out.logp
                val_buf[t] = out.value
                ent_buf[t] = out.entropy

                # Step env.
                res = venv.step(act.detach().cpu().numpy())
                obs = res.obs
                rew = torch.as_tensor(res.reward, device=device, dtype=torch.float32)
                done = torch.as_tensor(res.done, device=device, dtype=torch.bool)

                rew_buf[t] = rew
                done_buf[t] = done

                # Episode stats.
                ep_ret += res.reward
                ep_len += 1
                for i, inf in enumerate(res.info):
                    ep_gates[i] = int(inf.get("next_gate_idx", ep_gates[i]))

                if done.any().item():
                    d_idx = np.where(res.done)[0]
                    for i in d_idx:
                        writer.add_scalar("train/ep_return", float(ep_ret[i]), global_step)
                        writer.add_scalar("train/ep_len", int(ep_len[i]), global_step)
                        writer.add_scalar("train/ep_gates", int(ep_gates[i]), global_step)
                        writer.add_scalar("train/flight", global_flight, global_step)
                        ep_ret[i] = 0.0
                        ep_len[i] = 0
                        ep_gates[i] = 0
                        global_flight += 1  # Increment flight counter
                    # Reset hidden for done envs (partial observability).
                    h[:, done, :] = 0.0

                global_step += N

            # Bootstrap value for GAE.
            with torch.no_grad():
                obs_last = _to_torch_obs(obs, device)
                out_last = model.act(obs_last, h, deterministic=True)
                last_val = out_last.value  # (N,)

            adv, ret = _compute_gae(rew_buf, done_buf, val_buf, last_val, cfg.gamma, cfg.gae_lambda)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # PPO update: minibatch by env sequences to preserve recurrence.
            env_ids = np.arange(N)
            np.random.shuffle(env_ids)
            mb = max(1, cfg.minibatch_envs)

            clipfracs = []
            approx_kls = []

            for epoch in range(cfg.update_epochs):
                for start in range(0, N, mb):
                    mb_ids = env_ids[start : start + mb]
                    # Slice sequences (T,mb,...)
                    obs_mb = {k: v[:, mb_ids] for k, v in obs_buf.items()}
                    act_mb = act_buf[:, mb_ids]
                    logp_old = logp_buf[:, mb_ids]
                    adv_mb = adv[:, mb_ids]
                    ret_mb = ret[:, mb_ids]
                    val_old = val_buf[:, mb_ids]

                    mean, value, _hT = model.forward_sequence_masked(
                        obs_mb, done_buf[:, mb_ids], h0[:, mb_ids].contiguous()
                    )
                    log_std = model.log_std.expand_as(mean).clamp(-5.0, 2.0)
                    dist = SquashedDiagGaussian(mean=mean, log_std=log_std)

                    # Need u for log_prob; invert tanh approximately with atanh on clipped action.
                    a = act_mb
                    a_clip = torch.clamp(a, -1.0 + EPSILON, 1.0 - EPSILON)
                    u = 0.5 * (torch.log1p(a_clip) - torch.log1p(-a_clip))  # atanh
                    logp = dist.log_prob(u, a)
                    ent = dist.entropy_approx()

                    # Prevent numerical underflow in ratio computation
                    log_ratio = logp - logp_old
                    log_ratio = torch.clamp(log_ratio, -EPS_GRAD_UNDERFLOW, EPS_GRAD_UNDERFLOW)
                    ratio = torch.exp(log_ratio)
                    pg1 = adv_mb * ratio
                    pg2 = adv_mb * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                    policy_loss = -torch.min(pg1, pg2).mean()

                    # Value loss with clipping.
                    v = value
                    v_clipped = val_old + torch.clamp(v - val_old, -cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * torch.max((v - ret_mb) ** 2, (v_clipped - ret_mb) ** 2).mean()

                    ent_loss = -ent.mean()
                    loss = policy_loss + cfg.vf_coef * v_loss + cfg.ent_coef * ent_loss

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    opt.step()

                    # Diagnostics.
                    with torch.no_grad():
                        approx_kl = (logp_old - logp).mean().cpu().item()
                        approx_kls.append(approx_kl)
                        clipfrac = (torch.abs(ratio - 1.0) > cfg.clip_coef).float().mean().cpu().item()
                        clipfracs.append(clipfrac)

            # Logging.
            sps = global_step / max(time.time() - start_time, 1e-9)
            writer.add_scalar("train/sps", float(sps), global_step)
            writer.add_scalar("loss/approx_kl", float(np.mean(approx_kls)) if approx_kls else 0.0, global_step)
            writer.add_scalar("loss/clipfrac", float(np.mean(clipfracs)) if clipfracs else 0.0, global_step)
            writer.add_scalar("loss/entropy_mean", float(ent_buf.mean().cpu().item()), global_step)
            writer.add_scalar("loss/value_mean", float(val_buf.mean().cpu().item()), global_step)

            ev = _explained_variance(val_buf.detach().cpu().numpy().flatten(), ret.detach().cpu().numpy().flatten())
            writer.add_scalar("loss/explained_variance", float(ev), global_step)
            # (Optimization 3.4: Reduce print overhead - only print periodically)
            if global_step % (cfg.rollout_steps * 10) == 0:
                print(
                    f"step={global_step} sps={sps:.1f} kl={float(np.mean(approx_kls)) if approx_kls else 0.0:.4f} "
                    f"clipfrac={float(np.mean(clipfracs)) if clipfracs else 0.0:.3f} ev={ev:.3f}"
                , flush=True)

            # Checkpoint. (Optimization 4.1: Async saving to avoid I/O blocking)
            ckpt_path = os.path.join(cfg.run_dir, "checkpoints", f"flight{global_flight:09d}_step{global_step:09d}.pt")
            ckpt_saver.save(
                {
                    "global_step": global_step,
                    "global_flight": global_flight,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "cfg": cfg.__dict__,
                },
                ckpt_path,
            )

            # Eval.
            if global_step >= next_eval:
                eval_metrics = evaluate(
                    model=model,
                    device=device,
                    seed=cfg.seed + 9999,
                    episodes=cfg.eval_episodes,
                    run_dir=cfg.run_dir,
                    step=global_step,
                    flight=global_flight,
                    env_kwargs=current_env_kwargs(),
                    record_video=True,
                )
                for k, v in eval_metrics.items():
                    writer.add_scalar(k, v, global_step)
                print("eval:", {k: round(v, 3) for k, v in eval_metrics.items()}, flush=True)

                # Check if this is the best model and save if so
                # Find latest checkpoint by step number (flight number changes, step is stable)
                ckpt_dir = os.path.join(cfg.run_dir, "checkpoints")
                ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("flight") and f.endswith(".pt")]
                ckpt_files.sort(key=lambda x: int(x.split("_step")[1].split(".")[0]), reverse=True)
                latest_ckpt = os.path.join(ckpt_dir, ckpt_files[0]) if ckpt_files else None

                if latest_ckpt and os.path.exists(latest_ckpt):
                    # Pass current curriculum for difficulty tracking
                    curriculum_info = {
                        "n_gates": current_env_kwargs().get("n_gates", 3),
                        "gate_radius": current_env_kwargs().get("gate_radius", 1.25),
                        "track_type": current_env_kwargs().get("track_type", "straight"),
                        "turn_max_deg": current_env_kwargs().get("turn_max_deg", 20.0),
                    }
                    best_model_tracker.check_and_save(
                        eval_metrics,
                        latest_ckpt,
                        os.path.basename(os.path.normpath(cfg.run_dir)),
                        global_flight,  # Use flight number instead of step
                        curriculum=curriculum_info,
                    )

                next_eval += cfg.eval_every_steps

    finally:
        # (Optimization 4.1: Wait for pending async checkpoint saves)
        try:
            ckpt_saver.shutdown()
        except Exception as e:
            logger.warning("Error during checkpoint saver shutdown: %s", e)
        
        # Close vector env and TensorBoard writer
        try:
            venv.close()
        except Exception as e:
            logger.warning("Error closing vector env: %s", e)
        
        try:
            writer.close()
        except Exception as e:
            logger.warning("Error closing TensorBoard writer: %s", e)
