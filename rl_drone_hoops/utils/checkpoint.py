"""Checkpoint loading and saving utilities."""
from __future__ import annotations

import os
from typing import Optional


def latest_checkpoint_path(run_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a run directory.

    Checkpoints are expected to be named: step{step_number:09d}.pt

    Args:
        run_dir: Path to the run directory containing a 'checkpoints' subdirectory

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None

    best_step = -1
    best_path: Optional[str] = None

    for name in os.listdir(ckpt_dir):
        if not name.endswith(".pt"):
            continue

        step = None
        if name.startswith("step"):
            # step000123456.pt
            num = name[len("step") : -len(".pt")]
            if num.isdigit():
                step = int(num)
        elif "_step" in name:
            # flight000001835_step000600064.pt
            base = name[: -len(".pt")]
            maybe = base.split("_step")[-1]
            if maybe.isdigit():
                step = int(maybe)

        if step is None:
            continue
        if step > best_step:
            best_step = step
            best_path = os.path.join(ckpt_dir, name)

    return best_path
