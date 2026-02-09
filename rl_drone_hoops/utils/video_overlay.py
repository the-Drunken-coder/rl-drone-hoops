from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def overlay_text_topleft(
    frame_rgb: np.ndarray,
    lines: Iterable[str],
    *,
    pad: int = 6,
    fg: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int, int] = (0, 0, 0, 160),
) -> np.ndarray:
    """
    Draw a small multi-line text overlay in the top-left of an RGB frame.

    Returns a new uint8 RGB array.
    """
    # Lazy import: PIL is available in this environment but we keep it optional.
    from PIL import Image, ImageDraw, ImageFont

    if frame_rgb.dtype != np.uint8:
        fr = np.clip(frame_rgb, 0, 255).astype(np.uint8)
    else:
        fr = frame_rgb

    img = Image.fromarray(fr, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    lines_l = [str(s) for s in lines]
    if not lines_l:
        return fr.copy()

    # Measure text block.
    widths = []
    heights = []
    for s in lines_l:
        bbox = draw.textbbox((0, 0), s, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    line_h = max(heights) if heights else 10
    box_w = max(widths) + 2 * pad
    box_h = line_h * len(lines_l) + 2 * pad

    # Background box.
    draw.rectangle([0, 0, box_w, box_h], fill=bg)

    # Text.
    y = pad
    for s in lines_l:
        draw.text((pad, y), s, font=font, fill=(*fg, 255))
        y += line_h

    out = img.convert("RGB")
    return np.asarray(out, dtype=np.uint8)

