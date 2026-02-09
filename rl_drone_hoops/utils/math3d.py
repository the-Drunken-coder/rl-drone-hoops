from __future__ import annotations

import numpy as np


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo quaternion (w, x, y, z) to 3x3 rotation matrix.
    """
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )


def mat_to_rpy(R: np.ndarray) -> np.ndarray:
    """
    Roll-pitch-yaw from rotation matrix, aerospace convention.
    """
    # Guard numeric issues.
    sy = np.clip(-R[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sy)
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Quaternion (w,x,y,z) that rotates unit vector a to unit vector b.
    """
    a_u = unit(a, eps=eps).astype(np.float64)
    b_u = unit(b, eps=eps).astype(np.float64)
    c = float(np.dot(a_u, b_u))
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if c < -1.0 + 1e-12:
        # 180 deg: pick an arbitrary orthogonal axis.
        axis = unit(np.cross(a_u, np.array([1.0, 0.0, 0.0], dtype=np.float64)), eps=eps)
        if np.linalg.norm(axis) < eps:
            axis = unit(np.cross(a_u, np.array([0.0, 1.0, 0.0], dtype=np.float64)), eps=eps)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)
    v = np.cross(a_u, b_u)
    s = np.sqrt((1.0 + c) * 2.0)
    invs = 1.0 / max(s, eps)
    return np.array([0.5 * s, v[0] * invs, v[1] * invs, v[2] * invs], dtype=np.float64)
