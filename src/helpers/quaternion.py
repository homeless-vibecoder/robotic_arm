from __future__ import annotations

import numpy as np


def safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else 0.0


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    vx, vy, vz = v
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    rx = (ww + xx - yy - zz) * vx + 2.0 * (xy - wz) * vy + 2.0 * (xz + wy) * vz
    ry = 2.0 * (xy + wz) * vx + (ww - xx + yy - zz) * vy + 2.0 * (yz - wx) * vz
    rz = 2.0 * (xz - wy) * vx + 2.0 * (yz + wx) * vy + (ww - xx - yy + zz) * vz
    return np.array([rx, ry, rz])


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = safe_norm(axis)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    half = 0.5 * float(angle)
    s = np.sin(half)
    return quat_normalize(np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s]))


def quat_log(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    v = np.array([x, y, z])
    sin_half = safe_norm(v)
    if sin_half < eps:
        return 2.0 * v
    angle = 2.0 * np.arctan2(sin_half, max(w, -w))
    axis = v / sin_half
    return axis * angle


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cz = np.cos(yaw * 0.5)
    sz = np.sin(yaw * 0.5)
    cy = np.cos(pitch * 0.5)
    sy = np.sin(pitch * 0.5)
    cx = np.cos(roll * 0.5)
    sx = np.sin(roll * 0.5)
    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = cz * sy * cx + sz * cy * sx
    z = sz * cy * cx - cz * sy * sx
    return quat_normalize(np.array([w, x, y, z]))


def project_quat_angle_onto_axis(q: np.ndarray, axis: np.ndarray) -> float:
    q = quat_normalize(q)
    a = np.asarray(axis, dtype=float)
    a_n = safe_norm(a)
    if a_n == 0.0:
        return 0.0
    a = a / a_n
    w, x, y, z = q
    s = x * a[0] + y * a[1] + z * a[2]
    return 2.0 * np.arctan2(s, w)


__all__ = [
    "safe_norm",
    "quat_normalize",
    "quat_conjugate",
    "quat_multiply",
    "quat_rotate_vector",
    "quat_from_axis_angle",
    "quat_log",
    "quat_from_euler",
    "project_quat_angle_onto_axis",
]


