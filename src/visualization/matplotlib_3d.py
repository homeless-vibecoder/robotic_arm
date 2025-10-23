from __future__ import annotations

import numpy as np
from helpers.quaternion import quat_rotate_vector, quat_normalize


def _fixed_limits_from_points(points: np.ndarray, margin: float = 0.1):
    pts = np.asarray(points, dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    lim = span * (0.5 + margin)
    return (
        (center[0] - lim, center[0] + lim),
        (center[1] - lim, center[1] + lim),
        (center[2] - lim, center[2] + lim),
    )


def _fixed_limits_from_history(history, fk_fn, margin: float = 0.1):
    all_pts = []
    for angles in history.angle_history:
        fk = fk_fn(angles)
        all_pts.append(fk.joint_positions)
    all_pts = np.vstack(all_pts)
    return _fixed_limits_from_points(all_pts, margin=margin)


def plot_arm(
    ax,
    joint_positions: np.ndarray,
    fixed_limits=None,
    eef_position: np.ndarray | None = None,
    eef_orientation_quat: np.ndarray | None = None,
) -> None:
    P = np.asarray(joint_positions, dtype=float)
    # Links
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "-", color="tab:blue", linewidth=4.0)
    # Joints (small markers)
    ax.scatter(P[:-1, 0], P[:-1, 1], P[:-1, 2], color="tab:blue", s=15)
    # End-effector marker (different color/shape)
    ax.scatter(P[-1, 0], P[-1, 1], P[-1, 2], color="tab:red", s=80, marker="^")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    if fixed_limits is None:
        mins = P.min(axis=0)
        maxs = P.max(axis=0)
        center = (mins + maxs) / 2.0
        span = float(np.max(maxs - mins))
        if span <= 0:
            span = 1.0
        lim = span * 0.6
        ax.set_xlim(center[0] - lim, center[0] + lim)
        ax.set_ylim(center[1] - lim, center[1] + lim)
        ax.set_zlim(center[2] - lim, center[2] + lim)
    else:
        (xlim, ylim, zlim) = fixed_limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

    # Optional: draw end-effector orientation axes triad
    if eef_position is not None and eef_orientation_quat is not None:
        _draw_axes_triad(ax, np.asarray(eef_position, dtype=float), np.asarray(eef_orientation_quat, dtype=float))


def animate_history(history, fk_fn, interval_ms: int = 80, show: bool = True):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib import animation
    except Exception as exc:
        raise RuntimeError("matplotlib is required for visualization.") from exc

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Precompute fixed axis limits across the entire history
    fixed_limits = _fixed_limits_from_history(history, fk_fn, margin=0.1)

    lines = []

    def init():
        ax.clear()
        (xlim, ylim, zlim) = fixed_limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        return lines

    def update(frame_idx: int):
        ax.clear()
        angles = history.angle_history[frame_idx]
        fk = fk_fn(angles)
        plot_arm(
            ax,
            fk.joint_positions,
            fixed_limits=fixed_limits,
            eef_position=fk.end_effector_position,
            eef_orientation_quat=fk.end_effector_orientation,
        )
        return lines

    frames = len(history.angle_history)
    ani = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=interval_ms, blit=False)
    if show:
        plt.show()
    return ani


__all__ = ["plot_arm", "animate_history"]


def _draw_axes_triad(ax, origin: np.ndarray, quat_wxyz: np.ndarray, scale: float = 0.06):
    q = quat_normalize(quat_wxyz)
    # Basis axes rotated by orientation
    x_axis = quat_rotate_vector(q, np.array([1.0, 0.0, 0.0]))
    y_axis = quat_rotate_vector(q, np.array([0.0, 1.0, 0.0]))
    z_axis = quat_rotate_vector(q, np.array([0.0, 0.0, 1.0]))
    # Endpoints
    px = origin + scale * x_axis
    py = origin + scale * y_axis
    pz = origin + scale * z_axis
    # Draw as colored line segments
    ax.plot([origin[0], px[0]], [origin[1], px[1]], [origin[2], px[2]], color="red", linewidth=2)
    ax.plot([origin[0], py[0]], [origin[1], py[1]], [origin[2], py[2]], color="green", linewidth=2)
    ax.plot([origin[0], pz[0]], [origin[1], pz[1]], [origin[2], pz[2]], color="blue", linewidth=2)


