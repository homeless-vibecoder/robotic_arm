import os
import sys
from pathlib import Path
import numpy as np

# Allow running from repo root without install
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ik_solver import SixDOFArmIK
from visualization import plot_arm
from config import load_calibrated_home


def main():
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as exc:
        raise RuntimeError("matplotlib is required for visualization.") from exc

    # Load arm definition from calibration data
    root = Path(__file__).resolve().parents[1]
    calib_path = root / "config" / "data" / "calibrated_home.json"
    calib = load_calibrated_home(calib_path)
    p_home = calib["p_home"]
    a_home = calib["a_home"]
    tool_offset = np.array([0.05, 0.0, 0.0])

    ik = SixDOFArmIK(p_home=p_home, a_home=a_home, tool_offset=tool_offset)

    # Generate random waypoints via FK from random joint angles (reproducible)
    rng = np.random.default_rng(12345)
    num_waypoints = 3
    waypoints = []
    target_quats = []
    for _ in range(num_waypoints):
        rand_angles = rng.uniform(low=ik.lower_limits, high=ik.upper_limits)
        fk = ik.forward_kinematics(rand_angles)
        waypoints.append(fk.end_effector_position)
        target_quats.append(fk.end_effector_orientation)
    waypoints = np.asarray(waypoints)

    # Prepare plot (fixed axis limits from waypoints)
    def _fixed_limits_from_points(points, margin: float = 0.1):
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

    fixed_limits = _fixed_limits_from_points(waypoints, margin=0.1)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    (xlim, ylim, zlim) = fixed_limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color="tab:orange", s=30, label="waypoints")

    # Follow waypoints indefinitely; animate smooth motion using IK history
    target_speed = 0.15  # meters per second (approximate)
    min_dt = 0.01
    max_dt = 0.08

    try:
        while plt.fignum_exists(fig.number):
            for wp, tq in zip(waypoints, target_quats):
                if not plt.fignum_exists(fig.number):
                    break
                sol = ik.solve_ik(
                    target_position=wp,
                    target_orientation_quat=tq,
                    record_history=True,
                    max_iters=200,
                )
                # Animate along intermediate poses for smooth motion at roughly constant velocity
                if sol.history is not None and len(sol.history.angle_history) > 0:
                    prev_p = None
                    for angles in sol.history.angle_history:
                        if not plt.fignum_exists(fig.number):
                            break
                        fk = ik.forward_kinematics(angles)
                        ax.clear()
                        (xlim, ylim, zlim) = fixed_limits
                        ax.set_xlim(*xlim)
                        ax.set_ylim(*ylim)
                        ax.set_zlim(*zlim)
                        ax.scatter(
                            waypoints[:, 0],
                            waypoints[:, 1],
                            waypoints[:, 2],
                            color="tab:orange",
                            s=30,
                            label="waypoints",
                        )
                        plot_arm(
                            ax,
                            fk.joint_positions,
                            fixed_limits=fixed_limits,
                            eef_position=fk.end_effector_position,
                            eef_orientation_quat=fk.end_effector_orientation,
                        )
                        if prev_p is None:
                            dt = min_dt
                        else:
                            dist = float(np.linalg.norm(fk.end_effector_position - prev_p))
                            dt = max(min_dt, min(max_dt, (dist / target_speed) if target_speed > 0 else min_dt))
                        prev_p = fk.end_effector_position
                        plt.pause(dt)
                else:
                    # Fallback: jump to final pose if no history
                    fk = ik.forward_kinematics(sol.angles)
                    ax.clear()
                    (xlim, ylim, zlim) = fixed_limits
                    ax.set_xlim(*xlim)
                    ax.set_ylim(*ylim)
                    ax.set_zlim(*zlim)
                    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color="tab:orange", s=30, label="waypoints")
                    plot_arm(
                        ax,
                        fk.joint_positions,
                        fixed_limits=fixed_limits,
                        eef_position=fk.end_effector_position,
                        eef_orientation_quat=fk.end_effector_orientation,
                    )
                    plt.pause(max_dt)
    except KeyboardInterrupt:
        pass

    if plt.fignum_exists(fig.number):
        plt.show()


if __name__ == "__main__":
    main()


