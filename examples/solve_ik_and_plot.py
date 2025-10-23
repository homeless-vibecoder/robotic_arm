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
from helpers import quat_from_euler
from visualization import animate_history
from config import load_calibrated_home


def main():
    # Load arm definition from calibration data
    root = Path(__file__).resolve().parents[1]
    calib_path = root / "config" / "data" / "calibrated_home.json"
    calib = load_calibrated_home(calib_path)
    p_home = calib["p_home"]
    a_home = calib["a_home"]
    tool_offset = np.array([0.05, 0.0, 0.0])

    ik = SixDOFArmIK(p_home=p_home, a_home=a_home, tool_offset=tool_offset)

    # Pseudorandom initial joint angles (reproducible)
    rng = np.random.default_rng(12345)
    init_angles = rng.uniform(low=ik.lower_limits, high=ik.upper_limits)

    target_p = p_home[-1] + np.array([0.05, 0.05, 0.05])
    target_q = quat_from_euler(0.0, 0.0, 0.0)

    sol = ik.solve_ik(
        target_position=target_p,
        target_orientation_quat=target_q,
        initial_angles=init_angles,
        record_history=True,
        max_iters=200,
    )

    if not sol.success:
        print("Warning: IK did not converge; showing progress anyway.")

    animate_history(sol.history, ik.forward_kinematics, interval_ms=60, show=True)


if __name__ == "__main__":
    main()


