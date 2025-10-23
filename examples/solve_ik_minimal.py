import sys
from pathlib import Path
import numpy as np

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ik_solver import SixDOFArmIK
from helpers import quat_from_euler
from config import load_calibrated_home


def main():
    # Load calibrated arm
    calib_path = ROOT / "config" / "data" / "calibrated_home.json"
    calib = load_calibrated_home(calib_path)
    p_home = calib["p_home"]
    a_home = calib["a_home"]

    ik = SixDOFArmIK(p_home=p_home, a_home=a_home, tool_offset=np.array([0.05, 0.0, 0.0]))

    # Target pose
    target_p = p_home[-1] + np.array([0.05, 0.02, 0.03])
    target_q = quat_from_euler(0.0, 0.0, 0.0)

    sol = ik.solve_ik(target_position=target_p, target_orientation_quat=target_q, record_history=False)
    print("Success:", sol.success)
    print("Angles (rad):", np.array2string(sol.angles, precision=4))
    print("EEF position:", np.array2string(sol.fk.end_effector_position, precision=4))


if __name__ == "__main__":
    main()


