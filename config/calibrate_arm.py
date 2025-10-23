from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np


def _normalize_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return M / norms


def calibrate_home_from_measurements(
    measurements_path: str | Path,
    output_path: str | Path,
    max_joints: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Compute calibrated home pivots `p_home` and axes `a_home` for an N-DOF arm
    by averaging across provided measurements (per joint index). Axes are
    normalized after averaging. Supports arbitrary N (N ≥ 2).

    - measurements_path: JSON with key "measurements", a list of entries each
      having "joint_positions_world" and "joint_axes_world".
    - output_path: where to write calibrated_home.json
    - max_joints: optional cap on number of joints to consider.
    """
    mp = Path(measurements_path)
    data = json.loads(mp.read_text())
    measurements: List[Dict[str, Any]] = data.get("measurements", [])
    if not measurements:
        raise ValueError("No measurements found in measurements.json")

    # Determine joint count robustly from the first entry
    n_pos_first = len(measurements[0].get("joint_positions_world", []))
    n_ax_first = len(measurements[0].get("joint_axes_world", []))
    n_joints = min(n_pos_first, n_ax_first)
    if max_joints is not None:
        n_joints = min(n_joints, int(max_joints))

    sum_p = np.zeros((n_joints, 3), dtype=float)
    sum_a = np.zeros((n_joints, 3), dtype=float)
    count = 0

    for m in measurements:
        P = np.asarray(m.get("joint_positions_world", []), dtype=float)
        A = np.asarray(m.get("joint_axes_world", []), dtype=float)
        if P.ndim != 2 or P.shape[1] != 3:
            continue
        if A.ndim != 2 or A.shape[1] != 3:
            continue
        k = min(n_joints, P.shape[0], A.shape[0])
        sum_p[:k] += P[:k]
        sum_a[:k] += A[:k]
        count += 1

    if count == 0:
        raise ValueError("No valid measurement entries with 3D positions and axes")

    p_home = sum_p / float(count)
    a_home = _normalize_rows(sum_a / float(count))

    out = {
        "p_home": p_home.tolist(),
        "a_home": a_home.tolist(),
    }

    op = Path(output_path)
    op.write_text(json.dumps(out, indent=2))
    return {"p_home": p_home, "a_home": a_home}


def main():
    # Hardcoded paths per user's preference (no argparse)
    root = Path(__file__).resolve().parent
    measurements_path = root / "data" / "measurements.json"
    output_path = root / "data" / "calibrated_home.json"
    calibrate_home_from_measurements(measurements_path, output_path)
    print(f"Wrote calibrated home to: {output_path}")


if __name__ == "__main__":
    main()


