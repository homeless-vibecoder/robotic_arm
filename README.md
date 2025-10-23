## Robotic Arm (src layout)

![Robotic arm demo](images_(ignore)/Screenshot 2025-10-22 at 22.52.18.png)

N-DOF inverse kinematics with calibration helpers and 3D matplotlib visualization.

### What this project does
Solves IK of an N-DOF robotic arm whose joints rotate in a plane (servo-like). Includes minimal and visualization-rich demos.

### Features
- Supports arbitrary DOF arms
- Jacobian-based IK with damping
- Quaternion-based orientation handling

### Repository Layout
- `src/robotic_arm/`: helpers, IK solver, visualization
- `config/`: calibration scripts and data
- `examples/`: runnable scripts

### Calibration Workflow
1) Edit `config/data/measurements.json` and add one or more measurement entries with:
   - `joint_positions_world`: list of N `[x,y,z]`
   - `joint_axes_world`: list of N unit axes of rotation `[x,y,z]`
   - You may also record the servo angles used when capturing each measurement; calibration infers `p_home` and `a_home` from these data
2) Run calibration to produce `calibrated_home.json`:
```
python config/calibrate_arm.py
```
3) `calibrated_home.json` encodes `p_home` and `a_home` for your arm.

### Minimal Solver Usage
See `examples/solve_ik_minimal.py` for a small script. In short:
```python
from pathlib import Path
import numpy as np
from robotic_arm.ik_solver import SixDOFArmIK
from robotic_arm.helpers import quat_from_euler
from config import load_calibrated_home

root = Path(__file__).resolve().parents[1]
calib = load_calibrated_home(root/"config"/"data"/"calibrated_home.json")
ik = SixDOFArmIK(calib["p_home"], calib["a_home"], tool_offset=np.array([0.05,0,0]))
sol = ik.solve_ik(target_position=calib["p_home"][-1] + np.array([0.05,0.02,0.03]), target_orientation_quat=quat_from_euler(0,0,0))
```

### Examples
- `examples/solve_ik_and_plot.py`: IK with optimization history animation
- `examples/demo_follow_waypoints.py`: looping random waypoints with fixed-scale axes and smooth motion
- `examples/solve_ik_minimal.py`: solver-only demo (no extra visualization)

