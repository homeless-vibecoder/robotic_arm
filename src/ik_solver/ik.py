from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from helpers.quaternion import (
    quat_normalize,
    quat_conjugate,
    quat_multiply,
    quat_rotate_vector,
    quat_from_axis_angle,
    quat_log,
    project_quat_angle_onto_axis,
)


@dataclass
class FKResult:
    joint_positions: np.ndarray  # (N,3)
    joint_axes_world: np.ndarray  # (N,3)
    end_effector_position: np.ndarray  # (3,)
    end_effector_orientation: np.ndarray  # quaternion [w,x,y,z]


@dataclass
class IKSolution:
    success: bool
    angles: np.ndarray
    fk: FKResult
    iters: int
    message: str
    history: Optional["IKHistory"] = None


@dataclass
class IKHistory:
    iterations: List[int]
    angle_history: List[np.ndarray]
    eef_positions: List[np.ndarray]
    pos_err_norms: List[float]
    ori_err_norms: List[float]
    total_err_norms: List[float]
    step_norms: List[float]
    lambdas: List[float]


class SixDOFArmIK:
    def __init__(
        self,
        p_home: np.ndarray,
        a_home: np.ndarray,
        tool_offset: Optional[np.ndarray] = None,
        lower_limits: Optional[np.ndarray] = None,
        upper_limits: Optional[np.ndarray] = None,
    ) -> None:
        p_home = np.asarray(p_home, dtype=float)
        a_home = np.asarray(a_home, dtype=float)
        if p_home.ndim != 2 or p_home.shape[1] != 3:
            raise ValueError("p_home must have shape (N,3)")
        if a_home.shape != p_home.shape:
            raise ValueError("a_home must have shape (N,3) and match p_home")
        self.N: int = int(p_home.shape[0])
        if self.N < 2:
            raise ValueError("Arm must have at least 2 joints")
        a_norms = np.linalg.norm(a_home, axis=1)
        if np.any(a_norms <= 0.0):
            raise ValueError("All joint axes must be non-zero vectors")
        self.p_home: np.ndarray = p_home.copy()
        self.a_home: np.ndarray = (a_home / a_norms[:, None]).copy()
        self.tool_offset: np.ndarray = (
            np.zeros(3, dtype=float) if tool_offset is None else np.asarray(tool_offset, dtype=float)
        )
        self.angles: np.ndarray = np.zeros(self.N, dtype=float)
        if lower_limits is None:
            self.lower_limits = -0.5 * np.pi * np.ones(self.N, dtype=float)
        else:
            ll = np.asarray(lower_limits, dtype=float)
            if ll.shape != (self.N,):
                raise ValueError(f"lower_limits must have shape ({self.N},)")
            self.lower_limits = ll
        if upper_limits is None:
            self.upper_limits = 0.5 * np.pi * np.ones(self.N, dtype=float)
        else:
            ul = np.asarray(upper_limits, dtype=float)
            if ul.shape != (self.N,):
                raise ValueError(f"upper_limits must have shape ({self.N},)")
            self.upper_limits = ul
        if np.any(self.lower_limits >= self.upper_limits):
            raise ValueError("Each lower limit must be strictly less than upper limit")

    def set_state_from_angles(self, angles: np.ndarray) -> None:
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (self.N,):
            raise ValueError(f"angles must have shape ({self.N},)")
        self.angles = np.clip(angles, self.lower_limits, self.upper_limits)

    def set_state_from_quaternions(self, quats: np.ndarray) -> None:
        quats = np.asarray(quats, dtype=float)
        if quats.shape != (self.N, 4):
            raise ValueError(f"quaternions must have shape ({self.N}, 4)")
        angles = np.zeros(self.N, dtype=float)
        for i in range(self.N):
            angles[i] = project_quat_angle_onto_axis(quat_normalize(quats[i]), self.a_home[i])
        self.angles = np.clip(angles, self.lower_limits, self.upper_limits)

    def set_joint_limits(self, lower_limits: np.ndarray, upper_limits: np.ndarray) -> None:
        ll = np.asarray(lower_limits, dtype=float)
        ul = np.asarray(upper_limits, dtype=float)
        if ll.shape != (self.N,) or ul.shape != (self.N,):
            raise ValueError(f"limits must each have shape ({self.N},)")
        if np.any(ll >= ul):
            raise ValueError("Each lower limit must be strictly less than upper limit")
        self.lower_limits = ll
        self.upper_limits = ul
        self.angles = np.clip(self.angles, self.lower_limits, self.upper_limits)

    def forward_kinematics(self, angles: Optional[np.ndarray] = None) -> FKResult:
        if angles is None:
            angles = self.angles
        else:
            angles = np.asarray(angles, dtype=float)
            if angles.shape != (self.N,):
                raise ValueError(f"angles must have shape ({self.N},)")

        P = self.p_home.copy()
        joint_axes_world = np.zeros_like(self.a_home)
        q_cum = np.array([1.0, 0.0, 0.0, 0.0])

        for i in range(self.N):
            a_world = quat_rotate_vector(q_cum, self.a_home[i])
            joint_axes_world[i] = a_world

            q_i = quat_from_axis_angle(a_world, angles[i])
            for j in range(i + 1, self.N):
                rel = P[j] - P[i]
                P[j] = P[i] + quat_rotate_vector(q_i, rel)

            q_cum = quat_normalize(quat_multiply(q_i, q_cum))

        p_end = P[-1] + quat_rotate_vector(q_cum, self.tool_offset)
        q_end = quat_normalize(q_cum)

        return FKResult(
            joint_positions=P,
            joint_axes_world=joint_axes_world,
            end_effector_position=p_end,
            end_effector_orientation=q_end,
        )

    def compute_jacobian(self, angles: Optional[np.ndarray] = None) -> Tuple[np.ndarray, FKResult]:
        fk = self.forward_kinematics(angles)
        p_e = fk.end_effector_position
        P = fk.joint_positions
        A = fk.joint_axes_world
        J = np.zeros((6, self.N), dtype=float)
        for i in range(self.N):
            ai = A[i]
            pi = P[i]
            J[0:3, i] = np.cross(ai, (p_e - pi))
            J[3:6, i] = ai
        return J, fk

    def solve_ik(
        self,
        target_position: np.ndarray,
        target_orientation_quat: np.ndarray,
        initial_angles: Optional[np.ndarray] = None,
        max_iters: int = 200,
        position_weight: float = 1.0,
        orientation_weight: float = 0.5,
        damping: float = 1e-3,
        min_damping: float = 1e-6,
        step_limit: float = 0.5,
        pos_tol: float = 1e-2,
        ori_tol: float = 1e-2,
        record_history: bool = False,
    ) -> IKSolution:
        target_p = np.asarray(target_position, dtype=float).reshape(3)
        target_q = quat_normalize(np.asarray(target_orientation_quat, dtype=float).reshape(4))
        if initial_angles is None:
            theta = self.angles.copy()
        else:
            theta = np.asarray(initial_angles, dtype=float)
            if theta.shape != (self.N,):
                raise ValueError(f"initial_angles must have shape ({self.N},)")
        theta = np.clip(theta, self.lower_limits, self.upper_limits)

        lam = float(damping)
        prev_err = None

        history: Optional[IKHistory] = None
        if record_history:
            history = IKHistory(
                iterations=[],
                angle_history=[],
                eef_positions=[],
                pos_err_norms=[],
                ori_err_norms=[],
                total_err_norms=[],
                step_norms=[],
                lambdas=[],
            )

        for it in range(1, max_iters + 1):
            J, fk = self.compute_jacobian(theta)
            p = fk.end_effector_position
            q = fk.end_effector_orientation

            e_pos = target_p - p
            q_err = quat_multiply(target_q, quat_conjugate(q))
            e_rot = quat_log(q_err)

            e = np.hstack((position_weight * e_pos, orientation_weight * e_rot))

            W = np.diag([
                position_weight,
                position_weight,
                position_weight,
                orientation_weight,
                orientation_weight,
                orientation_weight,
            ])
            JT_W = J.T @ W
            A = JT_W @ J + (lam * lam) * np.eye(self.N)
            b = JT_W @ e

            try:
                delta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                A = JT_W @ J + (max(lam * 10.0, 1e-2) ** 2) * np.eye(self.N)
                delta = np.linalg.lstsq(A, b, rcond=None)[0]

            d_norm = float(np.linalg.norm(delta))
            if d_norm > step_limit and d_norm > 0.0:
                delta = (delta / d_norm) * step_limit

            theta = theta + delta
            theta = np.clip(theta, self.lower_limits, self.upper_limits)

            err_pos_norm = float(np.linalg.norm(e_pos))
            err_ori_norm = float(np.linalg.norm(e_rot))
            err_total = err_pos_norm + err_ori_norm

            if history is not None:
                history.iterations.append(it)
                history.angle_history.append(theta.copy())
                history.eef_positions.append(p.copy())
                history.pos_err_norms.append(err_pos_norm)
                history.ori_err_norms.append(err_ori_norm)
                history.total_err_norms.append(err_total)
                history.step_norms.append(d_norm)
                history.lambdas.append(lam)

            if err_pos_norm <= pos_tol and err_ori_norm <= ori_tol:
                self.angles = np.clip(theta, self.lower_limits, self.upper_limits)
                final_fk = self.forward_kinematics(theta)
                return IKSolution(
                    success=True,
                    angles=theta,
                    fk=final_fk,
                    iters=it,
                    message="Converged",
                    history=history,
                )

            if prev_err is not None:
                if err_total > prev_err * 1.001:
                    lam = min(1e1, lam * 5.0)
                else:
                    lam = max(min_damping, lam / 1.2)
            prev_err = err_total

        self.angles = theta.copy()
        final_fk = self.forward_kinematics(theta)
        return IKSolution(
            success=False,
            angles=theta,
            fk=final_fk,
            iters=max_iters,
            message="Max iterations reached without convergence",
            history=history,
        )


__all__ = [
    "SixDOFArmIK",
    "IKSolution",
    "FKResult",
    "IKHistory",
]


