"""Kinematic preparation of foot-pedal contact; never a runtime RL override."""

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from bicycle_sim.rl.rider_model import LEG_PARTS


def leg_pose_ik(m, qpos, *, clearance=0.0002, pedal_pitch_rad=0.08, strict=True):
    """Keep the actual pelvis; place slightly pitched soles above pedal sites."""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    joints = [m.joint(f"{s}_{p}_joint").id for s in ("left", "right") for p in LEG_PARTS]
    qi = m.jnt_qposadr[joints]
    mujoco.mj_kinematics(m, d)
    foot = [m.site(f"{s}_sole_reference").id for s in ("left", "right")]
    pedal = [m.site(f"{s}_pedal_contact").id for s in ("left", "right")]
    # Do not chase the passive pedal's rotation: that would remove the foot's
    # restoring orientation and allow the combined foot/pedal to tip under gravity.
    rotations = np.repeat(d.xmat[m.body("bike_frame").id].reshape(1, 3, 3), 2, axis=0)
    rotations = rotations @ Rotation.from_rotvec([0, pedal_pitch_rad, 0]).as_matrix()
    targets = d.site_xpos[pedal].copy() + clearance * rotations[:, :, 2]

    def residual(x):
        d.qpos[qi] = x
        mujoco.mj_kinematics(m, d)
        orientation = [
            Rotation.from_matrix(rotations[i].T @ d.site_xmat[s].reshape(3, 3)).as_rotvec()
            for i, s in enumerate(foot)
        ]
        return np.r_[(d.site_xpos[foot] - targets).ravel(), 0.15 * np.asarray(orientation).ravel()]

    lo, hi = m.jnt_range[joints].T
    fit = least_squares(
        residual,
        np.clip(qpos[qi], lo + 1e-7, hi - 1e-7),
        bounds=(lo + 1e-7, hi - 1e-7),
        max_nfev=60,
        gtol=1e-10,
    )
    errors = residual(fit.x)
    position_error = float(np.linalg.norm(errors[:6].reshape(2, 3), axis=1).max())
    angle_error = float(np.linalg.norm(errors[6:].reshape(2, 3) / 0.15, axis=1).max())
    if strict and (position_error > 0.001 or angle_error > 0.02):
        raise RuntimeError(
            f"Seated leg IK is unreachable: position={position_error:g} m, orientation={angle_error:g} rad"
        )
    return fit.x, {"max_position_error_m": position_error, "max_angle_error_rad": angle_error}


def refit_leg_cycle(m, settled_q, *, clearance=0.0002, pedal_pitch_rad=0.08):
    """Fit 25 cyclic poses for the settled pelvis and a nominal pedal pitch."""
    q = settled_q.copy()
    leg_joints = [m.joint(f"{s}_{p}_joint").id for s in ("left", "right") for p in LEG_PARTS]
    qi = m.jnt_qposadr[leg_joints]
    cq = m.joint("crank_spin").qposadr[0]
    pedals = [m.joint(f"{s}_pedal_spin").qposadr[0] for s in ("left", "right")]
    table, checks = [], []
    for phase in np.linspace(0, 2 * np.pi, 25):
        q[cq] = phase
        q[pedals] = -phase + pedal_pitch_rad
        pose, report = leg_pose_ik(m, q, clearance=clearance, pedal_pitch_rad=pedal_pitch_rad)
        q[qi] = pose
        table.append(pose)
        checks.append(report)
    table = np.array(table)
    closure_error = float(np.max(np.abs(table[-1] - table[0])))
    if closure_error > 0.01:
        raise RuntimeError(f"New leg cycle does not close: {closure_error:g} rad")
    table[-1] = table[0]
    return table, {
        "max_position_error_m": max(c["max_position_error_m"] for c in checks),
        "max_angle_error_rad": max(c["max_angle_error_rad"] for c in checks),
        "closure_error_rad": closure_error,
    }
