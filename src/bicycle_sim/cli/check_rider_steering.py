"""Experiment B prerequisite: arms physically turn the bar on a diagnostic stand.

Explicit ideal local palm connections and a local seat weld isolate transmission.
This is neither free riding nor a validated Dex3 friction grasp. No bar actuator.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from bicycle_sim.models.bicycle import numbers, sub


def build_rig(path=Path("assets/scenes/rider_fit_diagnostic.xml"), *, fixed_frame=True):
    original = mujoco.MjModel.from_xml_path(str(path.resolve()))
    initial = mujoco.MjData(original)
    mujoco.mj_resetDataKeyframe(original, initial, 0)
    mujoco.mj_forward(original, initial)
    root = ET.parse(path).getroot()
    root.find("compiler").set(
        "meshdir", str((path.parent / root.find("compiler").get("meshdir")).resolve())
    )
    frame = root.find("./worldbody/body[@name='bike_frame']")
    if fixed_frame:
        frame.remove(frame.find("freejoint"))
    root.remove(root.find("keyframe"))
    equality = root.find("equality")
    if equality is None:
        equality = sub(root, "equality")
    pelvis = initial.xpos[original.body("pelvis").id]
    sub(
        equality,
        "weld",
        name="diagnostic_seat_connection",
        body1="bike_frame",
        body2="pelvis",
        relpose=numbers([*pelvis, 1, 0, 0, 0]),
        solref="0.02 1",
    )
    steering = original.body("steering").id
    R = initial.xmat[steering].reshape(3, 3)
    for side in ("left", "right"):
        point = initial.site_xpos[original.site(f"{side}_grasp_reference").id]
        local = R.T @ (point - initial.xpos[steering])
        sub(
            root.find(".//body[@name='steering']"),
            "site",
            name=f"{side}_ideal_grasp_anchor",
            pos=numbers(local),
            size=".004",
        )
        sub(
            equality,
            "connect",
            name=f"{side}_ideal_grasp",
            site1=f"{side}_grasp_reference",
            site2=f"{side}_ideal_grasp_anchor",
            solref="0.02 1",
        )
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    q0 = m.qpos0.copy()
    for j in range(m.njnt):
        old = original.joint(m.joint(j).name).id
        size = 7 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        q0[m.jnt_qposadr[j] : m.jnt_qposadr[j] + size] = initial.qpos[
            original.jnt_qposadr[old] : original.jnt_qposadr[old] + size
        ]
    return m, q0, initial.ctrl.copy()


def audit():
    m, q0, ctrl0 = build_rig()
    arm_joints = [
        m.joint(f"{side}_{part}_joint").id
        for side in ("left", "right")
        for part in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
    qids = m.jnt_qposadr[arm_joints]
    aids = [m.actuator(m.joint(j).name).id for j in arm_joints]
    sites = [m.site(f"{s}_grasp_reference").id for s in ("left", "right")]
    anchors = [m.site(f"{s}_ideal_grasp_anchor").id for s in ("left", "right")]
    steer = m.joint("steer")
    rows = []
    for angle in (-0.15, 0.0, 0.15):
        reference = mujoco.MjData(m)
        reference.qpos[:] = q0
        reference.qpos[steer.qposadr[0]] = angle
        mujoco.mj_kinematics(m, reference)
        targets = reference.site_xpos[anchors].copy()

        def residual(x, reference=reference, targets=targets):
            reference.qpos[qids] = x
            mujoco.mj_kinematics(m, reference)
            return np.r_[(reference.site_xpos[sites] - targets).ravel(), 0.002 * (x - q0[qids])]

        lower, upper = m.jnt_range[arm_joints].T
        fit = least_squares(residual, q0[qids], bounds=(lower + 1e-6, upper - 1e-6), max_nfev=150)
        d = mujoco.MjData(m)
        d.qpos[:], d.ctrl[:] = q0, ctrl0
        errors, reactions, torques = [], [], []
        grip_forces = {side: 0.0 for side in ("left", "right")}
        for i in range(round(2 / m.opt.timestep)):
            blend = min(1.0, i * m.opt.timestep / 0.5)
            d.ctrl[aids] = q0[qids] + blend * (fit.x - q0[qids])
            mujoco.mj_step(m, d)
            if i * m.opt.timestep >= 1.5:
                errors.append(float(d.qpos[steer.qposadr[0]] - angle))
            reactions.append(float(np.linalg.norm(d.qfrc_constraint)))
            for side, previous_force in grip_forces.items():
                eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"{side}_ideal_grasp")
                mask = (d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == eq)
                grip_forces[side] = max(previous_force, float(np.linalg.norm(d.efc_force[mask])))
            torques.append(float(np.max(np.abs(d.qfrc_actuator[m.jnt_dofadr[arm_joints]]))))
        rows.append(
            {
                "target_steer_rad": angle,
                "actual_steer_rad": float(d.qpos[steer.qposadr[0]]),
                "final_window_rmse_rad": float(np.sqrt(np.mean(np.square(errors)))),
                "peak_arm_generalized_torque_Nm": max(torques),
                "peak_ideal_grasp_force_N": grip_forces,
                "peak_generalized_constraint_force_norm_mixed_units": max(reactions),
                "solver_warnings": int(d.warning.number.sum()),
                "finite": bool(np.isfinite(d.qpos).all()),
            }
        )
    return {
        "scope": "fixed stand; ideal local hand connections and seat weld; no steering actuator; no pedaling",
        "free_riding_validated": False,
        "results": rows,
        "passed": all(
            r["finite"] and not r["solver_warnings"] and r["final_window_rmse_rad"] < 0.03
            for r in rows
        ),
    }


def main():
    report = audit()
    path = Path("output/phase2/rider_steering_audit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
