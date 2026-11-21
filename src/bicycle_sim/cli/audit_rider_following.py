"""Fixed-frame, airborne-wheel diagnostic for the actual motor-assisted B setup.

Tests phase-following legs, unactuated pedal bearings and fixed waist/finger PD.
It does not test active pedaling, real grasping or free-riding balance.
"""

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from bicycle_sim.rl.rider_model import build_rider_model


def build_fixed(cfg):
    source, q0, ctrl, maps = build_rider_model(cfg)
    with tempfile.TemporaryDirectory(prefix="rider-follow-", dir="/tmp") as tmp:
        path = Path(tmp) / "scene.xml"
        mujoco.mj_saveLastXML(str(path), source)
        tree = ET.parse(path).getroot()
    frame = tree.find("./worldbody/body[@name='bike_frame']")
    for joint in list(frame):
        if joint.get("name") == "bike_root":
            frame.remove(joint)
    world = tree.find("worldbody")
    world.remove(world.find("./geom[@name='ground']"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    # Only the bike's leading free joint was removed. Actuator indices remain unchanged.
    return model, q0[7:].copy(), ctrl, maps["leg_act"], maps["leg_table"]


def audit(duration=12.0, *, velocity_feedforward=None):
    cfg = json.loads(Path("configs/rl/bicycle_b.json").read_text())
    if velocity_feedforward is None:
        velocity_feedforward = cfg.get("leg_velocity_feedforward", False)
    m, q0, ctrl, leg_act, table = build_fixed(cfg)
    d = mujoco.MjData(m)
    d.qpos[:], d.ctrl[:] = q0, ctrl
    crank_q = m.joint("crank_spin").qposadr[0]
    crank_v = m.joint("crank_spin").dofadr[0]
    motor = m.actuator("crank_drive").id
    waist_names = [f"waist_{part}_joint" for part in ("yaw", "roll", "pitch")]
    waist_j = [m.joint(n).id for n in waist_names]
    waist_q, waist_v = m.jnt_qposadr[waist_j], m.jnt_dofadr[waist_j]
    waist_limit = np.max(np.abs(m.jnt_actfrcrange[waist_j]), axis=1)
    finger_names = [m.actuator(a).name for a in range(m.nu) if "_hand_" in m.actuator(a).name]
    finger_j = [m.joint(n).id for n in finger_names]
    finger_q = m.jnt_qposadr[finger_j]
    sides = ("left", "right")
    foot_bodies = [m.body(f"{s}_ankle_roll_link").id for s in sides]
    pedal_bodies = [m.body(f"{s}_pedal").id for s in sides]
    soles = [m.site(f"{s}_sole_reference").id for s in sides]
    pedal_sites = [m.site(f"{s}_pedal_contact").id for s in sides]
    pedal_q = [m.joint(f"{s}_pedal_spin").qposadr[0] for s in sides]
    records = []
    work = {"motor_J": 0.0, "feet_to_pedals_J": 0.0}
    dt = m.opt.timestep
    for _ in range(round(duration / dt)):
        phase = (d.qpos[crank_q] % (2 * np.pi)) * 24 / (2 * np.pi)
        i = min(23, int(phase))
        target = table[i] * (1 - (phase - i)) + table[i + 1] * (phase - i)
        if velocity_feedforward:
            target_velocity = (table[i + 1] - table[i]) * 24 / (2 * np.pi) * d.qvel[crank_v]
            kp, kv = m.actuator_gainprm[leg_act, 0], -m.actuator_biasprm[leg_act, 2]
            target += kv / kp * target_velocity
        d.ctrl[leg_act] = target
        mujoco.mj_forward(m, d)
        normal = np.zeros(2)
        pedal_contact_power = 0.0
        for contact_id, contact in enumerate(d.contact):
            bodies = [m.geom_bodyid[g] for g in (contact.geom1, contact.geom2)]
            for side in range(2):
                if set(bodies) != {foot_bodies[side], pedal_bodies[side]}:
                    continue
                local_force = np.zeros(6)
                mujoco.mj_contactForce(m, d, contact_id, local_force)
                normal[side] += local_force[0]
                force = contact.frame.reshape(3, 3).T @ local_force[:3]
                if bodies[0] == pedal_bodies[side]:
                    force = -force
                jac = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jac, None, contact.pos, pedal_bodies[side])
                pedal_contact_power += float(force @ (jac @ d.qvel))
        motor_power = float(d.actuator_force[motor] * d.actuator_velocity[motor])
        work["motor_J"] += motor_power * dt
        work["feet_to_pedals_J"] += pedal_contact_power * dt
        mismatch = [
            np.arccos(
                np.clip(
                    d.site_xmat[a].reshape(3, 3)[:, 2] @ d.site_xmat[b].reshape(3, 3)[:, 2], -1, 1
                )
            )
            for a, b in zip(soles, pedal_sites, strict=True)
        ]
        grip_force = []
        for side in sides:
            eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"{side}_ideal_grasp")
            mask = (d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == eq)
            grip_force.append(float(np.linalg.norm(d.efc_force[mask])))
        records.append(
            {
                "time": float(d.time),
                "crank_speed": float(d.qvel[crank_v]),
                "normal": normal,
                "foot_distance": np.linalg.norm(
                    d.site_xpos[soles] - d.site_xpos[pedal_sites], axis=1
                ),
                "orientation_mismatch": np.array(mismatch),
                "pedal_relative": d.qpos[pedal_q].copy(),
                "waist_error": d.qpos[waist_q] - q0[waist_q],
                "waist_torque": d.qfrc_actuator[waist_v].copy(),
                "finger_error": float(np.max(np.abs(d.qpos[finger_q] - q0[finger_q]))),
                "grip_force": np.array(grip_force),
            }
        )
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or d.warning.number.any():
            raise FloatingPointError("Native dynamics failed; inspect the rig")
    settled = [r for r in records if r["time"] >= 2.0]
    if not settled:
        raise ValueError("Duration must exceed 2 seconds")

    def array(name):
        return np.asarray([r[name] for r in settled])

    report = {
        "scope": "fixed frame; wheels airborne; original B motor, local seat/palm connections and leg phase PD; no learned actions",
        "duration_s": duration,
        "statistics_after_s": 2.0,
        "leg_velocity_feedforward": velocity_feedforward,
        "crank_turns": float((d.qpos[crank_q] - q0[crank_q]) / (2 * np.pi)),
        "crank_speed_mean_rad_s": float(array("crank_speed").mean()),
        "crank_speed_std_rad_s": float(array("crank_speed").std()),
        "motor_target_rad_s": cfg["crank_speed_rad_s"],
        "contact_fraction_force_over_1N": dict(
            zip(sides, (array("normal") > 1).mean(axis=0).tolist(), strict=True)
        ),
        "normal_force_max_N": dict(zip(sides, array("normal").max(axis=0).tolist(), strict=True)),
        "foot_reference_distance_rms_m": dict(
            zip(sides, np.sqrt(np.mean(array("foot_distance") ** 2, axis=0)).tolist(), strict=True)
        ),
        "foot_reference_distance_max_m": dict(
            zip(sides, array("foot_distance").max(axis=0).tolist(), strict=True)
        ),
        "foot_pedal_normal_mismatch_max_deg": dict(
            zip(sides, np.rad2deg(array("orientation_mismatch").max(axis=0)).tolist(), strict=True)
        ),
        "pedal_relative_angle_range_rad": dict(
            zip(sides, np.ptp(array("pedal_relative"), axis=0).tolist(), strict=True)
        ),
        "waist_max_error_deg": dict(
            zip(
                waist_names,
                np.rad2deg(np.abs(array("waist_error")).max(axis=0)).tolist(),
                strict=True,
            )
        ),
        "waist_max_torque_Nm": dict(
            zip(waist_names, np.abs(array("waist_torque")).max(axis=0).tolist(), strict=True)
        ),
        "waist_saturation_fraction": dict(
            zip(
                waist_names,
                (np.abs(array("waist_torque")) >= waist_limit - 1e-3).mean(axis=0).tolist(),
                strict=True,
            )
        ),
        "max_finger_error_deg": float(np.rad2deg(array("finger_error").max())),
        "ideal_grip_force_max_N": dict(
            zip(sides, array("grip_force").max(axis=0).tolist(), strict=True)
        ),
        "work": work,
        "solver_warnings": int(d.warning.number.sum()),
    }
    return report, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument(
        "--leg-velocity-feedforward", action=argparse.BooleanOptionalAction, default=None
    )
    p.add_argument("--output", type=Path, default=Path("output/phase2/rider_following_audit"))
    args = p.parse_args()
    report, records = audit(args.duration, velocity_feedforward=args.leg_velocity_feedforward)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "timeseries.npz",
        **{k: np.asarray([r[k] for r in records]) for k in records[0]},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
