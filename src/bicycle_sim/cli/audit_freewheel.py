"""Reproducible isolated freewheel audit (not a riding success test)."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from bicycle_sim.models.bicycle import build_bicycle


def audit(config):
    root = build_bicycle(config, fixed_frame=True)
    root.find("worldbody").remove(root.find("./worldbody/geom[@name='ground']"))
    rows = []
    for dt in (0.002, 0.001, 0.0005):
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        m.opt.timestep, m.opt.gravity[:] = dt, 0
        c, r = m.joint("crank_spin").dofadr[0], m.joint("rear_wheel_spin").dofadr[0]
        for name, vc, vr, force in (
            ("forward_drive", 0, 0, 1),
            ("coast", 0, 10, 0),
            ("backpedal", -4, 0, 0),
            ("overrun", 2, 10, 0),
        ):
            d = mujoco.MjData(m)
            d.qvel[c], d.qvel[r] = vc, vr
            d.qfrc_applied[c] = force
            peak_rear, max_power, loss = 0.0, -np.inf, 0.0
            for _ in range(round(1 / dt)):
                mujoco.mj_forward(m, d)
                power = float(np.dot(d.qfrc_actuator, d.qvel))
                max_power = max(max_power, power)
                loss -= power * dt
                peak_rear = max(peak_rear, abs(d.qfrc_actuator[r]))
                mujoco.mj_step(m, d)
            rows.append(
                {
                    "case": name,
                    "timestep_s": dt,
                    "crank_velocity_rad_s": float(d.qvel[c]),
                    "rear_velocity_rad_s": float(d.qvel[r]),
                    "crank_angle_rad": float(d.qpos[m.joint("crank_spin").qposadr[0]]),
                    "rear_angle_rad": float(d.qpos[m.joint("rear_wheel_spin").qposadr[0]]),
                    "peak_rear_clutch_torque_Nm": peak_rear,
                    "maximum_clutch_power_W": max_power,
                    "clutch_dissipation_J": loss,
                    "solver_warnings": int(d.warning.number.sum()),
                }
            )
    return {"scope": "fixed airborne rig, zero gravity, internal clutch only", "results": rows}


def main():
    config = json.loads(Path("configs/bicycle_v0.json").read_text())
    result = audit(config)
    path = Path("output/phase2/freewheel_audit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
