"""Compare symmetry, small lean and motion without learned balance control."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from bicycle_sim.models.bicycle import build_bicycle
from bicycle_sim.rl.bicycle_env import training_tree


def audit():
    bike = json.loads(Path("configs/bicycle_v0.json").read_text())
    cfg = json.loads(Path("configs/rl/bicycle_a.json").read_text())
    rows = []
    for control in ("unpowered", "fixed_commands"):
        root = build_bicycle(bike) if control == "unpowered" else training_tree(bike, cfg)
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        frame = m.body("bike_frame").id
        for speed in (0.0, 2.0, 5.0):
            for tilt in (0.0, 0.01, 0.1, 1.0):
                d = mujoco.MjData(m)
                a = np.deg2rad(tilt)
                d.qpos[3:7] = [np.cos(a / 2), np.sin(a / 2), 0, 0]
                d.qvel[0] = speed
                for name in ("rear_wheel_spin", "front_wheel_spin"):
                    d.qvel[m.joint(name).dofadr[0]] = speed / 0.27
                d.qvel[m.joint("crank_spin").dofadr[0]] = speed / 0.54
                if control == "fixed_commands":
                    d.ctrl[m.actuator("crank_drive").id] = speed / 0.54
                    d.ctrl[m.actuator("steer_drive").id] = 0
                fall = None
                max_roll = 0.0
                for _ in range(round(20 / m.opt.timestep)):
                    mujoco.mj_step(m, d)
                    R = d.xmat[frame].reshape(3, 3)
                    roll = abs(float(np.arctan2(R[2, 1], R[2, 2])))
                    max_roll = max(max_roll, roll)
                    if roll > 0.65:
                        fall = float(d.time)
                        break
                rows.append(
                    {
                        "control": control,
                        "initial_speed_mps": speed,
                        "initial_roll_deg": tilt,
                        "time_to_0.65rad_s": fall,
                        "survived_20s": fall is None,
                        "maximum_roll_rad": max_roll,
                        "warnings": int(d.warning.number.sum()),
                    }
                )
    return {"duration_s": 20, "no_rider": True, "results": rows}


def main():
    report = audit()
    output = Path("output/phase2/balance_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["results"]:
        print(row)


if __name__ == "__main__":
    main()
