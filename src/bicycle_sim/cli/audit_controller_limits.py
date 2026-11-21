"""Test configured arm authority on a fixed bicycle stand, without a steer motor."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from bicycle_sim.cli.check_rider_steering import build_rig
from bicycle_sim.cli.diagnose_rider import arm_references
from bicycle_sim.rl.rider_model import ARM_PARTS


def audit(cfg):
    angles, offsets, errors = arm_references(
        cfg["steering_reference_limit_rad"], cfg["arm_offset_limit_rad"]
    )
    m, q0, ctrl0 = build_rig()
    joints = [m.joint(f"{s}_{p}_joint").id for s in ("left", "right") for p in ARM_PARTS]
    qi, vi = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    actuators = [m.actuator(m.joint(j).name).id for j in joints]
    steer = m.joint("steer").qposadr[0]
    rows = []
    for idx in (0, len(angles) // 2, len(angles) - 1):
        d = mujoco.MjData(m)
        d.qpos[:], d.ctrl[:] = q0, ctrl0
        offset = np.zeros(len(joints))
        positions, torques, saturated, depths = [], [], [], []
        for _ in range(round(2 / m.opt.timestep)):
            step = cfg["arm_target_rate_rad_s"] * m.opt.timestep
            offset += np.clip(offsets[idx] - offset, -step, step)
            d.ctrl[actuators] = q0[qi] + offset
            mujoco.mj_step(m, d)
            force = abs(d.qfrc_actuator[vi])
            torques.append(force.copy())
            saturated.append(force >= np.max(abs(m.jnt_actfrcrange[joints]), axis=1) - 1e-3)
            if d.time > 1.5:
                positions.append(float(d.qpos[steer]))
                depths.append(min((c.dist for c in d.contact), default=0))
        rows.append(
            {
                "target_rad": float(angles[idx]),
                "actual_rad": float(d.qpos[steer]),
                "last_window_rmse_rad": float(np.sqrt(np.mean((positions - angles[idx]) ** 2))),
                "ik_error_m": errors[idx],
                "peak_arm_torque_Nm": float(np.max(torques)),
                "arm_torque_saturation_fraction": float(np.mean(saturated)),
                "last_window_deepest_contact_m": float(min(depths)),
                "solver_warnings": int(d.warning.number.sum()),
                "finite": bool(np.isfinite(d.qpos).all()),
            }
        )
    return {
        "scope": "CPU fixed stand, ideal palm connections, no steering motor; not GPU free riding",
        "environment": cfg,
        "results": rows,
        "passed": all(
            r["finite"] and r["solver_warnings"] == 0 and r["last_window_rmse_rad"] < 0.01
            for r in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rl/bicycle_b.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/phase2/controller_limits_v4/stand_audit.json")
    )
    args = parser.parse_args()
    report = audit(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
