"""Check sampled front assembly clearances against legs and pedal mechanism."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def scan(scene_path, reference_path, steer_limit, samples=9):
    model = mujoco.MjModel.from_xml_path(str(scene_path.resolve()))
    data = mujoco.MjData(model)
    reference = np.load(reference_path)
    front, legs, crank = [], [], []
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if body in ("steering", "front_wheel") or name == "head_tube":
            front.append(g)
        if body.startswith(("left_", "right_")) and any(
            part in body for part in ("hip_", "knee_", "ankle_")
        ):
            legs.append(g)
        if body in ("crank", "left_pedal", "right_pedal"):
            crank.append(g)
    if not front or not legs or not crank:
        raise ValueError("Missing collision geoms in clearance groups")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steer")

    def label(g):
        return {
            "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]),
            "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
            "id": g,
        }

    results = {name: {"minimum_distance_m": 0.05} for name in ("legs", "crank_and_pedals")}
    for phase, qpos in zip(reference["phases"], reference["qpos"], strict=True):
        for angle in np.linspace(-steer_limit, steer_limit, samples):
            data.qpos[:] = qpos
            data.qpos[model.jnt_qposadr[joint]] = angle
            mujoco.mj_forward(model, data)
            for name, group in (("legs", legs), ("crank_and_pedals", crank)):
                for a in front:
                    for b in group:
                        distance = mujoco.mj_geomDistance(model, data, a, b, 0.05, None)
                        if distance < results[name]["minimum_distance_m"]:
                            results[name] = {
                                "minimum_distance_m": float(distance),
                                "front": label(a),
                                "other": label(b),
                                "phase_rad": float(phase),
                                "steer_rad": float(angle),
                            }
    return {
        "scope": "Static collision geometry grid; not dynamic or continuous swept-volume validation",
        "phase_count": len(reference["phases"]),
        "steering_count": samples,
        "steer_limit_rad": steer_limit,
        "required_margin_m": 0.01,
        "distance_query_cutoff_m": 0.05,
        "passed": all(r["minimum_distance_m"] >= 0.01 for r in results.values()),
        "groups": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path, default=Path("assets/scenes/rider_fit_diagnostic.xml")
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("output/phase1/rider_kinematic_reference.npz")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/bicycle_v0.json"))
    parser.add_argument("--output", type=Path, default=Path("output/phase1/front_clearance.json"))
    args = parser.parse_args()
    limit = json.loads(args.config.read_text())["geometry"]["steer_limit_rad"]
    report = scan(args.scene, args.reference, limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
