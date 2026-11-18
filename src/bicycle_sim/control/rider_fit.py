"""Kinematic fit only: no claim of dynamic balance, grasp stability or collision freedom."""

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from bicycle_sim.control.hand_contact import audit_hands
from bicycle_sim.models.bicycle import build_bicycle, numbers, sub


def combined_tree(config, robot_path, scene_dir):
    root = build_bicycle(config)
    robot = ET.parse(robot_path).getroot()
    import os

    root.find("compiler").set("meshdir", os.path.relpath(robot_path.parent / "assets", scene_dir))
    for child in robot.find("default"):
        root.find("default").append(copy.deepcopy(child))
    root.append(copy.deepcopy(robot.find("asset")))
    pelvis = copy.deepcopy(robot.find("./worldbody/body[@name='pelvis']"))
    seat = config["geometry"]["saddle_top"]
    pelvis.set(
        "pos",
        numbers(
            [
                seat[0],
                0,
                seat[2] + config.get("rider_reference", {}).get("pelvis_height_above_saddle", 0.17),
            ]
        ),
    )
    sub(pelvis, "site", name="rider_seat_reference", pos="0 0 -0.17", size="0.008")
    for side in ("left", "right"):
        ankle = pelvis.find(f".//body[@name='{side}_ankle_roll_link']")
        sub(ankle, "site", name=f"{side}_sole_reference", pos="0.035 0 -0.035", size="0.008")
        wrist = pelvis.find(f".//body[@name='{side}_wrist_yaw_link']")
        sub(
            wrist,
            "site",
            name=f"{side}_grasp_reference",
            pos=numbers(
                [
                    config.get("rider_reference", {})
                    .get("hand_initialization", {})
                    .get("palm_site_x_m", 0.10),
                    (-1 if side == "left" else 1)
                    * config.get("rider_reference", {})
                    .get("hand_initialization", {})
                    .get("palm_site_y_abs_m", 0.01732),
                    0,
                ]
            ),
            size="0.008",
        )
    root.find("worldbody").append(pelvis)
    root.append(copy.deepcopy(robot.find("actuator")))
    for sensor in robot.find("sensor"):
        root.find("sensor").append(copy.deepcopy(sensor))
    return root


def fit(
    config_path=Path("configs/bicycle_v0.json"),
    output=Path("output/phase1"),
    scene_name="rider_fit_diagnostic.xml",
):
    config = json.loads(config_path.read_text())
    scene_dir = Path("assets/scenes")
    scene_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    root = combined_tree(config, Path("assets/g1_dex3/g1_with_hands.xml"), scene_dir)
    path = scene_dir / scene_name
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    m = mujoco.MjModel.from_xml_path(str(path.resolve()))
    d = mujoco.MjData(m)

    def joint(name):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

    def site(name):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)

    legs = [
        joint(f"{side}_{part}_joint")
        for side in ("left", "right")
        for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    arms = [
        joint(f"{side}_{part}_joint")
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
    d.qpos[m.jnt_qposadr[joint("waist_pitch_joint")]] = config.get("rider_reference", {}).get(
        "waist_pitch_rad", 0
    )
    leg_q = m.jnt_qposadr[legs]
    arm_q = m.jnt_qposadr[arms]
    d.qpos[leg_q] = [-0.8, 0, 0, 1.6, -0.8, 0] * 2
    d.qpos[arm_q] = [0, 0.2, 0, 1, 0, 0, 0, 0, -0.2, 0, 1, 0, 0, 0]
    arm_seed = d.qpos[arm_q].copy()
    crank = m.jnt_qposadr[joint("crank_spin")]
    rear = m.jnt_qposadr[joint("rear_wheel_spin")]
    pedal_q = [m.jnt_qposadr[joint(f"{side}_pedal_spin")] for side in ("left", "right")]
    sole_sites = [site(f"{side}_sole_reference") for side in ("left", "right")]
    grasp_sites = [site(f"{side}_grasp_reference") for side in ("left", "right")]
    target_feet = [site(f"{side}_pedal_contact") for side in ("left", "right")]
    target_hands = [site(f"{side}_grip") for side in ("left", "right")]

    def kinematics():
        mujoco.mj_kinematics(m, d)

    def solve(ids, qids, residual):
        lower, upper = m.jnt_range[ids].T
        result = least_squares(
            residual,
            np.clip(d.qpos[qids], lower + 1e-4, upper - 1e-4),
            bounds=(lower + 1e-4, upper - 1e-4),
            max_nfev=150,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        d.qpos[qids] = result.x
        kinematics()
        return result

    arm_cache = {}
    states = []
    records = []
    # Include the closing phase to check cyclic reference continuity.
    for phase in np.linspace(0, 2 * np.pi, 25):
        d.qpos[crank], d.qpos[rear] = phase, 2 * phase
        d.qpos[pedal_q] = -phase
        kinematics()
        feet_targets = d.site_xpos[target_feet].copy()

        def leg_residual(x, feet_targets=feet_targets):
            d.qpos[leg_q] = x
            kinematics()
            position = (d.site_xpos[sole_sites] - feet_targets).ravel()
            orientation = np.concatenate(
                [Rotation.from_matrix(d.site_xmat[s].reshape(3, 3)).as_rotvec() for s in sole_sites]
            )
            return np.r_[position, 0.1 * orientation]

        solve(legs, leg_q, leg_residual)
        hand_records = []
        for steering in (
            -config["geometry"]["steer_limit_rad"],
            0,
            config["geometry"]["steer_limit_rad"],
        ):
            d.qpos[m.jnt_qposadr[joint("steer")]] = steering
            kinematics()
            targets = d.site_xpos[target_hands].copy()
            hand_cfg = config.get("rider_reference", {}).get("hand_initialization", {})
            targets[:, 2] += config["geometry"]["grip_radius"] + hand_cfg.get("clearance_m", 0.0005)
            # Left palm faces local -Y; right palm faces local +Y. Fingers point +X.
            target_rotations = [
                d.site_xmat[t].reshape(3, 3)
                @ Rotation.from_euler("x", sign * np.pi / 2).as_matrix()
                for t, sign in zip(target_hands, (1, -1), strict=True)
            ]

            d.qpos[arm_q] = arm_seed

            def arm_residual(
                x, targets=targets, target_rotations=target_rotations, hand_cfg=hand_cfg
            ):
                d.qpos[arm_q] = x
                kinematics()
                orientation = np.concatenate(
                    [
                        Rotation.from_matrix(target.T @ d.site_xmat[s].reshape(3, 3)).as_rotvec()
                        for s, target in zip(grasp_sites, target_rotations, strict=True)
                    ]
                )
                return np.r_[
                    (d.site_xpos[grasp_sites] - targets).ravel(),
                    hand_cfg.get("orientation_weight_m", 0.15) * orientation,
                    0.0001 * (x - arm_seed),
                ]

            if steering not in arm_cache:
                seeds = [
                    arm_seed,
                    np.array([-0.6, 0.7, 0.5, 1.5, 1.5, 0, 0, -0.6, -0.7, -0.5, 1.5, -1.5, 0, 0]),
                    np.array([-0.9, 0.3, 1.2, 0.8, 1.5, 0, 0, -0.9, -0.3, -1.2, 0.8, -1.5, 0, 0]),
                ]
                candidates = []
                for seed in seeds:
                    d.qpos[arm_q] = seed
                    result = solve(arms, arm_q, arm_residual)
                    candidates.append((float(np.linalg.norm(result.fun)), result.x.copy()))
                arm_cache[steering] = min(candidates, key=lambda candidate: candidate[0])[1]
            d.qpos[arm_q] = arm_cache[steering]
            kinematics()
            hand_records.append(
                {
                    "steer_rad": steering,
                    "palm_down_orientation_errors_deg": [
                        float(
                            np.rad2deg(
                                np.linalg.norm(
                                    Rotation.from_matrix(
                                        t.T @ d.site_xmat[s].reshape(3, 3)
                                    ).as_rotvec()
                                )
                            )
                        )
                        for s, t in zip(grasp_sites, target_rotations, strict=True)
                    ],
                    "position_errors_m": np.linalg.norm(
                        d.site_xpos[grasp_sites] - targets, axis=1
                    ).tolist(),
                }
            )
            if steering == 0:
                snapshot = d.qpos.copy()
        records.append(
            {
                "phase_rad": float(phase),
                "foot_errors_m": np.linalg.norm(
                    d.site_xpos[sole_sites] - feet_targets, axis=1
                ).tolist(),
                "hand_position_only": hand_records,
                "knee_angles_deg": np.rad2deg(
                    [
                        d.qpos[m.jnt_qposadr[joint(f"{side}_knee_joint")]]
                        for side in ("left", "right")
                    ]
                ).tolist(),
                "elbow_angles_deg": np.rad2deg(
                    [
                        snapshot[m.jnt_qposadr[joint(f"{side}_elbow_joint")]]
                        for side in ("left", "right")
                    ]
                ).tolist(),
            }
        )
        d.qpos[:] = snapshot
        mujoco.mj_forward(m, d)
        records[-1]["deep_penetrations"] = [
            {
                "body1": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]),
                "body2": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]),
                "distance_m": float(c.dist),
            }
            for c in d.contact
            if c.dist < -0.005
        ]
        records[-1]["neutral_hand_contact"] = audit_hands(m, d)
        states.append(snapshot)
    np.savez(
        output / "rider_kinematic_reference.npz",
        qpos=np.array(states),
        phases=np.linspace(0, 2 * np.pi, 25),
    )
    # A saved pose keyframe is not a constraint or an actuator reference.
    keys = sub(root, "keyframe")
    # Match the position-servo reference to the initial joints, not all-zero controls.
    initial_ctrl = states[0][m.jnt_qposadr[m.actuator_trnid[:, 0]]]
    sub(keys, "key", name="mounted_kinematic", qpos=numbers(states[0]), ctrl=numbers(initial_ctrl))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    report = {
        "scope": "Kinematic reachability and initial neutral palm-down near-touch; loaded grip and dynamics NOT validated",
        "scene": str(path),
        "config": str(config_path),
        "records": records,
        "acceptance_passed": False,
        "initial_hand_acceptance_passed": all(
            h["passed"] for h in records[0]["neutral_hand_contact"].values()
        ),
        "initial_hands": records[0]["neutral_hand_contact"],
        "notes": [
            "Neutral initial hands audited geometrically; nonzero steering, loaded contact and dynamic control remain unvalidated."
        ],
        "max_foot_position_error_m": max(max(x["foot_errors_m"]) for x in records),
        "max_hand_position_error_m": max(
            max(h["position_errors_m"]) for x in records for h in x["hand_position_only"]
        ),
    }
    (output / "rider_fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    fit()
