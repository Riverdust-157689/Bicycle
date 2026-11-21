"""Fixed-bike Dex3 finger closure probe, using contact instead of palm connects."""

import argparse
import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from bicycle_sim.cli.check_rider_steering import build_rig
from bicycle_sim.control.hand_synergy import FingerSynergy, remove_palm_connections
from bicycle_sim.rl.rider_model import ARM_PARTS


def collect_finger_contacts(m, data, finger_order=FingerSynergy.order, force_override=None):
    """Read canonical geom IDs, including data imported from MuJoCo-Warp.

    geom1/geom2 are deprecated independent fields: get_data_into fills geom[2]
    but does not populate those legacy fields.
    """
    bar = {m.geom(f"{s}_grip_surface").id for s in ("left", "right")}
    prefixes = {name: f"{name.split('_')[0]}_hand_{name.split('_')[1]}_" for name in finger_order}
    contacts = {name: {"count": 0, "normal_force_N": 0.0} for name in finger_order}
    if force_override is not None and np.shape(force_override) != (data.ncon, 6):
        raise ValueError("Contact force array must match the imported contact order")
    deepest, raw = 0.0, []
    for i, contact in enumerate(data.contact):
        geom1, geom2 = map(int, contact.geom)
        if geom1 not in bar and geom2 not in bar:
            continue
        other = geom2 if geom1 in bar else geom1
        if other < 0:
            continue
        body = m.body(m.geom_bodyid[other]).name
        if force_override is None:
            force = np.zeros(6)
            mujoco.mj_contactForce(m, data, i, force)
        else:
            force = np.asarray(force_override[i]).copy()
        raw.append(
            {
                "geom": [geom1, geom2],
                "pos": contact.pos.copy(),
                "frame": contact.frame.copy(),
                "dist": float(contact.dist),
                "force_local": force,
            }
        )
        for name in finger_order:
            if body.startswith(prefixes[name]):
                contacts[name]["count"] += 1
                contacts[name]["normal_force_N"] += float(force[0])
                deepest = min(deepest, float(contact.dist))
    return contacts, deepest, raw


def build_contact_rig():
    source, q0, ctrl0 = build_rig()
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="finger-grasp-") as tmp:
        path = Path(tmp) / "scene.xml"
        mujoco.mj_saveLastXML(str(path), source)
        root = ET.parse(path).getroot()
    remove_palm_connections(root)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.nq == source.nq and model.nu == source.nu
    return model, q0, ctrl0


def set_pregrasp_pose(m, q0, ctrl0, backoff):
    """Shift each palm back along its finger direction, preserving orientation."""
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    mujoco.mj_kinematics(m, d)
    sides = ("left", "right")
    sites = [m.site(f"{s}_grasp_reference").id for s in sides]
    targets = d.site_xpos[sites].copy()
    rotations = d.site_xmat[sites].copy()
    for i, side in enumerate(sides):
        R = d.xmat[m.body(f"{side}_wrist_yaw_link").id].reshape(3, 3)
        targets[i] -= backoff * R[:, 0]
    joints = [m.joint(f"{s}_{part}_joint").id for s in sides for part in ARM_PARTS]
    qi = m.jnt_qposadr[joints]
    nominal = q0[qi].copy()

    def residual(q):
        d.qpos[qi] = q
        mujoco.mj_kinematics(m, d)
        return np.r_[
            (d.site_xpos[sites] - targets).ravel(),
            0.04 * (d.site_xmat[sites] - rotations).ravel(),
            0.001 * (q - nominal),
        ]

    lo = np.maximum(m.jnt_range[joints, 0] + 1e-6, nominal - 0.2)
    hi = np.minimum(m.jnt_range[joints, 1] - 1e-6, nominal + 0.2)
    fit = least_squares(residual, nominal, bounds=(lo, hi), max_nfev=150)
    q0[qi] = fit.x
    acts = [m.actuator(m.joint(j).name).id for j in joints]
    ctrl0[acts] = fit.x
    return float(np.linalg.norm(residual(fit.x)[:6]))


def steering_arm_reference(m, q0, angle):
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    mujoco.mj_kinematics(m, d)
    sites = [m.site(f"{s}_grasp_reference").id for s in ("left", "right")]
    steering = m.body("steering").id
    R = d.xmat[steering].reshape(3, 3).copy()
    local = (d.site_xpos[sites] - d.xpos[steering]) @ R
    local_rot = np.array([R.T @ x.reshape(3, 3) for x in d.site_xmat[sites]])
    d.qpos[m.joint("steer").qposadr[0]] = angle
    mujoco.mj_kinematics(m, d)
    R = d.xmat[steering].reshape(3, 3).copy()
    targets = local @ R.T + d.xpos[steering]
    rotations = np.array([R @ x for x in local_rot]).reshape(2, 9)
    joints = [m.joint(f"{s}_{part}_joint").id for s in ("left", "right") for part in ARM_PARTS]
    qi = m.jnt_qposadr[joints]
    nominal = q0[qi].copy()

    def residual(q):
        d.qpos[qi] = q
        mujoco.mj_kinematics(m, d)
        return np.r_[
            (d.site_xpos[sites] - targets).ravel(),
            0.04 * (d.site_xmat[sites] - rotations).ravel(),
            0.001 * (q - nominal),
        ]

    fit = least_squares(
        residual,
        nominal,
        bounds=(
            np.maximum(m.jnt_range[joints, 0] + 1e-6, nominal - 0.5),
            np.minimum(m.jnt_range[joints, 1] - 1e-6, nominal + 0.5),
        ),
        max_nfev=150,
    )
    return np.array([m.actuator(m.joint(j).name).id for j in joints]), fit.x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("output/phase2/hand_contact_probe"))
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--device", default="cpu", help="cpu or a visible CUDA device")
    p.add_argument("--closure", type=float, nargs=6, default=[0.9] * 6)
    p.add_argument(
        "--steer-test",
        action="store_true",
        help="After closing, turn +0.1 then -0.1 rad via the arms",
    )
    p.add_argument(
        "--hand-backoff",
        type=float,
        default=0.005,
        help="Palm shift along negative finger direction, metres",
    )
    args = p.parse_args()
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        p.error("seconds must be finite and positive")
    if args.steer_test and args.seconds < 12:
        p.error("steer-test needs at least 12 seconds")
    if not np.isfinite(args.hand_backoff) or abs(args.hand_backoff) > 0.03:
        p.error("hand-backoff must be finite and within 3 cm")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Use a new output directory")
    from contextlib import nullcontext

    from bicycle_sim.rl.cuda_context import shared_cuda_stream

    if args.device != "cpu":
        import torch

        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            p.error("CUDA device requested but unavailable; no fallback")
    with nullcontext() if args.device == "cpu" else shared_cuda_stream(args.device):
        run_probe(args)


def run_probe(args):
    m, q0, ctrl0 = build_contact_rig()
    pose_error = set_pregrasp_pose(m, q0, ctrl0, args.hand_backoff)
    initial = mujoco.MjData(m)
    initial.qpos[:] = q0
    mujoco.mj_forward(m, initial)
    from bicycle_sim.control.hand_contact import audit_hands

    initial_hands = audit_hands(m, initial)
    for hand in initial_hands.values():
        hand["legacy_light_touch_passed"] = hand.pop("passed")
    initial_no_penetration = all(
        hand["minimum_hand_bar_distance_m"] >= -1e-6 for hand in initial_hands.values()
    )
    controller = FingerSynergy(m, q0)
    controller.targets(args.closure)
    arm_act, positive = steering_arm_reference(m, q0, 0.1)
    _, negative = steering_arm_reference(m, q0, -0.1)
    arm_neutral = ctrl0[arm_act].copy()
    d = mujoco.MjData(m)
    d.qpos[:], d.ctrl[:] = q0, ctrl0
    sim = None
    if args.device != "cpu":
        import mujoco_warp as mjwarp
        import torch
        import warp as wp
        from mjlab.sim import Simulation, SimulationCfg

        sim_cfg = SimulationCfg(nconmax=128, njmax=1024)
        sim_cfg.mujoco.timestep, sim_cfg.mujoco.iterations = m.opt.timestep, m.opt.iterations
        sim = Simulation(1, sim_cfg, m, device=args.device)
        sim.reset()
        sim.data.qpos[0] = torch.as_tensor(q0, device=args.device, dtype=sim.data.qpos[:].dtype)
        sim.data.qvel[0] = 0
        sim.data.ctrl[0] = torch.as_tensor(ctrl0, device=args.device, dtype=sim.data.ctrl[:].dtype)
        sim.forward()
    args.output.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(args.output / "scene.xml"), m)
    saved = mujoco.MjModel.from_xml_path(str((args.output / "scene.xml").resolve()))
    np.testing.assert_allclose(saved.actuator_gainprm, m.actuator_gainprm)
    np.savez_compressed(args.output / "initial_state.npz", qpos=q0, ctrl=ctrl0)
    history = []
    positions, times = [], []
    raw_contacts, raw_contact_samples = [], []
    for step in range(round(args.seconds / m.opt.timestep)):
        time = step * m.opt.timestep
        steer_reference = 0.0
        if args.steer_test:
            if time >= 8:
                blend = np.clip((time - 8) / 2, 0, 1)
                d.ctrl[arm_act] = positive * (1 - blend) + negative * blend
                steer_reference = 0.1 - 0.2 * blend
            elif time >= 4:
                blend = np.clip((time - 4) / 2, 0, 1)
                d.ctrl[arm_act] = arm_neutral * (1 - blend) + positive * blend
                steer_reference = 0.1 * blend
        controller.apply(d, args.closure, m.opt.timestep)
        if sim is None:
            mujoco.mj_step(m, d)
        else:
            sim.data.ctrl[0] = torch.as_tensor(
                d.ctrl, device=args.device, dtype=sim.data.ctrl[:].dtype
            )
            sim.step()
        if step % 25 != 0:
            continue
        force_override = None
        if sim is not None:
            if torch.any(sim.data.overflow[:] != 0):
                raise FloatingPointError("GPU contact/constraint capacity overflow")
            # Copy the GPU solver's actual contacts and forces; do not rebuild
            # contacts with native mj_forward when reporting the GPU result.
            mjwarp.get_data_into(d, m, sim.wp_data)
            if int(sim.data.nacon[0]) != d.ncon:
                raise RuntimeError("Single-world contact download is inconsistent")
            force_override = np.empty((d.ncon, 6))
            if d.ncon:
                ids = wp.array(np.arange(d.ncon, dtype=np.int32), device=args.device)
                forces = wp.zeros(d.ncon, dtype=wp.spatial_vector, device=args.device)
                # Extract the actual Warp solver forces directly, avoiding
                # dependence on CPU efc reindexing for the force measurement.
                mjwarp.contact_force(sim.wp_model, sim.wp_data, ids, False, forces)
                force_override = forces.numpy()
        if not np.isfinite(d.qpos).all():
            raise FloatingPointError("Nonfinite grasp state")
        contacts, deepest, raw = collect_finger_contacts(m, d, controller.order, force_override)
        raw_contacts.extend(raw)
        raw_contact_samples.extend([len(history)] * len(raw))
        history.append(
            {
                "time_s": float(d.time),
                "steer_reference_rad": float(steer_reference),
                "actual_steer_rad": float(d.qpos[m.joint("steer").qposadr[0]]),
                "total_contacts": d.ncon,
                "bar_contacts": len(raw),
                "finger_contacts": contacts,
                "deepest_finger_bar_distance_m": deepest,
            }
        )
        positions.append(d.qpos.copy())
        times.append(float(d.time))
    report = {
        "scope": f"{args.device} fixed bicycle, seat weld, arm pose targets, no palm connections; no riding validation",
        "backend": "native MuJoCo" if sim is None else "mjlab/MuJoCo-Warp",
        "contact_geom_source": "canonical contact.geom (not deprecated geom1/geom2)",
        "contact_force_source": "mj_contactForce"
        if sim is None
        else "direct mujoco_warp.contact_force",
        "controller": "one input per finger, software joint synergy; not mechanical underactuation",
        "finger_order": controller.order,
        "requested_closure": args.closure,
        "closed_joint_targets_rad": controller.closed.tolist(),
        "joint_order": [m.joint(m.actuator_trnid[a, 0]).name for a in controller.aids],
        "closure_rate_per_s": controller.rate,
        "seconds": args.seconds,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).parents[1] / "control/hand_synergy.py",
            )
        },
        "steer_test": args.steer_test,
        "hand_backoff_m": args.hand_backoff,
        "pregrasp_position_residual_m": pose_error,
        "initial_hand_geometry": initial_hands,
        "initial_no_penetration": initial_no_penetration,
        "kp_Nm_per_rad": 5.0,
        "kv_Nm_s_per_rad": 0.15,
        "warnings": int(d.warning.number.sum()) if sim is None else None,
        "finite": bool(np.isfinite(d.qpos).all()),
        "history": history,
    }
    windows = (
        ((3, 4), (6.5, 7.9), (10.5, 11.9))
        if args.steer_test
        else ((max(0, args.seconds - 1), args.seconds),)
    )
    report["settled_windows"] = []
    for start, end in windows:
        samples = [r for r in history if start < r["time_s"] < end]
        if not samples:
            continue
        report["settled_windows"].append(
            {
                "start_s": start,
                "end_s": end,
                "steering_rmse_rad": float(
                    np.sqrt(
                        np.mean(
                            [
                                (r["actual_steer_rad"] - r["steer_reference_rad"]) ** 2
                                for r in samples
                            ]
                        )
                    )
                ),
                "all_six_fingers_loaded_fraction": float(
                    np.mean(
                        [
                            all(f["normal_force_N"] > 0.5 for f in r["finger_contacts"].values())
                            for r in samples
                        ]
                    )
                ),
                "deepest_finger_bar_distance_m": min(
                    r["deepest_finger_bar_distance_m"] for r in samples
                ),
            }
        )
    report["contact_transfer_check_passed"] = bool(
        report["finite"]
        and initial_no_penetration
        and not report["warnings"]
        and report["settled_windows"]
        and all(
            w["all_six_fingers_loaded_fraction"] == 1 and w["steering_rmse_rad"] < 0.01
            for w in report["settled_windows"]
        )
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "trajectory.npz",
        qpos=np.asarray(positions),
        time_s=np.asarray(times),
        raw_contact_sample=np.asarray(raw_contact_samples, dtype=int),
        **{
            f"raw_contact_{key}": np.asarray([r[key] for r in raw_contacts])
            for key in ("geom", "pos", "frame", "dist", "force_local")
        },
    )
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
