"""Generate a contact-supported, no-weld initial state on a fixed-bike bench."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np
import torch

from bicycle_sim.cli.probe_hand_grasp import collect_finger_contacts
from bicycle_sim.control.seated_leg_reference import leg_pose_ik, refit_leg_cycle
from bicycle_sim.rl.cuda_context import shared_cuda_stream
from bicycle_sim.rl.rider_model import build_rider_model, model_fingerprint


def fixed_bench(m, q, ctrl, xml):
    root = ET.fromstring(xml)
    root.find("./worldbody/body[@name='bike_frame']").remove(
        root.find(".//joint[@name='bike_root']")
    )
    bench = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    state = bench.qpos0.copy()
    for j in range(bench.njnt):
        old = m.joint(bench.joint(j).name)
        n = 7 if bench.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        state[bench.jnt_qposadr[j] : bench.jnt_qposadr[j] + n] = q[
            old.qposadr[0] : old.qposadr[0] + n
        ]
    controls = np.array([ctrl[m.actuator(bench.actuator(a).name).id] for a in range(bench.nu)])
    return bench, state, controls


def sitting_contacts(m, d, force_override=None, *, with_details=False):
    seat, pelvis = m.geom("saddle").id, m.body("pelvis").id
    pedals = [m.geom(f"{s}_pedal_platform").id for s in ("left", "right")]
    feet = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
    load = np.zeros(3)
    depth = 0.0
    details = {
        name: {"penetration_m": 0.0, "deepest_contact": None}
        for name in ("seat", "left_foot", "right_foot", "hands")
    }

    def record(group, g1, g2, distance, force, position):
        penetration = max(0.0, -float(distance))
        if penetration > details[group]["penetration_m"]:
            details[group] = {
                "penetration_m": penetration,
                "deepest_contact": {
                    "geom_ids": [int(g1), int(g2)],
                    "geoms": [m.geom(g).name or f"geom#{g}" for g in (g1, g2)],
                    "bodies": [m.body(m.geom_bodyid[g]).name for g in (g1, g2)],
                    "normal_force_N": float(force),
                    "position_m": np.asarray(position).tolist(),
                },
            }

    for i, ct in enumerate(d.contact):
        g1, g2 = map(int, ct.geom)
        if min(g1, g2) < 0:
            continue
        f = np.zeros(6)
        if force_override is None:
            mujoco.mj_contactForce(m, d, i, f)
        else:
            f[:] = force_override[i]
        if (g1 == seat and m.geom_bodyid[g2] == pelvis) or (
            g2 == seat and m.geom_bodyid[g1] == pelvis
        ):
            load[0] += max(0, f[0])
            depth = min(depth, float(ct.dist))
            record("seat", g1, g2, ct.dist, f[0], ct.pos)
        for s in range(2):
            if (g1 == pedals[s] and m.geom_bodyid[g2] == feet[s]) or (
                g2 == pedals[s] and m.geom_bodyid[g1] == feet[s]
            ):
                load[s + 1] += max(0, f[0])
                depth = min(depth, float(ct.dist))
                record(("left_foot", "right_foot")[s], g1, g2, ct.dist, f[0], ct.pos)
    hands, hand_depth, raw = collect_finger_contacts(m, d, force_override=force_override)
    for ct in raw:
        if any("_hand_" in m.body(m.geom_bodyid[g]).name for g in ct["geom"]):
            record("hands", *ct["geom"], ct["dist"], ct["force_local"][0], ct["pos"])
    finger_forces = np.array([v["normal_force_N"] for v in hands.values()])
    result = (load, finger_forces, min(depth, hand_depth))
    return (*result, details) if with_details else result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/rl/bicycle_b_whole_body_sac.json"))
    p.add_argument(
        "--reference", type=Path, default=Path("output/phase1/rider_kinematic_reference.npz")
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--output", type=Path, default=Path("output/phase2/whole_body_init_gpu_v1"))
    p.add_argument(
        "--reference-output",
        type=Path,
        help="Accepted GPU pose destination; CPU only saves a diagnostic candidate",
    )
    args = p.parse_args()
    if args.seconds < 1 or args.output.exists():
        p.error("Use seconds >= 1 and a new output directory")
    if args.device != "cpu" and (
        not args.device.startswith("cuda:") or not torch.cuda.is_available()
    ):
        p.error("Requested GPU is unavailable")
    cfg = json.loads(args.config.read_text())
    if not cfg.get("whole_body_control"):
        p.error("Expected a whole-body config")
    target = args.reference_output or Path(cfg["whole_body_reference_path"])
    if args.device == "cpu":
        if args.reference_output is not None:
            p.error("CPU mode produces diagnostic candidates only")
        target = args.output / "cpu_candidate.json"
    if target.exists():
        p.error("Reference destination already exists; use a new destination")
    context = nullcontext() if args.device == "cpu" else shared_cuda_stream(args.device)
    with context:
        generate(args, cfg, target)


def generate(args, cfg, target):
    args.output.mkdir(parents=True)
    source, q0, ctrl, maps = build_rider_model(cfg, args.reference, load_settled=False)
    fingerprint = model_fingerprint(source)
    source_path = args.output / "source_free_scene.xml"
    mujoco.mj_saveLastXML(str(source_path), source)
    m, q, ctrl = fixed_bench(source, q0, ctrl, source_path.read_text())
    if m.neq:
        raise RuntimeError("Initialization must contain no equalities")
    mujoco.mj_saveLastXML(str(args.output / "scene.xml"), m)
    # The old grasp snapshot was generated without these sole boxes. Start
    # with matched pedal/foot pitch, not the overlapping pose in that snapshot.
    initial_phase = float(q[m.joint("crank_spin").qposadr[0]])
    pedal_pitch = cfg.get("pedal_reference_pitch_rad", 0.08)
    for side in ("left", "right"):
        q[m.joint(f"{side}_pedal_spin").qposadr[0]] = -initial_phase + pedal_pitch
    legs = [m.actuator(source.actuator(a).name).id for a in maps["leg_act"]]
    leg_qi = m.jnt_qposadr[m.actuator_trnid[legs, 0]]
    clearance = cfg.get("initial_foot_clearance_m", 0.0002)
    initial_legs, initial_ik = leg_pose_ik(m, q, clearance=clearance, pedal_pitch_rad=pedal_pitch)
    q[leg_qi] = initial_legs
    ctrl[legs] = initial_legs
    d = mujoco.MjData(m)
    d.qpos[:], d.ctrl[:] = q, ctrl
    mujoco.mj_forward(m, d)
    pelvis = m.body("pelvis").id
    pelvis_v = m.joint("floating_base_joint").dofadr[0]
    initial_p = d.xpos[pelvis].copy()
    crank = m.joint("crank_spin")
    initial_phase = float(q[crank.qposadr[0]])
    drive = m.actuator("crank_drive").id
    sim = None
    if args.device != "cpu":
        import mujoco_warp as mjwarp
        import warp as wp
        from mjlab.sim import Simulation, SimulationCfg

        settings = SimulationCfg(nconmax=cfg["nconmax"], njmax=cfg["njmax"])
        from bicycle_sim.rl.warp_capacity import configure_epa_horizon

        configure_epa_horizon(cfg.get("epa_horizon_capacity", 24))
        settings.mujoco.timestep, settings.mujoco.iterations = m.opt.timestep, m.opt.iterations
        sim = Simulation(1, settings, m, device=args.device)
        sim.reset()
        dtype = sim.data.qpos[:].dtype
        sim.data.qpos[0] = torch.as_tensor(q, dtype=dtype, device=args.device)
        sim.data.qvel[0] = 0
        sim.data.ctrl[0] = torch.as_tensor(ctrl, dtype=dtype, device=args.device)
        sim.forward()
        aids = torch.tensor(legs, device=args.device)
        limits = torch.tensor(m.actuator_ctrlrange[legs], dtype=dtype, device=args.device)
    trace, frames = [], []
    for step in range(round(args.seconds / m.opt.timestep) + 1):
        if step % 10 == 0:
            forces = None
            if sim is not None:
                if torch.any(sim.data.overflow[:] != 0):
                    raise FloatingPointError("Warp contact/constraint overflow")
                mjwarp.get_data_into(d, m, sim.wp_data)
                if int(sim.data.nacon[0]) != d.ncon:
                    raise RuntimeError("GPU contact count mismatch")
                forces = np.zeros((d.ncon, 6))
                if d.ncon:
                    ids = wp.array(np.arange(d.ncon, dtype=np.int32), device=args.device)
                    buffer = wp.zeros(d.ncon, dtype=wp.spatial_vector, device=args.device)
                    mjwarp.contact_force(sim.wp_model, sim.wp_data, ids, False, buffer)
                    forces = buffer.numpy()
            if not np.isfinite(d.qpos).all():
                raise FloatingPointError("Nonfinite initialization pose")
            load, fingers, depth, contact_details = sitting_contacts(
                m, d, forces, with_details=True
            )
            # This kinematic controller is initialization-only. RL always uses
            # its learned 12 leg actions around the exported cyclic reference.
            leg_target, ik_report = leg_pose_ik(
                m, d.qpos, clearance=clearance, pedal_pitch_rad=pedal_pitch, strict=False
            )
            R = d.xmat[pelvis].reshape(3, 3)
            trace.append(
                {
                    "time_s": float(d.time),
                    "support_normal_force_N": load.tolist(),
                    "finger_normal_force_N": fingers.tolist(),
                    "penetration_m": -depth,
                    "contact_details": contact_details,
                    "leg_ik": ik_report,
                    "pelvis_shift_m": (d.xpos[pelvis] - initial_p).tolist(),
                    "pelvis_tilt_rad": float(np.arccos(np.clip(R[2, 2], -1, 1))),
                    "crank_speed_rad_s": float(d.qvel[crank.dofadr[0]]),
                    "pelvis_speed_mps": float(np.linalg.norm(d.qvel[pelvis_v : pelvis_v + 3])),
                    "pelvis_angular_speed_rad_s": float(
                        np.linalg.norm(d.qvel[pelvis_v + 3 : pelvis_v + 6])
                    ),
                }
            )
            frames.append((d.qpos.copy(), d.qvel.copy()))
        if step == round(args.seconds / m.opt.timestep):
            break
        if sim is None:
            phase, speed = (
                torch.tensor([d.qpos[crank.qposadr[0]]]),
                torch.tensor([d.qvel[crank.dofadr[0]]]),
            )
            d.ctrl[legs] = leg_target
            d.ctrl[drive] = np.clip(10 * (initial_phase - phase.item()) - 2 * speed.item(), -8, 8)
            mujoco.mj_step(m, d)
        else:
            phase, speed = sim.data.qpos[:, crank.qposadr[0]], sim.data.qvel[:, crank.dofadr[0]]
            sim.data.ctrl[:, aids] = torch.as_tensor(leg_target, dtype=dtype, device=args.device)[
                None
            ].clamp(limits[:, 0], limits[:, 1])
            sim.data.ctrl[:, drive] = (10 * (initial_phase - phase) - 2 * speed).clamp(-8, 8)
            sim.step()
    window = [v for v in trace if v["time_s"] >= args.seconds - 0.51]
    load = np.array([v["support_normal_force_N"] for v in window])
    finger = np.array([v["finger_normal_force_N"] for v in window])
    supported = (load[:, 0] > 10) & (load[:, 1] > 1) & (load[:, 2] > 1)
    both_hands = (finger[:, :3].sum(-1) > 0.5) & (finger[:, 3:].sum(-1) > 0.5)
    checks = {
        "support_fraction": float(supported.mean()),
        "both_hands_fraction": float(both_hands.mean()),
        "max_penetration_m": max(v["penetration_m"] for v in window),
        "max_pelvis_tilt_rad": max(v["pelvis_tilt_rad"] for v in window),
        "max_pelvis_shift_m": max(np.linalg.norm(v["pelvis_shift_m"]) for v in window),
        "max_pelvis_speed_mps": max(v["pelvis_speed_mps"] for v in window),
        "max_pelvis_angular_speed_rad_s": max(v["pelvis_angular_speed_rad_s"] for v in window),
        "max_penetration_by_group_m": {
            name: max(v["contact_details"][name]["penetration_m"] for v in window)
            for name in ("seat", "left_foot", "right_foot", "hands")
        },
        "max_leg_ik_position_error_m": max(v["leg_ik"]["max_position_error_m"] for v in window),
        "max_leg_ik_angle_error_rad": max(v["leg_ik"]["max_angle_error_rad"] for v in window),
    }
    final_supported = bool(supported[-1] and both_hands[-1])
    passed = bool(
        checks["support_fraction"] >= 0.95
        and checks["both_hands_fraction"] >= 0.95
        and checks["max_penetration_m"] < 0.002
        and checks["max_pelvis_tilt_rad"] < 0.08
        and checks["max_pelvis_shift_m"] < 0.02
        and checks["max_pelvis_speed_mps"] < 0.05
        and checks["max_pelvis_angular_speed_rad_s"] < 0.2
        and final_supported
        and checks["max_leg_ik_position_error_m"] < 0.001
        and checks["max_leg_ik_angle_error_rad"] < 0.02
    )
    backend = "native CPU MuJoCo" if sim is None else "mjlab/MuJoCo-Warp"
    reference = {
        "backend": backend,
        "passed": passed,
        "model_fingerprint": fingerprint,
        "phase_reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "config": cfg,
        "diagnostics": checks,
        "source_run": str(args.output),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *sorted(Path("src/bicycle_sim").rglob("*.py")),
                args.config,
                args.reference,
                Path(cfg["grasp_reference_path"]),
                Path("assets/scenes/rider_fit_diagnostic.xml"),
                Path("uv.lock"),
            ]
        },
        "initialization_only_assistance": ["fixed bicycle frame", "temporary crank position hold"],
        "initialization_leg_controller": "IK of actual pelvis and pedal poses, preparation only",
        "initial_leg_ik": initial_ik,
        "joint_qpos": {},
        "control_targets": {},
    }
    for j in range(source.njnt):
        name = source.joint(j).name
        n = 7 if source.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        if name == "bike_root":
            values = q0[source.jnt_qposadr[j] : source.jnt_qposadr[j] + n]
        else:
            a = m.joint(name).qposadr[0]
            values = d.qpos[a : a + n]
        reference["joint_qpos"][name] = values.tolist()
    controls = d.ctrl if sim is None else sim.data.ctrl[0].cpu().numpy()
    reference["control_targets"] = {m.actuator(a).name: float(v) for a, v in enumerate(controls)}
    reference["control_targets"]["crank_drive"] = 0.0
    if passed:
        try:
            table, cycle_check = refit_leg_cycle(
                m, d.qpos, clearance=clearance, pedal_pitch_rad=pedal_pitch
            )
            reference["leg_reference"] = table.tolist()
            reference["leg_reference_fit"] = cycle_check
            checks["leg_reference_fit"] = cycle_check
        except RuntimeError as exc:
            passed = reference["passed"] = False
            checks["leg_reference_fit_error"] = str(exc)
    (args.output / "results.json").write_text(
        json.dumps(
            {"backend": backend, "passed": passed, "checks": checks, "trace": trace}, indent=2
        )
        + "\n"
    )
    np.savez_compressed(
        args.output / "trajectory.npz",
        qpos=np.array([f[0] for f in frames]),
        qvel=np.array([f[1] for f in frames]),
        time_s=[v["time_s"] for v in trace],
    )
    if passed:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(reference, indent=2) + "\n")
    print(
        json.dumps(
            {
                "backend": backend,
                "passed": passed,
                "checks": checks,
                "reference": str(target) if passed else None,
            },
            indent=2,
        )
    )
    if not passed:
        raise RuntimeError(
            "No-weld initialization did not pass contact/pose gates; inspect results"
        )


if __name__ == "__main__":
    main()
