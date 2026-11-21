"""Compare native and recorded/GPU dynamics from identical rider states and commands.

This diagnostic never trains. Ablations isolate contacts and ideal connections;
none changes the production model or the training configuration.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from bicycle_sim.models.tire import set_ellipsoid_tire_geoms
from bicycle_sim.rl.rider_model import build_rider_model

FIELDS = ("qacc", "qfrc_bias", "qfrc_passive", "qfrc_actuator", "qfrc_constraint", "actuator_force")
VARIANTS = ("baseline", "wheel_contacts_only", "no_contacts", "no_equalities")
TIRE_VARIANTS = ("ellipsoid_tires", "ellipsoid_wheel_contacts_only")


def ground_force_acceleration(m, data, headings):
    """Horizontal COM acceleration implied by all ground-contact forces."""
    force = np.zeros((len(headings), 3))
    ground = m.geom("ground").id
    for world, geom, wrench in zip(
        data["contact_world"], data["contact_geom"], data["contact_force_world"], strict=True
    ):
        if ground in geom:
            force[int(world)] += wrench[:3] * (1 if geom[0] == ground else -1)
    return np.sum(force * headings, axis=1) / mujoco.mj_getTotalmass(m)


def com_velocity(m, q, v):
    """Whole-system COM velocity, independent of constraint/force evaluation."""
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = q, v
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)
    mujoco.mj_subtreeVel(m, d)
    return d.subtree_linvel[0].copy()


def leg_control(m, maps, cfg, q, v):
    """Same phase and reference velocity feedforward used during training."""
    cq, cv = m.joint("crank_spin").qposadr[0], m.joint("crank_spin").dofadr[0]
    coordinate = (q[cq] % (2 * np.pi)) * 24 / (2 * np.pi)
    k = min(23, int(coordinate))
    table = maps["leg_table"]
    target = table[k] * (1 - (coordinate - k)) + table[k + 1] * (coordinate - k)
    acts = maps["leg_act"]
    if cfg.get("leg_velocity_feedforward", False):
        kvkp = -m.actuator_biasprm[acts, 2] / m.actuator_gainprm[acts, 0]
        target += kvkp * (table[k + 1] - table[k]) * (24 / (2 * np.pi)) * v[cv]
    return np.clip(target, m.actuator_ctrlrange[acts, 0], m.actuator_ctrlrange[acts, 1])


def variant_model(scene, variant):
    m = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    if variant in TIRE_VARIANTS:
        # Recompile XML so bounds, collision types and mesh transforms remain
        # consistent. Explicit wheel inertias must survive this geometry test.
        root = ET.parse(scene).getroot()
        compiler = root.find("compiler")
        if compiler is not None and compiler.get("meshdir"):
            compiler.set("meshdir", str((scene.parent / compiler.get("meshdir")).resolve()))
        set_ellipsoid_tire_geoms(root, m)
        changed = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        np.testing.assert_allclose(changed.body_mass, m.body_mass, rtol=0, atol=1e-10)
        np.testing.assert_allclose(changed.body_inertia, m.body_inertia, rtol=0, atol=1e-10)
        m = changed
    if variant in ("wheel_contacts_only", "ellipsoid_wheel_contacts_only"):
        keep = {m.geom(n).id for n in ("ground", "rear_wheel_tire", "front_wheel_tire")}
        for i in range(m.ngeom):
            if i not in keep:
                m.geom_contype[i] = m.geom_conaffinity[i] = 0
    elif variant == "no_contacts":
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    elif variant == "no_equalities":
        m.eq_active0[:] = False
    elif variant not in ("baseline", "ellipsoid_tires"):
        raise ValueError(variant)
    return m


def select_samples(recording, times):
    with np.load(recording / "trajectory.npz") as raw:
        track = {k: raw[k] for k in raw.files}
    first = np.flatnonzero(track["episode_id"] == 0)
    # A next frame is needed to recover the applied arm target and recorded acceleration.
    usable = first[:-1]
    ids = np.array([usable[np.argmin(abs(track["time_s"][usable] - t))] for t in times])
    if len(set(ids)) != len(ids):
        raise ValueError("Requested sample times map to duplicate frames; use earlier/fewer times")
    if "arm_executed_target_rad" not in track:
        raise ValueError("Use the new probe recording containing executed arm targets")
    return track, ids


def native_samples(m, maps, cfg, ctrl0, track, ids):
    result = {key: [] for key in FIELDS}
    for key in (
        "ctrl",
        "qvel_after",
        "recorded_ax",
        "native_ax",
        "rotation",
        "ncon",
        "warnings",
        "com_velocity",
        "horizontal_heading",
        "native_com_ax",
        "recorded_com_ax",
    ):
        result[key] = []
    contact_records = {
        key: []
        for key in (
            "contact_world",
            "contact_geom",
            "contact_dist",
            "contact_pos",
            "contact_force_world",
        )
    }
    for world, i in enumerate(ids):
        d = mujoco.MjData(m)
        d.qpos[:], d.qvel[:], d.ctrl[:] = track["qpos"][i], track["qvel"][i], ctrl0
        d.ctrl[maps["arm_act"]] = track["arm_executed_target_rad"][i + 1]
        d.ctrl[maps["leg_act"]] = leg_control(m, maps, cfg, d.qpos, d.qvel)
        mujoco.mj_forward(m, d)
        R = d.xmat[m.body("bike_frame").id].reshape(3, 3).copy()
        heading = R[:, 0].copy()
        heading[2] = 0
        heading /= np.linalg.norm(heading)
        com_start = com_velocity(m, d.qpos, d.qvel)
        result["com_velocity"].append(com_start)
        result["horizontal_heading"].append(heading)
        result["rotation"].append(R)
        result["ctrl"].append(d.ctrl.copy())
        result["ncon"].append(d.ncon)
        for key in FIELDS:
            result[key].append(getattr(d, key).copy())
        for cidx, c in enumerate(d.contact):
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, cidx, force)
            frame = c.frame.reshape(3, 3)
            contact_records["contact_world"].append(world)
            contact_records["contact_geom"].append([c.geom1, c.geom2])
            contact_records["contact_dist"].append(c.dist)
            contact_records["contact_pos"].append(c.pos.copy())
            contact_records["contact_force_world"].append(
                np.r_[frame.T @ force[:3], frame.T @ force[3:]]
            )
        dt = float(track["time_s"][i + 1] - track["time_s"][i])
        steps = round(dt / m.opt.timestep)
        if not np.isclose(steps * m.opt.timestep, dt):
            raise ValueError("Recording/control timestep mismatch")
        for _ in range(steps):
            d.ctrl[maps["leg_act"]] = leg_control(m, maps, cfg, d.qpos, d.qvel)
            mujoco.mj_step(m, d)
        result["qvel_after"].append(d.qvel.copy())
        result["native_com_ax"].append(heading @ (com_velocity(m, d.qpos, d.qvel) - com_start) / dt)
        result["recorded_com_ax"].append(
            heading @ (com_velocity(m, track["qpos"][i + 1], track["qvel"][i + 1]) - com_start) / dt
        )
        result["native_ax"].append((R.T @ (d.qvel[:3] - track["qvel"][i, :3]))[0] / dt)
        result["recorded_ax"].append(
            (R.T @ (track["qvel"][i + 1, :3] - track["qvel"][i, :3]))[0] / dt
        )
        result["warnings"].append(int(d.warning.number.sum()))
    return {key: np.asarray(value) for key, value in (result | contact_records).items()}


def leg_control_tensors(m, maps, like):
    """Match model constants to the actual simulation control buffer's dtype."""
    import torch

    leg = maps["leg_act"]
    kwargs = {"device": like.device, "dtype": like.dtype}
    return (
        torch.as_tensor(maps["leg_table"], **kwargs),
        torch.as_tensor(-m.actuator_biasprm[leg, 2] / m.actuator_gainprm[leg, 0], **kwargs),
        torch.as_tensor(m.actuator_ctrlrange[leg], **kwargs),
    )


def gpu_samples(m, maps, cfg, track, ids, native, device):
    import mujoco_warp as mjwarp
    import torch
    import warp as wp
    from mjlab.sim import Simulation, SimulationCfg

    from bicycle_sim.rl.rider_gpu_env import leg_servo_reference, phase_reference

    scfg = SimulationCfg(nconmax=cfg["nconmax"], njmax=cfg["njmax"])
    # Preserve the saved options instead of silently introducing new solver settings.
    for name in (
        "timestep",
        "impratio",
        "iterations",
        "tolerance",
        "ls_iterations",
        "ls_tolerance",
        "ccd_iterations",
    ):
        setattr(scfg.mujoco, name, getattr(m.opt, name))
    scfg.mujoco.gravity = tuple(m.opt.gravity)
    for name, enum, prefix in (
        ("integrator", mujoco.mjtIntegrator, "mjINT_"),
        ("cone", mujoco.mjtCone, "mjCONE_"),
        ("solver", mujoco.mjtSolver, "mjSOL_"),
        ("jacobian", mujoco.mjtJacobian, "mjJAC_"),
    ):
        setattr(scfg.mujoco, name, enum(getattr(m.opt, name)).name.removeprefix(prefix).lower())
    sim = Simulation(len(ids), scfg, m, device=device)
    sim.reset()
    sim.data.qpos[:] = torch.as_tensor(
        track["qpos"][ids], device=device, dtype=sim.data.qpos[:].dtype
    )
    sim.data.qvel[:] = torch.as_tensor(
        track["qvel"][ids], device=device, dtype=sim.data.qvel[:].dtype
    )
    sim.data.ctrl[:] = torch.as_tensor(native["ctrl"], device=device, dtype=sim.data.ctrl[:].dtype)
    sim.data.qacc_warmstart[:] = 0
    sim.forward()
    result = {key: getattr(sim.data, key)[:].cpu().numpy().copy() for key in FIELDS}
    result["nefc"] = sim.data.nefc[:].cpu().numpy().copy()
    contacts = sim.data.contact
    count = int(sim.data.nacon[0])
    result["contact_world"] = contacts.worldid[:count].cpu().numpy().copy()
    result["contact_geom"] = contacts.geom[:count].cpu().numpy().copy()
    result["contact_dist"] = contacts.dist[:count].cpu().numpy().copy()
    result["contact_pos"] = contacts.pos[:count].cpu().numpy().copy()
    if count:
        contact_ids = wp.array(np.arange(count, dtype=np.int32), device=device)
        forces = wp.zeros(count, dtype=wp.spatial_vector, device=device)
        mjwarp.contact_force(sim.wp_model, sim.wp_data, contact_ids, True, forces)
        result["contact_force_world"] = forces.numpy()
    else:
        result["contact_force_world"] = np.empty((0, 6))
    result["ncon"] = np.bincount(result["contact_world"], minlength=len(ids))
    table, kvkp, limits = leg_control_tensors(m, maps, sim.data.ctrl[:])
    leg = maps["leg_act"]
    cq, cv = m.joint("crank_spin").qposadr[0], m.joint("crank_spin").dofadr[0]
    dt = track["time_s"][ids + 1] - track["time_s"][ids]
    if not np.allclose(dt, dt[0]):
        raise ValueError("Samples have different control durations")
    for _ in range(round(float(dt[0]) / m.opt.timestep)):
        phase, speed = sim.data.qpos[:, cq], sim.data.qvel[:, cv]
        ref = (
            leg_servo_reference(phase, speed, table, kvkp)
            if cfg.get("leg_velocity_feedforward", False)
            else phase_reference(phase, table)
        )
        sim.data.ctrl[:, leg] = ref.clamp(limits[:, 0], limits[:, 1])
        sim.step()
    result["qvel_after"] = sim.data.qvel[:].cpu().numpy().copy()
    result["qpos_after"] = sim.data.qpos[:].cpu().numpy().copy()
    delta_v = result["qvel_after"][:, :3] - track["qvel"][ids, :3]
    result["gpu_ax"] = (
        np.einsum("nij,nj->ni", native["rotation"].transpose(0, 2, 1), delta_v)[:, 0] / dt
    )
    result["gpu_com_ax"] = np.array(
        [
            native["horizontal_heading"][j]
            @ (com_velocity(m, q, v) - native["com_velocity"][j])
            / dt[j]
            for j, (q, v) in enumerate(zip(result["qpos_after"], result["qvel_after"], strict=True))
        ]
    )
    if not np.isfinite(result["qvel_after"]).all() or torch.any(sim.data.overflow[:] != 0):
        raise FloatingPointError("GPU diagnostic has nonfinite state or capacity overflow")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "recording", type=Path, nargs="?", default=Path("output/phase2/b_controls_v4/feedback")
    )
    p.add_argument(
        "--gpu", action="store_true", help="Also run matched-state GPU force/step comparisons"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--times", type=float, nargs="+", default=[0.2, 1, 2, 3, 4, 5])
    p.add_argument("--variants", nargs="+", choices=VARIANTS + TIRE_VARIANTS, default=VARIANTS)
    p.add_argument("--output", type=Path, default=Path("output/phase2/b_backend_audit"))
    args = p.parse_args()
    if args.gpu:
        import torch

        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            p.error("--gpu requires visible CUDA; omit it for native-only analysis")
    if (args.output / "report.json").exists():
        p.error("Use a new output directory")
    cfg = json.loads((args.recording / "run.json").read_text())["environment"]
    template, _, ctrl0, maps = build_rider_model(cfg)
    track, ids = select_samples(args.recording, args.times)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "recording": str(args.recording),
        "sample_times_s": track["time_s"][ids].tolist(),
        "gpu_ran": args.gpu,
        "variants": {},
    }
    from contextlib import nullcontext

    from bicycle_sim.rl.cuda_context import shared_cuda_stream

    context = shared_cuda_stream(args.device) if args.gpu else nullcontext()
    with context:
        for variant in args.variants:
            m = variant_model(args.recording / "scene.xml", variant)
            if (m.nq, m.nv, m.nu) != (template.nq, template.nv, template.nu):
                raise ValueError("Saved topology does not match the task mapping")
            native = native_samples(m, maps, cfg, ctrl0, track, ids)
            np.savez_compressed(args.output / f"{variant}_native.npz", **native)
            summary = {
                "native_ax_mps2": native["native_ax"].tolist(),
                "native_com_ax_mps2": native["native_com_ax"].tolist(),
                "native_ncon": native["ncon"].tolist(),
                "native_warnings": native["warnings"].tolist(),
                "native_ground_force_ax_mps2": ground_force_acceleration(
                    m, native, native["horizontal_heading"]
                ).tolist(),
            }
            if variant == "baseline":
                summary["recorded_gpu_ax_mps2"] = native["recorded_ax"].tolist()
                summary["recorded_gpu_com_ax_mps2"] = native["recorded_com_ax"].tolist()
            if args.gpu:
                gpu = gpu_samples(m, maps, cfg, track, ids, native, args.device)
                np.savez_compressed(args.output / f"{variant}_gpu.npz", **gpu)
                summary["gpu_ax_mps2"] = gpu["gpu_ax"].tolist()
                summary["gpu_com_ax_mps2"] = gpu["gpu_com_ax"].tolist()
                summary["gpu_ncon"] = gpu["ncon"].tolist()
                summary["gpu_ground_force_ax_mps2"] = ground_force_acceleration(
                    m, gpu, native["horizontal_heading"]
                ).tolist()
                summary["mean_gpu_minus_native_ax_mps2"] = float(
                    np.mean(gpu["gpu_ax"] - native["native_ax"])
                )
                summary["mean_gpu_minus_native_com_ax_mps2"] = float(
                    np.mean(gpu["gpu_com_ax"] - native["native_com_ax"])
                )
                summary["forward_max_abs_difference"] = {
                    key: float(np.max(abs(gpu[key] - native[key]))) for key in FIELDS
                }
            report["variants"][variant] = summary
            print(variant, json.dumps(summary), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "report.json")


if __name__ == "__main__":
    main()
