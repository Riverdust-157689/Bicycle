"""Native CPU seat-contact bench; fixed bike, free pelvis, no palm/seat equality.

The controller is the existing nominal arm/hand PD and phase leg reference,
not a learned whole-body controller. This tests support geometry, not riding.
"""

import argparse
import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from bicycle_sim.models.bicycle import numbers
from bicycle_sim.rl.rider_gpu_env import leg_servo_reference
from bicycle_sim.rl.rider_model import LEG_PARTS, build_rider_model


def pelvis_hull(m, q):
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    body = m.body("pelvis").id
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_contype[g]]
    if len(gs) != 1 or m.geom_type[gs[0]] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError("Expected the current single pelvis collision mesh")
    g = gs[0]
    mesh = m.geom_dataid[g]
    start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
    vertices = m.mesh_vert[start : start + count] @ d.geom_xmat[g].reshape(3, 3).T
    vertices += d.geom_xpos[g]
    return vertices


def pad_height(vertices, center_x):
    """Set two rounded rails 0.5 mm below the initial convex pelvis hull."""
    equations = ConvexHull(vertices).equations
    lower = equations[equations[:, 2] < -1e-7]
    upper = equations[equations[:, 2] > 1e-7]
    gaps = []
    for yc in (-0.030, 0.030):
        x, y = np.meshgrid(
            np.linspace(center_x - 0.06, center_x + 0.06, 61),
            np.linspace(yc - 0.0159, yc + 0.0159, 41),
        )
        xy = np.column_stack((x.ravel(), y.ravel()))
        radial = np.maximum(np.abs(xy[:, 0] - center_x) - 0.045, 0) ** 2
        radial += (xy[:, 1] - yc) ** 2
        xy, radial = xy[radial < 0.016**2], radial[radial < 0.016**2]
        lo = np.max(-(xy @ lower[:, :2].T + lower[:, 3]) / lower[:, 2], axis=1)
        hi = np.min(-(xy @ upper[:, :2].T + upper[:, 3]) / upper[:, 2], axis=1)
        valid = lo <= hi
        gaps.extend((lo[valid] - np.sqrt(0.016**2 - radial[valid])).tolist())
    return min(gaps) - 0.0005


def make_bench(source, source_xml, q0, ctrl0, variant, vertices):
    # mj_saveLastXML follows the last compiled XML globally; do not serialize
    # `source` again after compiling the first bench model.
    root = ET.fromstring(source_xml)
    frame = root.find("./worldbody/body[@name='bike_frame']")
    frame.remove(frame.find("joint[@name='bike_root']"))
    equality = root.find("equality")
    if equality is not None:
        root.remove(equality)
    seat = frame.find("geom[@name='saddle']")
    pos = np.fromstring(seat.get("pos"), sep=" ")
    half = np.fromstring(seat.get("size"), sep=" ")
    top = pos[2] + half[2]
    rise = float(vertices[:, 2].min() - top - 0.0005)
    seat.set("friction", numbers([variant.get("mu", 0.6), 0, 0]))
    if variant["shape"] in ("raised", "ellipsoid"):
        pos[2] += rise
        seat.set("pos", numbers(pos))
    if variant["shape"] == "ellipsoid":
        seat.set("type", "ellipsoid")
    if variant.get("tilt_deg"):
        a = np.deg2rad(variant["tilt_deg"]) / 2
        seat.set("quat", numbers([np.cos(a), 0, np.sin(a), 0]))
    if variant["shape"] == "pads":
        z = pad_height(vertices, pos[0])
        for side, y in (("left", 0.03), ("right", -0.03)):
            attrs = dict(seat.attrib)
            attrs.update(
                name=f"saddle_{side}_pad",
                type="capsule",
                size=".016",
                fromto=numbers([pos[0] - 0.045, y, z, pos[0] + 0.045, y, z]),
                density="0",
                rgba=".22 .32 .45 1",
            )
            attrs.pop("pos", None)
            ET.SubElement(frame, "geom", attrs)
    # The source has explicit inertias; adding zero-mass pads changes only contact.
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    q = m.qpos0.copy()
    for j in range(m.njnt):
        old = source.joint(m.joint(j).name).id
        count = 7 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        a, b = m.jnt_qposadr[j], source.jnt_qposadr[old]
        q[a : a + count] = q0[b : b + count]
    ctrl = np.array([ctrl0[source.actuator(m.actuator(a).name).id] for a in range(m.nu)])
    return m, q, ctrl, rise


def run_case(m, q0, ctrl, table, mode, seconds, output, push_axis=None, device="cpu"):
    # Torch is only used for the existing interpolation/feed-forward helper.
    import torch

    output.mkdir(parents=True)
    mujoco.mj_saveLastXML(str(output / "scene.xml"), m)
    d = mujoco.MjData(m)
    d.qpos[:], d.ctrl[:] = q0, ctrl
    mujoco.mj_forward(m, d)
    pelvis = m.body("pelvis").id
    seat_geoms = {g for g in range(m.ngeom) if m.geom(g).name.startswith("saddle")}
    crank = m.joint("crank_spin")
    legs = [m.joint(f"{side}_{part}_joint").id for side in ("left", "right") for part in LEG_PARTS]
    aids = np.array([m.actuator(m.joint(j).name).id for j in legs])
    ratio = torch.tensor(m.actuator_biasprm[aids, 2] / -m.actuator_gainprm[aids, 0])
    table = torch.tensor(table)
    speed = 3.7037037037 if mode == "pedaling" else 0.0
    drive = m.actuator("crank_drive").id
    initial_pelvis = d.xpos[pelvis].copy()
    sim = None
    if device != "cpu":
        import mujoco_warp as mjwarp
        import warp as wp
        from mjlab.sim import Simulation, SimulationCfg

        settings = SimulationCfg(nconmax=128, njmax=1024)
        settings.mujoco.timestep, settings.mujoco.iterations = m.opt.timestep, m.opt.iterations
        sim = Simulation(1, settings, m, device=device)
        sim.reset()
        dtype = sim.data.qpos[:].dtype
        sim.data.qpos[0] = torch.as_tensor(q0, dtype=dtype, device=device)
        sim.data.qvel[0] = 0
        sim.data.ctrl[0] = torch.as_tensor(ctrl, dtype=dtype, device=device)
        sim.forward()
        table, ratio = table.to(device=device, dtype=dtype), ratio.to(device=device, dtype=dtype)
        gpu_aids = torch.as_tensor(aids, device=device)
        limits = torch.as_tensor(m.actuator_ctrlrange[aids], dtype=dtype, device=device)
    frames, stats = [], []
    for step in range(round(seconds / m.opt.timestep) + 1):
        if step % 10 == 0:
            forces = None
            if sim is not None:
                if (
                    torch.any(sim.data.overflow[:] != 0)
                    or not torch.isfinite(sim.data.qpos[:]).all()
                ):
                    raise FloatingPointError("Warp capacity overflow or nonfinite state")
                mjwarp.get_data_into(d, m, sim.wp_data)
                if int(sim.data.nacon[0]) != d.ncon:
                    raise RuntimeError("Single-world contact download is inconsistent")
                if d.ncon:
                    ids = wp.array(np.arange(d.ncon, dtype=np.int32), device=device)
                    buffer = wp.zeros(d.ncon, dtype=wp.spatial_vector, device=device)
                    mjwarp.contact_force(sim.wp_model, sim.wp_data, ids, False, buffer)
                    forces = buffer.numpy()
            force_z, pelvis_force_z, deepest, contacts, other_bodies = 0.0, 0.0, 0.0, 0, []
            for i, contact in enumerate(d.contact):
                g1, g2 = map(int, contact.geom)
                if g1 not in seat_geoms and g2 not in seat_geoms:
                    continue
                other = g2 if g1 in seat_geoms else g1
                if m.geom_bodyid[other] == m.body("bike_frame").id:
                    continue
                force = np.zeros(6)
                if forces is None:
                    mujoco.mj_contactForce(m, d, i, force)
                else:
                    force[:] = forces[i]
                # Contact frame normal is geom1 -> geom2; force is on geom2.
                world_force = contact.frame.reshape(3, 3).T @ force[:3]
                fz = world_force[2] * (1 if g1 in seat_geoms else -1)
                force_z += fz
                if m.geom_bodyid[other] == pelvis:
                    pelvis_force_z += fz
                else:
                    other_bodies.append(m.body(m.geom_bodyid[other]).name)
                deepest = min(deepest, float(contact.dist))
                contacts += 1
            shift = d.xpos[pelvis] - initial_pelvis
            R = d.xmat[pelvis].reshape(3, 3)
            stats.append(
                {
                    "time_s": float(d.time),
                    "seat_force_z_N": force_z,
                    "pelvis_seat_force_z_N": pelvis_force_z,
                    "seat_contacts": contacts,
                    "deepest_m": deepest,
                    "pelvis_shift_m": shift.tolist(),
                    "pelvis_tilt_rad": float(np.arccos(np.clip(R[2, 2], -1, 1))),
                    "crank_angle_rad": float(d.qpos[crank.qposadr[0]]),
                    "crank_speed_rad_s": float(d.qvel[crank.dofadr[0]]),
                    "other_seat_bodies": other_bodies,
                }
            )
            frames.append((d.qpos.copy(), d.qvel.copy()))
            if np.linalg.norm(shift[:2]) > 0.15 or shift[2] < -0.15:
                break
        if step == round(seconds / m.opt.timestep):
            break
        if sim is not None:
            ref = leg_servo_reference(
                sim.data.qpos[:, crank.qposadr[0]], sim.data.qvel[:, crank.dofadr[0]], table, ratio
            )
            sim.data.ctrl[:, gpu_aids] = ref.clamp(limits[:, 0], limits[:, 1])
            sim.data.ctrl[:, drive] = speed
            sim.data.xfrc_applied[:] = 0
            if push_axis is not None and 3 <= step * m.opt.timestep < 3.2:
                sim.data.xfrc_applied[0, pelvis, push_axis] = 40
            sim.step()
            continue
        ref = leg_servo_reference(
            torch.tensor([d.qpos[crank.qposadr[0]]]),
            torch.tensor([d.qvel[crank.dofadr[0]]]),
            table,
            ratio,
        )
        d.ctrl[aids] = np.clip(
            ref[0].numpy(), m.actuator_ctrlrange[aids, 0], m.actuator_ctrlrange[aids, 1]
        )
        d.ctrl[drive] = speed
        d.xfrc_applied[pelvis] = 0
        if push_axis is not None and 3 <= d.time < 3.2:
            d.xfrc_applied[pelvis, push_axis] = 40
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            raise FloatingPointError("Nonfinite state")
    window = [s for s in stats if s["time_s"] >= 1]
    summary = {
        "duration_s": float(d.time),
        "robot_weight_N": float(m.body_subtreemass[pelvis] * 9.81),
        "final_pelvis_shift_m": stats[-1]["pelvis_shift_m"],
        "backend": "native CPU MuJoCo" if sim is None else "mjlab/MuJoCo-Warp",
        "warnings": [int(w.number) for w in d.warning] if sim is None else None,
        "contact_force_source": "mj_contactForce"
        if sim is None
        else "direct mujoco_warp.contact_force",
        "equality_count": m.neq,
        "bike_fixed": True,
        "mean_seat_force_z_N": float(np.mean([s["seat_force_z_N"] for s in window]))
        if window
        else None,
        "pelvis_supported_fraction": float(
            np.mean([s["pelvis_seat_force_z_N"] > 1 for s in window])
        )
        if window
        else None,
        "max_penetration_m": -min(s["deepest_m"] for s in stats),
        "max_pelvis_tilt_rad": max(s["pelvis_tilt_rad"] for s in stats),
        "max_horizontal_shift_m": max(np.linalg.norm(s["pelvis_shift_m"][:2]) for s in stats),
        "push_axis": push_axis,
        "push_force_N": 40 if push_axis is not None else 0,
        "other_seat_bodies": sorted({b for s in stats for b in s["other_seat_bodies"]}),
    }
    np.savez_compressed(
        output / "trajectory.npz",
        qpos=np.array([f[0] for f in frames]),
        qvel=np.array([f[1] for f in frames]),
        time_s=[s["time_s"] for s in stats],
    )
    (output / "results.json").write_text(
        json.dumps({"summary": summary, "trace": stats}, indent=2) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--suite", choices=("nominal", "push"), default="nominal")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", nargs="+", help="Variant names, e.g. original raised_box")
    parser.add_argument(
        "--output", type=Path, default=Path("output/phase2/saddle_contact_audit_v1")
    )
    args = parser.parse_args()
    if args.seconds <= 1 or args.output.exists():
        parser.error("Use seconds > 1 and a new output directory")
    if args.device != "cpu":
        import torch

        if not args.device.startswith("cuda:") or not torch.cuda.is_available():
            parser.error("The requested Warp backend requires visible CUDA")
    cfg = json.loads(Path("configs/rl/bicycle_b_contact_sac.json").read_text())
    source, q0, ctrl, maps = build_rider_model(cfg)
    with tempfile.TemporaryDirectory(prefix="seat-source-", dir="/tmp") as tmp:
        path = Path(tmp) / "source.xml"
        mujoco.mj_saveLastXML(str(path), source)
        source_xml = path.read_text()
    vertices = pelvis_hull(source, q0)
    args.output.mkdir(parents=True)
    report = {
        "backend": "native CPU MuJoCo" if args.device == "cpu" else "mjlab/MuJoCo-Warp",
        "learning": False,
        "initial_pelvis_saddle_gap_m": float(
            vertices[:, 2].min()
            - source.geom_pos[source.geom("saddle").id, 2]
            - source.geom_size[source.geom("saddle").id, 2]
        ),
        "assistance": [
            "fixed bicycle frame",
            "nominal arm/hand/waist PD",
            "phase leg PD",
            "crank speed motor",
        ],
        "seat_or_hand_equalities": False,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *sorted(Path("src/bicycle_sim").rglob("*.py")),
                Path("configs/rl/bicycle_b_contact_sac.json"),
                Path("assets/scenes/rider_fit_diagnostic.xml"),
                Path(cfg["grasp_reference_path"]),
                Path("output/phase1/rider_kinematic_reference.npz"),
                Path("uv.lock"),
            ]
        },
        "cases": {},
    }
    variants = [
        {"name": "original", "shape": "original"},
        {"name": "raised_box", "shape": "raised"},
        {"name": "raised_ellipsoid", "shape": "ellipsoid"},
        {"name": "paired_round_pads", "shape": "pads"},
        {"name": "raised_mu03", "shape": "raised", "mu": 0.3},
        {"name": "raised_mu09", "shape": "raised", "mu": 0.9},
        {"name": "raised_tilt_pos3", "shape": "raised", "tilt_deg": 3},
        {"name": "raised_tilt_neg3", "shape": "raised", "tilt_deg": -3},
    ]
    if args.suite == "push":
        if args.seconds < 5:
            parser.error("Push suite requires at least 5 seconds")
        variants = [
            v for v in variants if v["name"] in ("raised_box", "raised_mu03", "raised_mu09")
        ]
    if args.only:
        if set(args.only) - {v["name"] for v in variants}:
            parser.error("Unknown variant for this suite")
        variants = [v for v in variants if v["name"] in args.only]
    for variant in variants:
        modes = ("static", "pedaling") if args.suite == "nominal" else ("push_x", "push_y")
        for mode in modes:
            m, q, c, rise = make_bench(source, source_xml, q0, ctrl, variant, vertices)
            key = f"{variant['name']}_{mode}"
            axis = {"push_x": 0, "push_y": 1}.get(mode)
            from bicycle_sim.rl.cuda_context import shared_cuda_stream

            context = nullcontext() if args.device == "cpu" else shared_cuda_stream(args.device)
            with context:
                result = run_case(
                    m,
                    q,
                    c,
                    maps["leg_table"],
                    "pedaling" if axis is not None else mode,
                    args.seconds,
                    args.output / key,
                    push_axis=axis,
                    device=args.device,
                )
            report["cases"][key] = dict(variant=variant, suggested_rise_m=rise, **result)
            print(key, json.dumps(result), flush=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
