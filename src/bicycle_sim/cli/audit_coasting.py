"""Loaded, unpowered straight-line coast bench; no balance policy or reward tuning."""

import argparse
import json
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

from bicycle_sim.models.bicycle import build_bicycle, numbers, sub
from bicycle_sim.models.tire import set_ellipsoid_tire_geoms, set_wheel_ground_rolling_contact


def build_coast_bench(*, mass_kg, rolling_m=0.0, sliding_mu=0.8, dt=0.002, airborne=False):
    cfg = json.loads(Path("configs/bicycle_v0.json").read_text())
    cfg["contact_parameters"]["tire"]["friction"][0] = sliding_mu
    root = build_bicycle(cfg)
    initial = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    set_ellipsoid_tire_geoms(root, initial)
    set_wheel_ground_rolling_contact(root, initial, rolling_m)
    frame = root.find("./worldbody/body[@name='bike_frame']")
    frame.remove(frame.find("freejoint"))
    # Bench guides constrain lateral displacement and orientation, leaving actual
    # forward/vertical dynamics and tire normal loads. They supply no forward work.
    for name, axis in (("coast_x", "1 0 0"), ("coast_z", "0 0 1")):
        sub(
            frame,
            "joint",
            name=name,
            type="slide",
            axis=axis,
            damping="0",
            armature="0",
            frictionloss="0",
        )
    steering = root.find(".//body[@name='steering']")
    steering.remove(steering.find("joint[@name='steer']"))
    root.remove(root.find("sensor"))
    extra = mass_kg - initial.body_mass.sum()
    if extra < 0:
        raise ValueError("Total bench mass must be at least the bicycle mass")
    if extra > 0:
        ballast = sub(frame, "body", name="rider_ballast", pos=".24 0 .9")
        sub(ballast, "inertial", mass=extra, pos="0 0 0", diaginertia=numbers([extra * 0.05] * 3))
    if airborne:
        root.find("option").set("gravity", "0 0 0")
        root.find("worldbody").remove(root.find("./worldbody/geom[@name='ground']"))
        for pair in list(root.findall("./contact/pair")):
            root.find("contact").remove(pair)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    m.opt.timestep = dt
    return m


def run_coast(m, *, speed=2.0, seconds=25.0, torque_nm=0.0):
    d = mujoco.MjData(m)
    x = m.joint("coast_x").dofadr[0]
    z = m.joint("coast_z").dofadr[0]
    wheel = [m.joint(f"{s}_wheel_spin").dofadr[0] for s in ("rear", "front")]
    crank = m.joint("crank_spin").dofadr[0]
    radius = 0.27
    for _ in range(round(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    d.qvel[:] = 0
    d.qvel[x], d.qvel[wheel] = speed, speed / radius
    d.qfrc_applied[crank] = torque_nm
    # Crank is stationary in the coast test: the rear wheel must overrun it.
    origin = d.qpos[m.joint("coast_x").qposadr[0]]
    samples = []
    clutch_work = 0.0
    contact_work = 0.0
    max_clutch_power = -np.inf
    max_drive_torque = 0.0
    max_vertical_speed = 0.0
    for step in range(round(seconds / m.opt.timestep) + 1):
        mujoco.mj_forward(m, d)
        clutch_power = float(d.qfrc_actuator @ d.qvel)
        max_clutch_power = max(max_clutch_power, clutch_power)
        clutch_work += clutch_power * m.opt.timestep
        contact_work += float(d.qfrc_constraint @ d.qvel) * m.opt.timestep
        max_drive_torque = max(max_drive_torque, abs(float(d.qfrc_actuator[wheel[0]])))
        max_vertical_speed = max(max_vertical_speed, abs(float(d.qvel[z])))
        if step % round(0.1 / m.opt.timestep) == 0:
            normal = 0.0
            max_slip = max(abs(float(d.qvel[x] - radius * d.qvel[w])) for w in wheel)
            for c in range(d.ncon):
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, c, f)
                normal += f[0]
            samples.append(
                {
                    "time_s": step * m.opt.timestep,
                    "speed_mps": float(d.qvel[x]),
                    "distance_m": float(d.qpos[m.joint("coast_x").qposadr[0]] - origin),
                    "normal_load_N": normal,
                    "max_slip_mps": max_slip,
                    "crank_speed_rad_s": float(d.qvel[crank]),
                }
            )
        if step < round(seconds / m.opt.timestep):
            mujoco.mj_step(m, d)
    below = [s["time_s"] for s in samples if abs(s["speed_mps"]) < 0.1]
    return {
        "mass_kg": float(m.body_mass.sum()),
        "timestep_s": m.opt.timestep,
        "initial_speed_mps": speed,
        "applied_crank_torque_Nm": torque_nm,
        "speed_at_10s_mps": samples[100]["speed_mps"] if seconds >= 10 else None,
        "speed_at_20s_mps": samples[200]["speed_mps"] if seconds >= 20 else None,
        "time_below_0_1mps_s": min(below, default=None),
        "distance_m": samples[-1]["distance_m"],
        "clutch_work_J": clutch_work,
        "maximum_clutch_power_W": max_clutch_power,
        "maximum_clutch_rear_torque_Nm": max_drive_torque,
        "contact_work_J": contact_work,
        "max_vertical_speed_mps": max_vertical_speed,
        "solver_warnings": int(d.warning.number.sum()),
        "samples": samples,
    }


def run_coast_warp(m, *, speed=2.0, seconds=25.0, device="cuda:0"):
    """The same unpowered bench on the training backend, with captured graphs."""
    from mjlab.sim import Simulation, SimulationCfg

    from bicycle_sim.rl.warp_capacity import configure_epa_horizon

    configure_epa_horizon(128)
    settings = SimulationCfg(nconmax=32, njmax=128)
    settings.mujoco.timestep = m.opt.timestep
    settings.mujoco.iterations = m.opt.iterations
    sim = Simulation(1, settings, m, device=device)
    sim.reset()
    x = m.joint("coast_x").dofadr[0]
    qx = m.joint("coast_x").qposadr[0]
    wheels = [m.joint(f"{s}_wheel_spin").dofadr[0] for s in ("rear", "front")]
    crank = m.joint("crank_spin").dofadr[0]
    for _ in range(round(0.5 / m.opt.timestep)):
        sim.step()
    sim.data.qvel[:] = 0
    sim.data.qvel[0, x] = speed
    for w in wheels:
        sim.data.qvel[0, w] = speed / 0.27
    origin = float(sim.data.qpos[0, qx])
    samples, peak_power = [], -np.inf
    for step in range(round(seconds / m.opt.timestep) + 1):
        if step % round(0.1 / m.opt.timestep) == 0:
            sim.forward()
            from bicycle_sim.rl.physics_diagnostics import RiderPhysicsError, physics_failure_report

            failure = physics_failure_report(
                sim.data.qpos[:], sim.data.qvel[:], sim.data.overflow[:]
            )
            if failure:
                raise RiderPhysicsError(failure, {})
            power = float((sim.data.qfrc_actuator[:] * sim.data.qvel[:]).sum())
            peak_power = max(peak_power, power)
            samples.append(
                {
                    "time_s": step * m.opt.timestep,
                    "speed_mps": float(sim.data.qvel[0, x]),
                    "distance_m": float(sim.data.qpos[0, qx]) - origin,
                    "crank_speed_rad_s": float(sim.data.qvel[0, crank]),
                }
            )
        if step < round(seconds / m.opt.timestep):
            sim.step()
    below = [s["time_s"] for s in samples if abs(s["speed_mps"]) < 0.1]
    return {
        "mass_kg": float(m.body_mass.sum()),
        "timestep_s": m.opt.timestep,
        "initial_speed_mps": speed,
        "applied_crank_torque_Nm": 0.0,
        "speed_at_10s_mps": samples[100]["speed_mps"] if seconds >= 10 else None,
        "speed_at_20s_mps": samples[200]["speed_mps"] if seconds >= 20 else None,
        "time_below_0_1mps_s": min(below, default=None),
        "distance_m": samples[-1]["distance_m"],
        "maximum_clutch_power_W": peak_power,
        "samples": samples,
    }


def free_lean(*, mass_kg, rolling_m, initial_roll=0.02, seconds=2.0):
    """No guide or controller: check that the loaded bike can still fall."""
    cfg = json.loads(Path("configs/bicycle_v0.json").read_text())
    root = build_bicycle(cfg)
    initial = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    set_ellipsoid_tire_geoms(root, initial)
    set_wheel_ground_rolling_contact(root, initial, rolling_m)
    ballast = sub(
        root.find("./worldbody/body[@name='bike_frame']"), "body", name="ballast", pos=".24 0 .9"
    )
    extra = mass_kg - initial.body_mass.sum()
    sub(ballast, "inertial", mass=extra, pos="0 0 0", diaginertia=numbers([extra * 0.05] * 3))
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    d.qpos[3:7] = [np.cos(initial_roll / 2), np.sin(initial_roll / 2), 0, 0]
    frame = m.body("bike_frame").id
    peak = 0.0
    for _ in range(round(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        peak = max(peak, float(np.arccos(np.clip(d.xmat[frame, 8], -1, 1))))
        if peak > 0.35:
            break
    return {
        "rolling_friction_m": rolling_m,
        "initial_roll_rad": initial_roll,
        "time_s": d.time,
        "max_tilt_rad": peak,
        "exceeded_20deg": bool(peak > 0.35),
        "solver_warnings": int(d.warning.number.sum()),
    }


def run_validations(mass):
    out = {}
    for dt in (0.002, 0.001, 0.0005):
        out[f"dt_{dt}"] = run_coast(build_coast_bench(mass_kg=mass, rolling_m=0.00405, dt=dt))
    out["pedal_4Nm"] = run_coast(build_coast_bench(mass_kg=mass, rolling_m=0.00405), torque_nm=4)
    out["rest"] = run_coast(build_coast_bench(mass_kg=mass, rolling_m=0.00405), speed=0, seconds=2)
    out["reverse"] = run_coast(build_coast_bench(mass_kg=mass, rolling_m=0.00405), speed=-2)
    for rolling in (0, 0.00405):
        out[f"airborne_{rolling}"] = run_coast(
            build_coast_bench(mass_kg=mass, rolling_m=rolling, airborne=True), seconds=2
        )
        for roll in (0.001, 0.02):
            out[f"lean_{rolling}_{roll}"] = free_lean(
                mass_kg=mass, rolling_m=rolling, initial_roll=roll
            )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("output/phase2/coasting_audit"))
    p.add_argument("--backend", choices=("native", "warp"), default="native")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new output directory")
    if args.backend == "native" and args.device != "cpu":
        p.error("The native backend runs on CPU; choose --backend warp for GPU")
    from bicycle_sim.rl.cuda_context import shared_cuda_stream

    context = shared_cuda_stream(args.device) if args.device.startswith("cuda") else nullcontext()
    args.output.mkdir(parents=True)
    from bicycle_sim.rl.rider_model import build_rider_model

    cfg = json.loads(Path("configs/rl/bicycle_b_whole_body_direct_noassist_sac.json").read_text())
    source, *_ = build_rider_model(cfg)
    mass = float(source.body_mass.sum())
    rows = []
    with context:
        for sliding, rolling in (
            (0.8, 0),
            (1.6, 0),
            (0.8, 0.00135),
            (0.8, 0.0027),
            (0.8, 0.00405),
            (0.8, 0.0054),
        ):
            m = build_coast_bench(mass_kg=mass, rolling_m=rolling, sliding_mu=sliding)
            row = (
                run_coast(m) if args.backend == "native" else run_coast_warp(m, device=args.device)
            )
            row.update(
                sliding_mu=sliding, rolling_friction_m=rolling, approximate_Crr=rolling / 0.27
            )
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
            (args.output / "partial_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    report = {
        "backend": f"{args.backend} MuJoCo {args.device}",
        "scope": "straight upright x/z guide; rigid ballast matched to current total rider+bike mass; no controller; no propulsion",
        "environment": cfg,
        "results": rows,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.backend == "native":
        (args.output / "validation.json").write_text(
            json.dumps(run_validations(mass), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
