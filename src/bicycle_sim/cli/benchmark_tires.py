"""Compare tire collision discretizations in memory; do not replace production assets."""

import copy
import hashlib
import json
import math
import platform
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from bicycle_sim.cli.audit_physics import lean
from bicycle_sim.models.bicycle import build_bicycle, numbers, sub
from bicycle_sim.models.tire import rounded_envelope_vertices

CONFIG_PATH = Path("configs/bicycle_v0.json")


def change_tire(body, segments, config, root):
    if not segments:
        return
    original = body.find("geom")
    attrs = dict(original.attrib)
    name = attrs.pop("name")
    for key in ("type", "size", "quat"):
        attrs.pop(key, None)
    body.remove(original)
    if segments < 0:
        n = abs(segments)
        crown = 24
        radius = config["geometry"]["wheel_radius"]
        minor = config["geometry"]["wheel_half_width"]
        vertices = rounded_envelope_vertices(radius, minor, n, crown)
        asset = root.find("asset")
        if asset is None:
            asset = sub(root, "asset")
        mesh_name = f"{name}_rounded_envelope"
        sub(asset, "mesh", name=mesh_name, vertex=numbers(vertices))
        sub(body, "geom", name=name, type="mesh", mesh=mesh_name, **attrs)
        return
    minor = config["geometry"]["wheel_half_width"]
    major = config["geometry"]["wheel_radius"] - minor
    for i in range(segments):
        a, b = 2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments
        sub(
            body,
            "geom",
            name=f"{name}_{i}",
            type="capsule",
            size=minor,
            fromto=numbers(
                [
                    major * math.cos(a),
                    0,
                    major * math.sin(a),
                    major * math.cos(b),
                    0,
                    major * math.sin(b),
                ]
            ),
            **attrs,
        )


def whole_bike(config, segments):
    baseline = copy.deepcopy(config)
    baseline["geometry"]["tire_profile"] = {"type": "cylinder"}
    root = build_bicycle(baseline)
    for name in ("rear_wheel", "front_wheel"):
        change_tire(root.find(f".//body[@name='{name}']"), segments, config, root)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def roller(config, segments, mass, timestep):
    baseline = copy.deepcopy(config)
    baseline["geometry"]["tire_profile"] = {"type": "cylinder"}
    original = build_bicycle(baseline)
    root = ET.Element("mujoco", model="tire_vertical_foreaft_rail_diagnostic")
    for tag in ("compiler", "option", "default"):
        root.append(copy.deepcopy(original.find(tag)))
    root.find("option").set("timestep", str(timestep))
    world = sub(root, "worldbody")
    world.append(copy.deepcopy(original.find("./worldbody/geom[@name='ground']")))
    carrier = sub(world, "body", name="axle_carrier")
    sub(carrier, "joint", name="travel", type="slide", axis="1 0 0", damping=0, armature=0)
    sub(carrier, "joint", name="vertical", type="slide", axis="0 0 1", damping=0, armature=0)
    wheel = copy.deepcopy(original.find(".//body[@name='rear_wheel']"))
    wheel_mass = config["mass_targets_kg"]["rear_wheel"]
    sub(carrier, "inertial", pos="0 0 0", mass=mass - wheel_mass, diaginertia=".01 .01 .01")
    change_tire(wheel, segments, config, root)
    carrier.append(wheel)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def rolling_metrics(config, segments, mass, dt, speed):
    m = roller(config, segments, mass, dt)
    d = mujoco.MjData(m)
    for _ in range(round(0.5 / dt)):
        mujoco.mj_step(m, d)
    # Lateral/roll motion is blocked by the diagnostic rail, not by a balance policy.
    d.qvel[0] = speed
    d.qvel[2] = speed / config["geometry"]["wheel_radius"]
    rows = []
    for i in range(round(3 / dt)):
        d.qfrc_applied[0] = 100 * (speed - d.qvel[0])
        mujoco.mj_step(m, d)
        if i < round(1 / dt):
            continue
        force = 0.0
        for j in range(d.ncon):
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m, d, j, wrench)
            force += wrench[0]
        rows.append([d.qpos[1], force, d.ncon, d.qvel[0]])
    a = np.array(rows)
    averaging_steps = round(0.02 / dt)
    averaged_force = (
        a[: len(a) // averaging_steps * averaging_steps, 1]
        .reshape(-1, averaging_steps)
        .mean(axis=1)
    )
    return {
        "mass_per_wheel_kg": mass,
        "timestep_s": dt,
        "target_speed_m_s": speed,
        "height_peak_to_peak_mm": float(np.ptp(a[:, 0]) * 1000),
        "height_std_mm": float(np.std(a[:, 0]) * 1000),
        "normal_force_mean_N": float(np.mean(a[:, 1])),
        "normal_force_cv": float(np.std(a[:, 1]) / np.mean(a[:, 1])),
        "normal_force_peak_N": float(np.max(a[:, 1])),
        "normal_force_20ms_cv": float(np.std(averaged_force) / np.mean(averaged_force)),
        "normal_force_20ms_peak_N": float(np.max(averaged_force)),
        "contact_count_mean": float(np.mean(a[:, 2])),
        "contact_count_max": int(np.max(a[:, 2])),
        "mean_speed_m_s": float(np.mean(a[:, 3])),
        "warnings": d.warning.number.tolist(),
        "finite": bool(np.isfinite(d.qpos).all() and np.isfinite(a).all()),
    }


def cpu_timing(model, blocks=10, steps=500, rolling=False, radius=0.27):
    data = mujoco.MjData(model)
    if rolling:
        data.qvel[0] = 2.5
        data.qvel[2] = 2.5 / radius
    for _ in range(300):
        mujoco.mj_step(model, data)
    snapshot = mujoco.MjData(model)
    mujoco.mj_copyData(snapshot, model, data)
    times = []
    for _ in range(blocks):
        mujoco.mj_copyData(data, model, snapshot)
        start = time.perf_counter()
        mujoco.mj_step(model, data, nstep=steps)
        times.append(time.perf_counter() - start)
    return {
        "median_us_per_step": float(np.median(times) / steps * 1e6),
        "block_seconds": times,
        "ngeom": model.ngeom,
        "nmesh": model.nmesh,
        "compiled_mesh_vertices": int(sum(model.mesh_vertnum)),
        "scope": "Single CPU environment; repeated fixed-length blocks from the same warmed state. Not GPU training throughput",
        "block_simulation_s": steps * model.opt.timestep,
        "initial_rolling": rolling,
    }


def main():
    config = json.loads(CONFIG_PATH.read_text())
    out = Path("output/phase1/tire_benchmark")
    out.mkdir(parents=True, exist_ok=True)
    try:
        smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False, timeout=10
        )
        gpu_probe = {"returncode": smi.returncode, "output": smi.stdout + smi.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu_probe = {"error": str(error)}
    report = {
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "machine": platform.platform(),
        "gpu_probe": gpu_probe,
        "notes": [
            "Wheel radius, width, mass, inertia, friction and contact softness held constant.",
            "Rolling tests use a guided single-wheel rig with towing velocity control.",
            "Do not treat the rail fixture as a self-balancing bicycle.",
            "Timings are CPU only; production assets/config and dependencies are unchanged.",
        ],
        "variants": {},
    }
    for n in (0, 48, 96, 192, -96, -192, -384):
        label = "cylinder" if n == 0 else (f"convex_{abs(n)}x24" if n < 0 else f"ring_{n}")
        m = whole_bike(config, n)
        result = {
            "circumferential_segments": abs(n),
            "construction": ("convex_envelope" if n < 0 else ("capsule_ring" if n else "cylinder")),
            "geometric_chord_sag_mm": 0
            if not n
            else 1000
            * (
                config["geometry"]["wheel_radius"]
                if n < 0
                else config["geometry"]["wheel_radius"] - config["geometry"]["wheel_half_width"]
            )
            * (1 - math.cos(math.pi / abs(n))),
            "tilt": {str(angle): lean(m, angle, duration=3) for angle in (0.1, 1)},
            "cpu_timing": cpu_timing(m),
            "rolling": [],
        }
        for mass in (4.5, 21.6971176):
            for dt in (0.002, 0.001):
                for speed in (0.5, 2.5, 4.0):
                    result["rolling"].append(rolling_metrics(config, n, mass, dt, speed))
        # Measure isolated rolling physics too, avoiding Python sampling in the timing loop.
        rig = roller(config, n, 21.6971176, 0.002)
        result["guided_cpu_timing"] = cpu_timing(
            rig, rolling=True, radius=config["geometry"]["wheel_radius"]
        )
        report["variants"][label] = result
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        base = result["rolling"][10]  # loaded, 1 ms, 2.5 m/s
        print(
            label,
            "us/step",
            round(result["cpu_timing"]["median_us_per_step"], 2),
            "loaded 1ms @2.5m/s height mm",
            round(base["height_peak_to_peak_mm"], 4),
            flush=True,
        )
    print(out / "results.json")


if __name__ == "__main__":
    main()
