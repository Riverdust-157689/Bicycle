"""Run controlled CPU contact/tilt diagnostics; wheel alternatives exist in memory only."""

import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from bicycle_sim.models.bicycle import build_bicycle

C = json.loads(Path("configs/bicycle_v0.json").read_text())


def scene(shape="current", tau=0.02, load=0):
    cfg = copy.deepcopy(C)
    if shape != "current":
        cfg["geometry"]["tire_profile"] = {"type": "cylinder"}
    r = build_bicycle(cfg)
    for element in r.iter("geom"):
        element.set("solref", f"{tau} 1")
    if load:
        b = ET.SubElement(
            r.find(".//body[@name='bike_frame']"),
            "body",
            name="rigid_payload",
            pos=" ".join(
                str(x)
                for x in [
                    C["geometry"]["saddle_top"][0],
                    0,
                    C["geometry"]["saddle_top"][2]
                    + C["rider_reference"]["pelvis_height_above_saddle"],
                ]
            ),
        )
        ET.SubElement(b, "inertial", pos="0 0 0", mass=str(load), diaginertia=".2 .2 .2")
    for name in ("rear_wheel", "front_wheel"):
        b = r.find(f".//body[@name='{name}']")
        g = b.find("geom")
        if shape == "ellipsoid":
            g.set("type", "ellipsoid")
            g.set(
                "size",
                f"{C['geometry']['wheel_radius']} {C['geometry']['wheel_radius']} {C['geometry']['wheel_half_width']}",
            )
        if shape == "rounded_ring":
            b.remove(g)
            n = 48
            minor = C["geometry"]["wheel_half_width"]
            major = C["geometry"]["wheel_radius"] - minor
            for i in range(n):
                a, beta = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
                ET.SubElement(
                    b,
                    "geom",
                    name=f"{name}_tire_{i}",
                    type="capsule",
                    size=str(minor),
                    priority="1",
                    friction="0.8 0 0",
                    solref=f"{tau} 1",
                    fromto=f"{major * math.cos(a)} 0 {major * math.sin(a)} {major * math.cos(beta)} 0 {major * math.sin(beta)}",
                )
    return mujoco.MjModel.from_xml_string(ET.tostring(r, encoding="unicode"))


def lean(m, deg, duration=6):
    d = mujoco.MjData(m)
    angle = math.radians(deg)
    d.qpos[3:7] = [math.cos(angle / 2), math.sin(angle / 2), 0, 0]
    worst = 0
    fall = None
    for _ in range(round(duration / m.opt.timestep)):
        mujoco.mj_step(m, d)
        rot = d.xmat[1].reshape(3, 3)
        v = math.degrees(math.acos(np.clip(rot[2, 2], -1, 1)))
        worst = max(worst, v)
        if v > 30 and fall is None:
            fall = float(d.time)
    return {
        "final_tilt_deg": v,
        "max_tilt_deg": worst,
        "time_to_30deg_s": fall,
        "warnings": d.warning.number.tolist(),
    }


def transmission_and_steering():
    results = {}
    for friction in (0.0, 0.8):
        m = scene()
        for name in ("rear_wheel_tire", "front_wheel_tire"):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            m.geom_friction[g, :] = [friction, 0, 0]
            if friction == 0:
                m.geom_condim[g] = 1  # Pure normal contact, no zero-width friction cone.
        d = mujoco.MjData(m)
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "crank_spin")
        rear = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "rear_wheel_spin")
        for _ in range(round(1 / m.opt.timestep)):
            d.qfrc_applied[m.jnt_dofadr[j]] = 1.0
            mujoco.mj_step(m, d)
        results[f"crank_torque_1Nm_mu_{friction}"] = {
            "forward_distance_m": float(d.qpos[0]),
            "forward_speed_m_s": float(d.qvel[0]),
            "crank_angle_rad": float(d.qpos[m.jnt_qposadr[j]]),
            "rear_wheel_angle_rad": float(d.qpos[m.jnt_qposadr[rear]]),
            "rear_ground_slip_m_s": float(
                C["geometry"]["wheel_radius"] * d.qvel[m.jnt_dofadr[rear]] - d.qvel[0]
            ),
            "warnings": d.warning.number.tolist(),
        }
    root = build_bicycle(C, fixed_frame=True)
    world = root.find("worldbody")
    world.remove(world.find("./geom[@name='ground']"))
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "steer")
    wheel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "front_wheel_spin")
    rows = []
    for angle in (-0.2, 0, 0.2):
        d.qpos[m.jnt_qposadr[j]] = angle
        mujoco.mj_forward(m, d)
        axis = d.xaxis[wheel]
        rows.append(
            {"steer_rad": angle, "front_wheel_heading_rad": float(math.atan2(-axis[0], axis[1]))}
        )
    results["steer_wheel_alignment"] = rows
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_grip")
    point = d.site_xpos[site]
    lever = point - d.xanchor[j]
    force = np.cross(d.xaxis[j], lever)
    force *= 1 / np.linalg.norm(force)
    qforce = np.zeros(m.nv)
    mujoco.mj_applyFT(m, d, force, np.zeros(3), point, m.site_bodyid[site], qforce)
    results["one_newton_tangential_grip_force_steer_torque_Nm"] = float(qforce[m.jnt_dofadr[j]])
    return results


def collision_sweep():
    root = build_bicycle(C, fixed_frame=True)
    root.find("worldbody").remove(root.find("./worldbody/geom[@name='ground']"))
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    address = {
        name: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in (
            "steer",
            "crank_spin",
            "rear_wheel_spin",
            "left_pedal_spin",
            "right_pedal_spin",
        )
    }
    bad = []
    for phase in np.linspace(0, 2 * np.pi, 49):
        for steer in np.linspace(-0.35, 0.35, 9):
            d.qpos[address["crank_spin"]] = phase
            d.qpos[address["rear_wheel_spin"]] = 2 * phase
            d.qpos[address["left_pedal_spin"]] = d.qpos[address["right_pedal_spin"]] = -phase
            d.qpos[address["steer"]] = steer
            mujoco.mj_forward(m, d)
            for contact in d.contact:
                if contact.dist < -0.0001:
                    bad.append(
                        {
                            "phase": float(phase),
                            "steer": float(steer),
                            "distance_m": float(contact.dist),
                            "geoms": [
                                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                                for g in (contact.geom1, contact.geom2)
                            ],
                        }
                    )
    return {"phase_count": 49, "steering_count": 9, "penetrations_beyond_0_1mm": bad}


def current_parameters():
    m = scene()
    geoms = {}
    for name in (
        "ground",
        "rear_wheel_tire",
        "front_wheel_tire",
        "left_grip_surface",
        "saddle",
        "left_pedal_platform",
    ):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        geoms[name] = {
            "type": int(m.geom_type[g]),
            "friction": m.geom_friction[g].tolist(),
            "priority": int(m.geom_priority[g]),
            "condim": int(m.geom_condim[g]),
            "solref": m.geom_solref[g].tolist(),
            "solimp": m.geom_solimp[g].tolist(),
        }
    joints = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof = m.jnt_dofadr[j]
        joints[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)] = {
            "damping": float(m.dof_damping[dof]),
            "frictionloss": float(m.dof_frictionloss[dof]),
            "armature": float(m.dof_armature[dof]),
        }
    return {
        "mujoco_version": mujoco.__version__,
        "geoms": geoms,
        "joints": joints,
        "exclude_count": m.nexclude,
    }


def main():
    res = {
        "balance_learning_ready": False,
        "notes": [
            "Rounded convex production tire adopted; loaded riding, camber convergence and GPU validation pending.",
            "Alternative tires change collision shape only; masses and inertias are held fixed.",
            "Rigid payload is a load/compliance probe, not a full rider simulation.",
        ],
        "roll": {},
        "parameters": current_parameters(),
        "drive_and_steer": transmission_and_steering(),
        "collision_sweep": collision_sweep(),
    }
    for shape in ["current", "cylinder", "ellipsoid", "rounded_ring"]:
        res["roll"][shape] = {
            str(deg): lean(scene(shape), deg) for deg in [0, 0.1, 0.5, 1, 2, 3, 5, 8]
        }
        print(shape, "1deg:", res["roll"][shape]["1"], flush=True)
    res["compliance"] = []
    for tau in [0.01, 0.02, 0.04]:
        for load in [0, 34.3942352]:
            m = scene("cylinder", tau=tau, load=load)
            d = mujoco.MjData(m)
            for _ in range(1000):
                mujoco.mj_step(m, d)
            pen = max([-c.dist for c in d.contact], default=0)
            contactforces = []
            for i, c in enumerate(d.contact):
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                contactforces.append(f[0])
            res["compliance"].append(
                {
                    "shape": "cylinder",
                    "timeconst_s": tau,
                    "mass_kg": float(m.body_mass.sum()),
                    "penetration_m": pen,
                    "normal_force_sum_N": sum(contactforces),
                    "roll_1deg": lean(m, 1),
                    "warnings": d.warning.number.tolist(),
                }
            )
    Path("output/phase1/physics_audit").mkdir(parents=True, exist_ok=True)
    Path("output/phase1/physics_audit/contact_tilt.json").write_text(
        json.dumps(res, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
