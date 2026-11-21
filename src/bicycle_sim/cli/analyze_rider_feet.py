"""Audit pedal proximity from recorded poses; no dynamics are re-integrated."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from bicycle_sim.rl.rider_recording import extract_cohort_episode


def longest_sampled_run(mask, dt):
    current = longest = 0.0
    for flag, step in zip(mask[1:], dt, strict=True):
        current = current + float(step) if flag else 0.0
        longest = max(current, longest)
    return longest


def box_axis_gap(d, m, first, second):
    """Signed SAT gap: penetration depth or a lower bound on separation."""
    A = d.geom_xmat[first].reshape(3, 3)
    B = d.geom_xmat[second].reshape(3, 3)
    axes = np.concatenate((A.T, B.T, np.cross(A.T[:, None, :], B.T[None, :, :]).reshape(9, 3)))
    length = np.linalg.norm(axes, axis=1)
    axes = axes[length > 1e-10] / length[length > 1e-10, None]
    radius = np.abs(axes @ A) @ m.geom_size[first] + np.abs(axes @ B) @ m.geom_size[second]
    return float(np.max(np.abs(axes @ (d.geom_xpos[second] - d.geom_xpos[first])) - radius))


def audit_feet(m, track):
    """Geometry is evaluated at GPU-recorded poses; forces stay GPU measured."""
    d = mujoco.MjData(m)
    feet = [m.geom(f"{s}_riding_sole").id for s in ("left", "right")]
    pedals = [m.geom(f"{s}_pedal_platform").id for s in ("left", "right")]
    if any(m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in feet + pedals):
        raise ValueError("This foot audit requires the current box sole/pedal colliders")
    gap = np.empty((len(track["time_s"]), 2))
    for i, q in enumerate(track["qpos"]):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for side in range(2):
            axis_gap = box_axis_gap(d, m, feet[side], pedals[side])
            # Native distance queries can return zero for separated coaxial
            # boxes. SAT provides a reliable lower bound in that case, and
            # the penetration depth for intersecting boxes.
            gap[i, side] = (
                axis_gap
                if axis_gap <= 0
                else max(
                    axis_gap, mujoco.mj_geomDistance(m, d, feet[side], pedals[side], 1.0, None)
                )
            )
    dt = np.diff(track["time_s"])
    if np.any(dt < 0):
        raise ValueError("Select a single episode before auditing foot contact")
    duration = float(dt.sum())

    def fraction(mask):
        return float(np.sum(mask[1:] * dt) / duration) if duration > 0 else None

    report = {
        "duration_s": duration,
        "samples": len(gap),
        "sample_interval_s": float(np.median(dt)) if len(dt) else None,
        "geometry_source": "box SAT penetration / native separation lower-bounded by SAT, at recorded poses; no dynamics steps",
        "native_distance_query_limit_m": 1.0,
        "proximity_definition": "sole box to pedal box signed distance in [-2 mm, +5 mm]",
        "note": "Proximity is not loaded contact or proof of propulsion; control-step sampling cannot guarantee continuous contact.",
    }
    forces = track.get("support_normal_force_N")
    for side, name in enumerate(("left", "right")):
        g = gap[:, side]
        values = {
            "near_pedal_fraction": fraction((g >= -0.002) & (g <= 0.005)),
            "min_signed_gap_m": float(g.min()),
            "max_signed_gap_m": float(g.max()),
            "longest_gap_over_2cm_s": longest_sampled_run(g > 0.02, dt),
        }
        if forces is not None:
            values.update(
                loaded_contact_fraction=fraction(forces[:, side + 1] > 0.5),
                longest_unloaded_sampled_time_s=longest_sampled_run(forces[:, side + 1] <= 0.5, dt),
            )
        report[name] = values
    cq = m.joint("crank_spin").qposadr[0]
    report["crank_net_revolutions"] = float(
        (track["qpos"][-1, cq] - track["qpos"][0, cq]) / (2 * np.pi)
    )
    speed = track.get("crank_speed_rad_s")
    if speed is not None:
        report["crank_reverse_fraction"] = fraction(speed < -0.1)
    if speed is not None and "foot_to_crank_torque_Nm" in track and duration > 0:
        power = track["foot_to_crank_torque_Nm"] * speed
        report["sampled_mean_foot_to_crank_power_W"] = float(np.sum(power[1:] * dt) / duration)
    return report, gap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recording", type=Path)
    p.add_argument("--output", type=Path, help="Default: recording/successful_feet_audit.json")
    args = p.parse_args()
    results = json.loads((args.recording / "results.json").read_text())
    from bicycle_sim.rl.rider_recording import episode_is_successful

    ids = [r["env"] for r in results["episodes"] if episode_is_successful(r)]
    if not ids:
        p.error("No 20-second survival episodes to audit")
    if not (args.recording / "cohort_trajectory.npz").exists():
        p.error("Re-record with --record-all-envs --first-episodes-only")
    m = mujoco.MjModel.from_xml_path(str((args.recording / "scene.xml").resolve()))
    with np.load(args.recording / "cohort_trajectory.npz") as data:
        cohort = {k: data[k] for k in data.files}
    records = []
    for env_id in ids:
        track = extract_cohort_episode(cohort, env_id)
        report, _ = audit_feet(m, track)
        records.append(dict(env=env_id, **report))
    path = args.output or args.recording / "successful_feet_audit.json"
    path.write_text(
        json.dumps({"successful_episodes": len(records), "episodes": records}, indent=2) + "\n"
    )
    print(path)
    for r in records:
        print(
            json.dumps(
                {
                    "env": r["env"],
                    "left_near_fraction": r["left"]["near_pedal_fraction"],
                    "right_near_fraction": r["right"]["near_pedal_fraction"],
                    "left_max_gap_m": r["left"]["max_signed_gap_m"],
                    "right_max_gap_m": r["right"]["max_signed_gap_m"],
                    "crank_net_revolutions": r["crank_net_revolutions"],
                }
            )
        )


if __name__ == "__main__":
    main()
