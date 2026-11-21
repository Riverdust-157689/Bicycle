"""Plot seat bench traces and exact convex-pelvis center cross sections."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from scipy.spatial import ConvexHull


def section(vertices, x):
    points = []
    for triangle in ConvexHull(vertices).simplices:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = vertices[triangle[i]], vertices[triangle[j]]
            if (a[0] - x) * (b[0] - x) < 0:
                p = a + (b - a) * ((x - a[0]) / (b[0] - a[0]))
                points.append(p[1:])
    points = np.array(points)
    return points[ConvexHull(points).vertices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    args = parser.parse_args()
    choices = {
        "original": "Original seat: 17.6 mm initial gap",
        "raised_box": "Flat seat raised 17.1 mm",
        "paired_round_pads": "Two rounded pads (alternative)",
    }
    choices = {
        n: label
        for n, label in choices.items()
        if (args.audit / f"{n}_pedaling" / "results.json").exists()
    }
    if not choices:
        parser.error("No supported pedaling case found")
    report = json.loads((args.audit / "results.json").read_text())
    fig, axes = plt.subplots(
        2, len(choices), figsize=(4.5 * len(choices), 8), layout="constrained", squeeze=False
    )
    for col, (name, label) in enumerate(choices.items()):
        case = args.audit / f"{name}_pedaling"
        m = mujoco.MjModel.from_xml_path(str((case / "scene.xml").resolve()))
        track = np.load(case / "trajectory.npz")
        d = mujoco.MjData(m)
        d.qpos[:] = track["qpos"][0]
        mujoco.mj_forward(m, d)
        pelvis = m.body("pelvis").id
        g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == pelvis and m.geom_contype[g])
        mesh = m.geom_dataid[g]
        start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        v = m.mesh_vert[start : start + count] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        ax = axes[0, col]
        polygon = section(v, 0.24) * 1000
        ax.add_patch(
            Polygon(
                polygon,
                facecolor="#87b4d5",
                edgecolor="#305b7a",
                alpha=0.8,
                label="Pelvis collision hull",
            )
        )
        for g in range(m.ngeom):
            if not m.geom(g).name.startswith("saddle"):
                continue
            p, size = d.geom_xpos[g] * 1000, m.geom_size[g] * 1000
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                patch = Rectangle((p[1] - size[1], p[2] - size[2]), size[1] * 2, size[2] * 2)
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                patch = Circle((p[1], p[2]), size[0])
            else:
                patch = Ellipse((p[1], p[2]), size[1] * 2, size[2] * 2)
            patch.set(facecolor="#b98357", edgecolor="#6b4324", alpha=0.9)
            ax.add_patch(patch)
        ax.set(
            title=label,
            xlabel="Lateral y (mm)",
            ylabel="Height z (mm)",
            xlim=(-85, 85),
            ylim=(715, 920),
            aspect="equal",
        )
        ax.grid(alpha=0.2)
        ax = axes[1, col]
        result = json.loads((case / "results.json").read_text())
        trace = result["trace"]
        t = [s["time_s"] for s in trace]
        ax.plot(t, [s["seat_force_z_N"] for s in trace], label="Seat vertical support")
        ax.axhline(
            result["summary"]["robot_weight_N"], color="gray", linestyle="--", label="Robot weight"
        )
        ax.set(xlabel="Time (s)", ylabel="Force (N)", ylim=(-10, 360))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"{report['backend']} bench: fixed bicycle, no seat/palm attachment; nominal PD + driven crank\n"
        "Top: initial center cross section x=0.24 m (true scale). Bottom: pedaling contact force."
    )
    fig.savefig(args.audit / "seat_comparison.png", dpi=160)
    print(args.audit / "seat_comparison.png")


if __name__ == "__main__":
    main()
