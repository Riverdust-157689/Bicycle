"""Build a primitive bicycle. No simulator dependency is needed for generation."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from bicycle_sim.models.tire import rounded_envelope_vertices


def numbers(values):
    return " ".join(f"{float(v):.10g}" for v in values)


def sub(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def relative(point, origin):
    return [p - o for p, o in zip(point, origin, strict=True)]


def inertial(body, mass, half_sizes, pos=(0, 0, 0)):
    # Equivalent box inertia: intentionally approximate for non-wheel parts.
    x, y, z = half_sizes
    sub(
        body,
        "inertial",
        pos=numbers(pos),
        mass=mass,
        diaginertia=numbers(
            [mass * (y * y + z * z) / 3, mass * (x * x + z * z) / 3, mass * (x * x + y * y) / 3]
        ),
    )


def build_bicycle(config: dict, *, fixed_frame: bool = False) -> ET.Element:
    g, masses = config["geometry"], config["mass_targets_kg"]
    sim, drive = config["simulation_initial"], config["transmission"]
    radius, width = g["wheel_radius"], g["wheel_half_width"]
    rear, front, bb = g["rear_axle"], g["front_axle"], g["bottom_bracket"]
    if radius <= 0 or width <= 0 or g["crank_radius"] <= 0:
        raise ValueError("Wheel and crank sizes must be positive")
    if not math.isclose(front[0] - rear[0], g["wheelbase"]):
        raise ValueError("Wheelbase and axle positions disagree")
    if not all(math.isclose(p[2], radius) for p in (rear, front)):
        raise ValueError("Initial wheel centers must lie one radius above the floor")
    if drive["powered_rear_wheel_in_riding_mode"]:
        raise ValueError("This model deliberately has no rear wheel motor")
    alpha = math.radians(g["head_angle_from_horizontal_deg"])
    head_z = g.get("head_tube_lower_height", 0.52)
    head = [front[0] + g["trail"] - head_z / math.tan(alpha), 0, head_z]
    axis = [-math.cos(alpha), 0, math.sin(alpha)]
    root = ET.Element("mujoco", model="bicycle_fixed_diagnostic" if fixed_frame else "bicycle_free")
    sub(root, "compiler", angle="radian", autolimits="true")
    sub(
        root,
        "option",
        timestep=sim["timestep_s"],
        integrator=sim["integrator"],
        gravity="0 0 -9.81",
        iterations="60",
    )
    contacts = config["contact_parameters"]
    bearings = config["bearing_parameters"]

    def surface(kind):
        values = contacts[kind]
        return {
            "friction": numbers(values["friction"]),
            "priority": values["priority"],
            "condim": values["condim"],
            "solref": numbers(contacts["solref"]),
            "solimp": numbers(contacts["solimp"]),
        }

    visual = sub(root, "visual")
    sub(visual, "quality", numslices="96", numstacks="16")
    default = sub(root, "default")
    sub(default, "geom", density="0", **surface("default"))
    sub(
        default,
        "joint",
        damping=bearings["default_damping"],
        frictionloss=bearings["frictionloss"],
        armature=bearings["armature"],
    )
    world = sub(root, "worldbody")
    sub(world, "light", pos="0 -2 3", dir="0 0 -1")
    sub(world, "camera", name="side", pos="0.4 -2.5 1.2", xyaxes="1 0 0 0 0.3 1")
    sub(
        world,
        "geom",
        name="ground",
        type="plane",
        size="5 5 0.1",
        rgba="0.8 0.8 0.8 1",
        **surface("ground"),
    )
    frame = sub(world, "body", name="bike_frame")
    if not fixed_frame:
        sub(frame, "freejoint", name="bike_root")
    inertial(
        frame,
        masses["frame_including_saddle"],
        [g["wheelbase"] * 0.475, 0.09, (g["saddle_top"][2] - rear[2]) / 2],
        [g["wheelbase"] * 0.425, 0, (g["saddle_top"][2] + rear[2]) / 2],
    )

    def tube(body, name, a, b, color="0.15 0.4 0.75 1", r=0.016):
        sub(body, "geom", name=name, type="capsule", fromto=numbers([*a, *b]), size=r, rgba=color)

    saddle = g["saddle_top"]
    seat_joint = [saddle[0], 0, saddle[2] - 0.07]
    for name, a, b in [
        ("chainstay", rear, bb),
        ("seatstay", rear, seat_joint),
        ("seat_tube", bb, seat_joint),
        ("down_tube", bb, head),
        ("top_tube", seat_joint, head),
        (
            "seat_post",
            seat_joint,
            [saddle[0], saddle[1], saddle[2] - 2 * g["saddle_half_size"][2] - 0.016],
        ),
    ]:
        tube(frame, name, a, b)
    seat_center = [*saddle[:2], saddle[2] - g["saddle_half_size"][2]]
    sub(
        frame,
        "geom",
        name="saddle",
        type="box",
        pos=numbers(seat_center),
        size=numbers(g["saddle_half_size"]),
        rgba="0.15 0.15 0.15 1",
        **surface("saddle"),
    )
    sub(frame, "site", name="saddle_contact", pos=numbers(saddle), size="0.008")
    sub(frame, "site", name="bike_imu", pos="0.35 0 0.4", size="0.008")

    def wheel(parent, name, pos, mass):
        body = sub(parent, "body", name=name, pos=numbers(pos))
        sub(body, "joint", name=f"{name}_spin", axis="0 1 0", limited="false")
        transverse = mass * (radius * radius / 4 + width * width / 3)
        sub(
            body,
            "inertial",
            pos="0 0 0",
            mass=mass,
            diaginertia=numbers([transverse, mass * radius * radius / 2, transverse]),
        )
        profile = g.get("tire_profile", {"type": "cylinder"})
        if profile["type"] == "rounded_convex":
            asset = root.find("asset")
            if asset is None:
                asset = sub(root, "asset")
            mesh_name = f"{name}_rounded_envelope"
            points = rounded_envelope_vertices(
                radius, width, profile["circumferential_samples"], profile["crown_samples"]
            )
            sub(asset, "mesh", name=mesh_name, vertex=numbers(points))
            sub(
                body,
                "geom",
                name=f"{name}_tire",
                type="mesh",
                mesh=mesh_name,
                rgba="0.1 0.1 0.1 1",
                **surface("tire"),
            )
        elif profile["type"] == "cylinder":
            sub(
                body,
                "geom",
                name=f"{name}_tire",
                type="cylinder",
                size=numbers([radius, width]),
                quat=numbers([math.sqrt(0.5), -math.sqrt(0.5), 0, 0]),
                rgba="0.1 0.1 0.1 1",
                **surface("tire"),
            )
        else:
            raise ValueError("Unknown tire profile")
        sub(body, "site", name=f"{name}_axle", size="0.008")
        return body

    wheel(frame, "rear_wheel", rear, masses["rear_wheel"])
    steer = sub(frame, "body", name="steering", pos=numbers(head))
    sub(
        steer,
        "joint",
        name="steer",
        axis=numbers(axis),
        range=numbers([-g["steer_limit_rad"], g["steer_limit_rad"]]),
        damping=bearings["steer_damping"],
    )
    # Render the head tube along the ACTUAL steering axis. Stem and swept bars
    # are separate rigid members, so their silhouette cannot be mistaken for the hinge.
    upper_z = g.get("head_tube_upper_height", head_z + 0.12)
    head_top = [front[0] + g["trail"] - upper_z / math.tan(alpha), 0, upper_z]
    clamp_z = g.get("stem_clamp_height", upper_z)
    column_top = [front[0] + g["trail"] - clamp_z / math.tan(alpha), 0, clamp_z]
    tube(frame, "head_tube", head, head_top, "0.15 0.4 0.75 1", r=0.025)
    tube(steer, "steering_column", [0, 0, 0], relative(column_top, head), "0.4 0.4 0.4 1", r=0.013)
    tube(steer, "fork", [0, 0, 0], relative(front, head))
    grips = g["grip_centers"]
    stem = [grips[0][0] + g.get("handlebar_sweep_back", 0), 0, grips[0][2]]
    if g.get("handlebar_mount_mode", "stem") == "direct":
        if math.dist(column_top, stem) > 1e-6:
            raise ValueError("Direct handlebar mount must coincide with the steering column top")
    else:
        tube(
            steer,
            "stem",
            relative(column_top, head),
            relative(stem, head),
            "0.4 0.4 0.4 1",
            r=0.018,
        )
    half_length = g.get("grip_length", 0.12) / 2
    extent_points = [head, head_top, column_top, front, stem] + [
        [p[0], p[1] + sign * half_length, p[2]] for p, sign in zip(grips, (1, -1), strict=True)
    ]
    lower = [min(p[i] for p in extent_points) for i in range(3)]
    upper = [max(p[i] for p in extent_points) for i in range(3)]
    center = [(lo + hi) / 2 for lo, hi in zip(lower, upper, strict=True)]
    half_sizes = [max((hi - lo) / 2, 0.01) for lo, hi in zip(lower, upper, strict=True)]
    inertial(
        steer, masses["steering_assembly_including_handlebar"], half_sizes, relative(center, head)
    )
    for side, sign, point in zip(("left", "right"), (1, -1), grips, strict=True):
        inner = [point[0], point[1] - sign * half_length, point[2]]
        outer = [point[0], point[1] + sign * half_length, point[2]]
        tube(
            steer,
            f"{side}_handlebar_sweep",
            relative(stem, head),
            relative(inner, head),
            "0.4 0.4 0.4 1",
            r=0.012,
        )
        tube(
            steer,
            f"{side}_grip_surface",
            relative(inner, head),
            relative(outer, head),
            "0.12 0.12 0.12 1",
            r=g["grip_radius"],
        )
        steer.find(f"./geom[@name='{side}_grip_surface']").attrib.update(
            {k: str(v) for k, v in surface("grip").items()}
        )
        sub(steer, "site", name=f"{side}_grip", pos=numbers(relative(point, head)), size="0.008")
    wheel(steer, "front_wheel", relative(front, head), masses["front_wheel"])

    crank = sub(frame, "body", name="crank", pos=numbers(bb))
    sub(crank, "joint", name="crank_spin", axis="0 1 0", limited="false")
    r = g["crank_radius"]
    arm_offset = g["crank_arm_lateral_offset"]
    pedal_offset = g["pedal_lateral_offset"]
    spindle_length = g["pedal_spindle_length"]
    half_width = g["pedal_half_size"][1]
    if not (
        arm_offset < pedal_offset - half_width
        and pedal_offset + half_width <= arm_offset + spindle_length + 1e-9
    ):
        raise ValueError("Pedal platform must lie outside crank arm and on its spindle")
    inertial(crank, masses["crank_assembly"], [r, arm_offset + spindle_length, 0.015])
    for side, sign in (("left", 1), ("right", -1)):
        # One rigid bent crank: lateral axle -> radial arm -> lateral pedal spindle.
        # Bends are geometry, not extra joints. Both arms share one bottom-bracket hinge.
        elbow = [0, sign * arm_offset, 0]
        bearing = [sign * r, sign * arm_offset, 0]
        spindle_end = [sign * r, sign * (arm_offset + spindle_length), 0]
        tube(crank, f"{side}_crank_axle", [0, 0, 0], elbow, r=0.01)
        tube(crank, f"{side}_crank_arm", elbow, bearing, "0.75 0.35 0.12 1", r=0.012)
        tube(crank, f"{side}_pedal_spindle", bearing, spindle_end, "0.8 0.8 0.8 1", r=0.006)
        sub(crank, "site", name=f"{side}_pedal_bearing", pos=numbers(bearing), size="0.007")
        pedal = sub(crank, "body", name=f"{side}_pedal", pos=numbers(bearing))
        # Pedal bearing is parallel to the bottom bracket and rotates independently.
        sub(pedal, "joint", name=f"{side}_pedal_spin", axis="0 1 0", limited="false")
        platform_center = [0, sign * (pedal_offset - arm_offset), 0]
        inertial(pedal, masses["each_pedal"], g["pedal_half_size"], platform_center)
        sub(
            pedal,
            "geom",
            name=f"{side}_pedal_platform",
            type="box",
            pos=numbers(platform_center),
            size=numbers(g["pedal_half_size"]),
            rgba="0.55 0.6 0.65 1",
            **surface("pedal"),
        )
        sub(
            pedal,
            "site",
            name=f"{side}_pedal_contact",
            pos=numbers([0, platform_center[1], g["pedal_half_size"][2]]),
            size="0.008",
        )
    # Avoid collisions inside bearings and at shared rigid connection points.
    contact = sub(root, "contact")
    for a, b in [
        ("bike_frame", "steering"),
        ("bike_frame", "rear_wheel"),
        ("bike_frame", "crank"),
        ("steering", "front_wheel"),
    ]:
        sub(contact, "exclude", body1=a, body2=b)
    equality = sub(root, "equality")
    ratio = drive["rear_wheel_revolutions_per_crank_revolution"]
    sub(
        equality,
        "joint",
        name="chain_coupling",
        joint1="rear_wheel_spin",
        joint2="crank_spin",
        polycoef=numbers([0, ratio, 0, 0, 0]),
    )
    custom = sub(root, "custom")
    sub(custom, "text", name="bicycle_geometry", data=json.dumps(g))
    sensors = sub(root, "sensor")
    for name in ("steer", "rear_wheel_spin", "front_wheel_spin", "crank_spin"):
        sub(sensors, "jointpos", name=f"{name}_position", joint=name)
        sub(sensors, "jointvel", name=f"{name}_velocity", joint=name)
    sub(sensors, "gyro", name="bike_gyro", site="bike_imu")
    sub(sensors, "accelerometer", name="bike_acceleration", site="bike_imu")
    # No actuators: subsequent rider forces or diagnostic generalized forces drive it.
    return root


def generate(config_path: Path, output: Path) -> list[Path]:
    raw = config_path.read_bytes()
    config = json.loads(raw)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fixed in (("scene_free.xml", False), ("scene_fixed_diagnostic.xml", True)):
        tree = ET.ElementTree(build_bicycle(config, fixed_frame=fixed))
        ET.indent(tree, space="  ")
        path = output / name
        tree.write(path, encoding="unicode")
        paths.append(path)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": hashlib.sha256(raw).hexdigest(),
                "status": "generated_prototype_requires_physical_validation",
                "notes": [
                    "Fixed scene is a diagnostic rig, not successful riding.",
                    "Tire contact profile is configured in geometry.tire_profile; inertias held fixed during geometry adoption.",
                    "Bilateral ideal transmission without a freewheel.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return paths
