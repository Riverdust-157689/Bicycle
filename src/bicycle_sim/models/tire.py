"""Rounded tire outer-envelope samples. MuJoCo uses the convex hull for collisions."""

import math


def set_wheel_ground_rolling_contact(root, model, rolling_friction_m):
    """Add rolling resistance only at the two tire/ground contact pairs.

    MuJoCo's rolling coefficient has units of metres: torque <= b * normal
    force, approximately Crr = b / wheel radius. Retain the existing tire's
    sliding and normal-contact parameters. Other robot contacts stay unchanged.
    Zero deliberately retains the historical condim=3 model.
    """
    import xml.etree.ElementTree as ET

    if not math.isfinite(rolling_friction_m) or rolling_friction_m < 0:
        raise ValueError("wheel_ground_rolling_friction_m must be finite and nonnegative")
    if rolling_friction_m == 0:
        return
    if rolling_friction_m < 1e-5:
        raise ValueError("Use zero to disable rolling contact, or a coefficient >= 1e-5 m")
    ground = model.geom("ground").id
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for name in ("rear_wheel_tire", "front_wheel_tire"):
        tire = model.geom(name).id
        if model.geom_priority[tire] <= model.geom_priority[ground]:
            raise ValueError("Rolling contact adapter expects the tire to have higher priority")
        mu = model.geom_friction[tire, 0]
        # All five friction dimensions are active for condim=6. MuJoCo-Warp
        # warns that an exact zero torsional coefficient can produce NaNs.
        torsion = max(float(model.geom_friction[tire, 1]), 1e-5)
        ET.SubElement(
            contact,
            "pair",
            name=f"{name}_ground_rolling",
            geom1="ground",
            geom2=name,
            condim="6",
            friction=f"{mu:.12g} {mu:.12g} {torsion:.12g} {rolling_friction_m:.12g} {rolling_friction_m:.12g}",
            solref=" ".join(map(str, model.geom_solref[tire])),
            solimp=" ".join(map(str, model.geom_solimp[tire])),
            margin=str(model.geom_margin[tire]),
            gap=str(model.geom_gap[tire]),
        )


def set_ellipsoid_tire_geoms(root, model):
    """Replace mounted mesh tires in XML; keep body-frame radius and half width.

    Explicit inertias, contact settings, joints and compiled source stay intact.
    Callers recompile the XML so collision bounds and types are updated together.
    """
    import mujoco
    import numpy as np

    for name in ("rear_wheel_tire", "front_wheel_tire"):
        g = model.geom(name).id
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Ellipsoid conversion expects the original mesh tires")
        mesh = model.geom_dataid[g]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = (
            model.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T + model.geom_pos[g]
        )
        bounds = np.max(abs(vertices), axis=0)
        radius, width = max(bounds[0], bounds[2]), bounds[1]
        geom = root.find(f".//geom[@name='{name}']")
        geom.attrib.pop("mesh")
        geom.set("type", "ellipsoid")
        geom.set("size", f"{radius:.12g} {width:.12g} {radius:.12g}")


def rounded_envelope_vertices(radius, half_width, circumferential, crown):
    if not (radius > half_width > 0 and circumferential >= 12 and crown >= 8):
        raise ValueError("Invalid rounded tire dimensions or sampling")
    if circumferential % 4 or crown % 4:
        raise ValueError("Sampling counts must be divisible by four to preserve axis extrema")
    major = radius - half_width
    points = []
    for i in range(circumferential):
        a = 2 * math.pi * i / circumferential
        for j in range(crown):
            b = 2 * math.pi * j / crown
            rho = major + half_width * math.cos(b)
            points.extend((rho * math.cos(a), half_width * math.sin(b), rho * math.sin(a)))
    return points
