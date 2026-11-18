"""Geometric acceptance for a palm-down, lightly touching initial hand pose."""

import mujoco
import numpy as np


def audit_hands(model, data):
    """Evaluate actual collision geoms; do not change collision masks or physics."""
    steering = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "steering")
    bar_geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == steering and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    results = {}
    for side, sign in (("left", -1), ("right", 1)):
        hand_geoms = []
        palm_geom = None
        for g in range(model.ngeom):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
            if body.startswith((f"{side}_hand_", f"{side}_wrist_")):
                hand_geoms.append(g)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
                if mesh == f"{side}_hand_palm_link":
                    palm_geom = g
        if palm_geom is None or not hand_geoms or not bar_geoms:
            raise ValueError(f"Missing collision geometry for {side}")
        minimum = min(
            mujoco.mj_geomDistance(model, data, h, b, 0.25, None)
            for h in hand_geoms
            for b in bar_geoms
        )
        grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_grip_surface")
        palm_gap = mujoco.mj_geomDistance(model, data, palm_geom, grip, 0.25, None)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp_reference")
        normal = data.site_xmat[site].reshape(3, 3) @ np.array([0.0, sign, 0.0])
        error = float(np.rad2deg(np.arccos(np.clip(-normal[2], -1, 1))))
        results[side] = {
            "minimum_hand_bar_distance_m": float(minimum),
            "palm_grip_distance_m": float(palm_gap),
            "palm_down_error_deg": error,
            "passed": bool(minimum >= -1e-6 and -1e-6 <= palm_gap <= 0.0002 and error <= 0.1),
        }
    return results
