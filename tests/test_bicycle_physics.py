"""Acceptance checks for the standalone prototype, not for rider performance."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from bicycle_sim.models.bicycle import build_bicycle, generate

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/bicycle_v0.json").read_text())


def model(*, fixed=False, airborne=False):
    root = build_bicycle(CONFIG, fixed_frame=fixed)
    if airborne:
        ground = root.find("./worldbody/geom[@name='ground']")
        root.find("worldbody").remove(ground)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def index(m, name):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert j >= 0
    return m.jnt_qposadr[j], m.jnt_dofadr[j]


def drive(dt):
    m = model(fixed=True, airborne=True)
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    rear, _ = index(m, "rear_wheel_spin")
    crank, dof = index(m, "crank_spin")
    for _ in range(round(1 / dt)):
        d.qfrc_applied[dof] = 1.0  # diagnostic crank torque, never a rear-wheel motor
        mujoco.mj_step(m, d)
    return m, d, float(d.qpos[rear]), float(d.qpos[crank])


def test_generation_is_deterministic(tmp_path):
    config_path = ROOT / "configs/bicycle_v0.json"
    paths = generate(config_path, tmp_path)
    first = [p.read_bytes() for p in paths]
    assert first == [p.read_bytes() for p in generate(config_path, tmp_path)]


def test_compile_mass_and_actuation():
    m = model()
    assert (m.nq, m.nv, m.nu) == (13, 12, 1)
    assert m.actuator_gainprm[0, 0] == 0  # unpowered freewheel, no wheel motor
    assert m.body_mass.sum() == pytest.approx(9.0)
    assert np.all(m.body_inertia[1:] > 0)


def test_loaded_robot_snapshot_compiles():
    m = mujoco.MjModel.from_xml_path(str(ROOT / "assets/g1_dex3/g1_with_hands.xml"))
    assert (m.nq, m.nv, m.nu) == (50, 49, 43)


def test_only_tires_touch_floor_at_rest():
    m = model()
    d = mujoco.MjData(m)
    for _ in range(50):
        mujoco.mj_step(m, d)
    pairs = {
        tuple(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (c.geom1, c.geom2))
        for c in d.contact
    }
    assert pairs == {("ground", "rear_wheel_tire"), ("ground", "front_wheel_tire")}
    assert max(-c.dist for c in d.contact) < 0.002


def test_fixed_diagnostic_has_no_bearing_self_collisions():
    m = model(fixed=True, airborne=True)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert d.ncon == 0


def test_crank_torque_drives_rear_wheel_with_correct_ratio():
    _m, d, rear, crank = drive(0.002)
    assert crank > 0.1 and rear > crank
    _, cr = index(_m, "crank_spin")
    _, rw = index(_m, "rear_wheel_spin")
    assert 0 <= 2 * d.qvel[cr] - d.qvel[rw] < 1.0  # compliant clutch slip
    assert d.ncon == 0
    assert np.isfinite(d.qpos).all()
    assert not d.warning.number.any()


def test_transmission_step_size_sensitivity():
    _, _, rear_coarse, crank_coarse = drive(0.002)
    _, _, rear_fine, crank_fine = drive(0.001)
    # Compliant one-way damper: current 2 ms -> 1 ms angle difference is 2.3%.
    assert rear_coarse == pytest.approx(rear_fine, rel=0.03)
    assert crank_coarse == pytest.approx(crank_fine, rel=0.03)


def test_initial_lean_is_not_artificially_stabilized():
    m = model()
    d = mujoco.MjData(m)
    angle = np.deg2rad(8)
    d.qpos[3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all()
    # Body's vertical axis departs substantially from world up after an uncorrected lean.
    frame = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bike_frame")
    assert d.xmat[frame].reshape(3, 3)[2, 2] < np.cos(np.deg2rad(20))
    assert not d.warning.number.any()


def test_pedal_bearings_are_independent_and_platforms_outside_arms():
    m = model(fixed=True, airborne=True)
    d = mujoco.MjData(m)
    crank, _ = index(m, "crank_spin")
    left, _ = index(m, "left_pedal_spin")
    right, _ = index(m, "right_pedal_spin")
    for phase in np.linspace(0, 2 * np.pi, 25):
        d.qpos[crank] = phase
        # Kinematic reference only: an unactuated pedal is not constrained horizontal.
        d.qpos[left] = d.qpos[right] = -phase
        mujoco.mj_forward(m, d)
        for side, sign in (("left", 1), ("right", -1)):
            site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_pedal_contact")
            expected = np.array(CONFIG["geometry"]["bottom_bracket"]) + [
                sign * CONFIG["geometry"]["crank_radius"] * np.cos(phase),
                sign * CONFIG["geometry"]["pedal_lateral_offset"],
                -sign * CONFIG["geometry"]["crank_radius"] * np.sin(phase)
                + CONFIG["geometry"]["pedal_half_size"][2],
            ]
            np.testing.assert_allclose(d.site_xpos[site], expected, atol=1e-8)
            np.testing.assert_allclose(d.site_xmat[site].reshape(3, 3), np.eye(3), atol=1e-8)
    # Rotating the left platform alone must not move either bearing or the right platform.
    right_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_pedal_contact")
    before = d.site_xpos[right_site].copy()
    d.qpos[left] += 0.4
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(d.site_xpos[right_site], before, atol=1e-8)


def test_downward_force_on_left_platform_transmits_to_crank_and_wheel():
    m = model(fixed=True, airborne=True)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_pedal_contact")
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_pedal")
    _, crank_dof = index(m, "crank_spin")
    rear_q, _ = index(m, "rear_wheel_spin")
    generalized_force = np.zeros(m.nv)
    mujoco.mj_applyFT(
        m, d, np.array([0.0, 0.0, -50.0]), np.zeros(3), d.site_xpos[site], body, generalized_force
    )
    assert generalized_force[crank_dof] == pytest.approx(50 * CONFIG["geometry"]["crank_radius"])
    # Velocity-dependent engagement is evaluated after the crank first accelerates.
    d.qfrc_applied[:] = generalized_force
    mujoco.mj_step(m, d, nstep=2)
    assert d.qpos[rear_q] > 0


def test_head_tube_visible_axis_matches_actual_steering_axis_and_trail():
    root = build_bicycle(CONFIG)
    m = model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "steer")
    tube = root.find(".//geom[@name='head_tube']")
    ends = np.fromstring(tube.get("fromto"), sep=" ").reshape(2, 3)
    visible_axis = ends[1] - ends[0]
    visible_axis /= np.linalg.norm(visible_axis)
    np.testing.assert_allclose(visible_axis, d.xaxis[j], atol=1e-8)
    intersection = d.xanchor[j] - d.xaxis[j] * d.xanchor[j, 2] / d.xaxis[j, 2]
    assert intersection[0] - CONFIG["geometry"]["front_axle"][0] == pytest.approx(
        CONFIG["geometry"]["trail"]
    )


def test_direct_bar_mount_has_no_adapter_and_preserves_grip_positions():
    root = build_bicycle(CONFIG)
    g = CONFIG["geometry"]
    if g.get("handlebar_mount_mode") != "direct":
        pytest.skip("Configuration explicitly requests a conventional stem")
    assert root.find(".//geom[@name='stem']") is None
    column = root.find(".//geom[@name='steering_column']")
    endpoint = np.fromstring(column.get("fromto"), sep=" ")[3:]
    branch = root.find(".//geom[@name='left_handlebar_sweep']")
    start = np.fromstring(branch.get("fromto"), sep=" ")[:3]
    np.testing.assert_allclose(endpoint, start, atol=1e-8)
    m = model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for side, target in zip(("left", "right"), g["grip_centers"], strict=True):
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, side + "_grip")
        np.testing.assert_allclose(d.site_xpos[site], target, atol=1e-8)


def test_seat_post_endcap_stays_below_saddle_underside():
    root = build_bicycle(CONFIG)
    post = root.find(".//geom[@name='seat_post']")
    ends = np.fromstring(post.get("fromto"), sep=" ").reshape(2, 3)
    top = ends[:, 2].max() + float(post.get("size"))
    underside = CONFIG["geometry"]["saddle_top"][2] - 2 * CONFIG["geometry"]["saddle_half_size"][2]
    assert top <= underside + 1e-9


def test_tire_contact_has_explicit_effective_friction_and_compliance():
    m = model()
    d = mujoco.MjData(m)
    for _ in range(50):
        mujoco.mj_step(m, d)
    assert d.ncon > 0
    for contact in d.contact:
        assert contact.dim == 3
        np.testing.assert_allclose(contact.friction[:2], [0.8, 0.8])
        np.testing.assert_allclose(contact.solref, [0.02, 1])
        np.testing.assert_allclose(contact.solimp, [0.9, 0.95, 0.001, 0.5, 2])


def test_production_tire_is_one_rounded_mesh_per_wheel():
    m = model()
    for name in ("rear_wheel", "front_wheel"):
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        geoms = np.flatnonzero(m.geom_bodyid == body)
        assert len(geoms) == 1
        assert m.geom_type[geoms[0]] == mujoco.mjtGeom.mjGEOM_MESH
    assert m.nmesh == 2


def test_tiny_lean_grows_without_control():
    from bicycle_sim.cli.audit_physics import lean

    result = lean(model(), 0.1, duration=3)
    assert result["time_to_30deg_s"] is not None
    assert not any(result["warnings"])
