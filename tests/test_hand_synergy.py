import mujoco
import numpy as np
import pytest

from bicycle_sim.cli.probe_hand_grasp import (
    build_contact_rig,
    collect_finger_contacts,
    set_pregrasp_pose,
)
from bicycle_sim.control.hand_synergy import FingerSynergy


def test_contact_rig_removes_only_palm_connections():
    m, _, _ = build_contact_rig()
    assert m.neq == 1
    assert m.equality(0).name == "diagnostic_seat_connection"
    for side in ("left", "right"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"{side}_ideal_grasp") == -1
        assert m.geom(f"{side}_grip_surface").contype[0] != 0


def test_imported_contacts_use_canonical_geom_ids_when_legacy_fields_are_unset():
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom name="left_grip_surface" type="plane" size="1 1 .1"/>
    <geom name="right_grip_surface" type="sphere" size=".01" pos="3 0 3"/>
    <body name="left_hand_index_0_link" pos="0 0 .05"><freejoint/>
    <geom type="sphere" size=".1" mass=".1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    assert data.ncon == 1
    canonical = data.contact.geom.copy()
    # Reproduce get_data_into: valid canonical IDs, legacy fields remain zero.
    for contact in data.contact:
        contact.geom1 = contact.geom2 = 0
    np.testing.assert_array_equal(data.contact.geom, canonical)
    contacts, deepest, raw = collect_finger_contacts(m, data)
    assert contacts["left_index"]["count"] == 1
    assert contacts["left_index"]["normal_force_N"] > 0
    assert deepest < 0
    np.testing.assert_array_equal(raw[0]["geom"], canonical[0])
    contacts, _, _ = collect_finger_contacts(
        m, data, force_override=np.array([[23.0, 0, 0, 0, 0, 0]])
    )
    assert contacts["left_index"]["normal_force_N"] == 23
    with pytest.raises(ValueError):
        collect_finger_contacts(m, data, force_override=np.empty((0, 6)))


def test_six_inputs_move_mirrored_fingers_without_raising_motor_limits():
    m, q0, _ = build_contact_rig()
    gains, torque_limits = m.actuator_gainprm.copy(), m.jnt_actfrcrange.copy()
    control = FingerSynergy(m, q0)
    targets = control.targets(np.ones(6))
    np.testing.assert_allclose(targets[:7], -targets[7:], atol=1e-6)
    assert np.all(targets >= control.limits[:, 0])
    assert np.all(targets <= control.limits[:, 1])
    np.testing.assert_array_equal(m.jnt_actfrcrange, torque_limits)
    other = np.setdiff1d(np.arange(m.nu), control.aids)
    np.testing.assert_array_equal(m.actuator_gainprm[other], gains[other])
    # Changing one input affects only its own finger flexion joints.
    u = np.zeros(6)
    u[1] = 1
    changed = abs(control.targets(u) - control.targets(np.zeros(6))) > 1e-8
    assert np.all(control.columns[changed] == 1)
    assert changed.sum() == 2


def test_finger_closure_is_rate_limited_and_invalid_input_does_not_mutate_state():
    m, q0, _ = build_contact_rig()
    control = FingerSynergy(m, q0, closure_rate=0.5)
    d = mujoco.MjData(m)
    control.apply(d, np.ones(6), 0.1)
    np.testing.assert_allclose(control.closure, 0.05)
    old_closure, old_ctrl = control.closure.copy(), d.ctrl.copy()
    with pytest.raises(ValueError):
        control.apply(d, [np.nan] * 6, 0.1)
    np.testing.assert_array_equal(control.closure, old_closure)
    np.testing.assert_array_equal(d.ctrl, old_ctrl)


def test_pregrasp_shift_preserves_palm_orientation_and_does_not_move_finger_joints():
    m, q0, ctrl = build_contact_rig()
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    mujoco.mj_kinematics(m, d)
    sites = [m.site(f"{s}_grasp_reference").id for s in ("left", "right")]
    positions, rotations = d.site_xpos[sites].copy(), d.site_xmat[sites].copy()
    finger_q = [m.jnt_qposadr[j] for j in range(m.njnt) if "_hand_" in m.joint(j).name]
    old_fingers = q0[finger_q].copy()
    assert set_pregrasp_pose(m, q0, ctrl, 0.005) < 1e-5
    d.qpos[:] = q0
    mujoco.mj_kinematics(m, d)
    np.testing.assert_allclose(
        np.linalg.norm(d.site_xpos[sites] - positions, axis=1), 0.005, atol=1e-5
    )
    np.testing.assert_allclose(d.site_xmat[sites], rotations, atol=1e-4)
    np.testing.assert_array_equal(q0[finger_q], old_fingers)
