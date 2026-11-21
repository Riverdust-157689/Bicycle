"""Check the delivered mounted keyframe against enabled hand/bar collision geometry."""

from pathlib import Path

import mujoco

from bicycle_sim.control.hand_contact import audit_hands

ROOT = Path(__file__).resolve().parents[1]


def mounted():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/scenes/rider_fit_diagnostic.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def test_initial_palms_face_down_without_hand_or_finger_penetration():
    model, data = mounted()
    result = audit_hands(model, data)
    assert all(h["passed"] for h in result.values())
    assert all(0 <= h["minimum_hand_bar_distance_m"] <= 0.0002 for h in result.values())
    for contact in data.contact:
        bodies = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
            for g in (contact.geom1, contact.geom2)
        ]
        if "steering" in bodies:
            assert contact.dist >= -1e-6


def test_hand_acceptance_detects_downward_penetration():
    model, data = mounted()
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    data.qpos[model.jnt_qposadr[root] + 2] -= 0.01
    mujoco.mj_forward(model, data)
    result = audit_hands(model, data)
    assert not any(h["passed"] for h in result.values())
    assert all(h["minimum_hand_bar_distance_m"] < -0.005 for h in result.values())


def test_mounted_position_servos_start_at_pose_reference():
    model, data = mounted()
    import numpy as np

    joint_actuators = model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
    expected = data.qpos[model.jnt_qposadr[model.actuator_trnid[joint_actuators, 0]]]

    np.testing.assert_allclose(data.ctrl[joint_actuators], expected, atol=1e-8)
    assert data.ctrl[model.actuator("freewheel").id] == 0
