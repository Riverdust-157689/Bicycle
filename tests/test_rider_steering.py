"""Assisted stand verifies arm-to-bar force transmission, not riding ability."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from bicycle_sim.cli.check_rider_steering import audit, build_rig


def test_robot_arms_turn_handlebar_without_a_steering_actuator():
    m, _, _ = build_rig()
    steer = m.joint("steer").id
    assert not any(
        m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == steer
        for a in range(m.nu)
    )
    report = audit()
    assert report["passed"]
    assert not report["free_riding_validated"]
    assert all(r["peak_arm_generalized_torque_Nm"] > 0 for r in report["results"])


def test_wider_controller_targets_are_reachable_with_existing_motor_limits():
    from bicycle_sim.cli.audit_controller_limits import audit as audit_limits

    cfg = json.loads(Path("configs/rl/bicycle_b.json").read_text())
    report = audit_limits(cfg)
    assert report["passed"]
    assert all(r["ik_error_m"] < 0.001 for r in report["results"])
    assert all(r["last_window_deepest_contact_m"] > -0.005 for r in report["results"])
    np.testing.assert_allclose([r["target_rad"] for r in report["results"]], [-0.35, 0, 0.35])


def test_ik_rejects_targets_outside_the_physical_steering_range():
    from bicycle_sim.cli.diagnose_rider import arm_references

    with pytest.raises(ValueError):
        arm_references(0.5, 0.5)
    with pytest.raises(ValueError):
        arm_references(0.35, -0.1)


def test_gpu_trace_analysis_aligns_feedback_with_the_previous_frame():
    from bicycle_sim.cli.analyze_rider_probe import summarize

    target = np.array([np.nan, 0, 0.03, 0.3, 0.35])
    track = {
        "episode_id": np.zeros(5, dtype=int),
        "time_s": np.arange(5) * 0.2,
        "desired_steer_rad": target,
        "actual_steer_rad": np.nan_to_num(target),
        "roll_rad": np.array([0, -0.01, -0.1, -0.2, -0.3]),
        "roll_rate_rad_s": np.zeros(5),
        "speed_mps": np.ones(5) * 2,
        "motor_torque_Nm": np.ones(5),
        "terminated": np.array([False, False, False, False, True]),
        "truncated": np.zeros(5, dtype=bool),
    }
    report, _ = summarize(track, {"steering_reference_limit_rad": 0.35, "max_crank_torque_Nm": 8})
    assert report["feedback_formula_max_error_rad"] < 1e-12
    assert report["tracking_rms_rad"] == 0
    assert report["first_reference_saturation_s"] == pytest.approx(0.8)
    assert report["terminated"]  # Execution and a failed episode can coexist.
    track["desired_steer_rad"][:] = np.nan
    report, _ = summarize(
        track,
        {"steering_reference_limit_rad": 0.35, "max_crank_torque_Nm": 8},
        controller="sac_mean_policy",
    )
    # A direct arm policy has no requested steering angle: do not fabricate
    # steering tracking error or saturation from the unused PD diagnostic.
    assert report["feedback_formula_max_error_rad"] is None
    assert report["tracking_rms_rad"] is None
    assert report["reference_saturation_fraction"] is None
