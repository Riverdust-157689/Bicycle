"""Loaded coast-down, non-propulsive contacts and continued pedaling capability."""

import numpy as np
import pytest

from bicycle_sim.cli.audit_coasting import build_coast_bench, free_lean, run_coast, run_coast_warp

MASS = 43.3942352


def test_rolling_resistance_removes_nearly_lossless_coast_and_can_be_overcome():
    old = run_coast(build_coast_bench(mass_kg=MASS))
    high_sliding = run_coast(build_coast_bench(mass_kg=MASS, sliding_mu=1.6))
    rolling = run_coast(build_coast_bench(mass_kg=MASS, rolling_m=0.00405))
    fine = run_coast(build_coast_bench(mass_kg=MASS, rolling_m=0.00405, dt=0.001))
    driven = run_coast(build_coast_bench(mass_kg=MASS, rolling_m=0.00405), torque_nm=4)
    assert old["speed_at_20s_mps"] > 1.7
    assert abs(old["speed_at_20s_mps"] - high_sliding["speed_at_20s_mps"]) < 0.001
    assert rolling["speed_at_20s_mps"] < 0.05
    assert 12 < rolling["time_below_0_1mps_s"] < 17
    assert abs(rolling["speed_at_10s_mps"] - fine["speed_at_10s_mps"]) < 0.001
    assert driven["speed_at_20s_mps"] > 2
    for result in (old, rolling, fine, driven):
        assert result["maximum_clutch_power_W"] <= 1e-8
        assert result["solver_warnings"] == 0
    assert rolling["maximum_clutch_rear_torque_Nm"] < 1e-8


def test_contact_resistance_does_not_brake_airborne_motion_or_create_propulsion():
    airborne = run_coast(
        build_coast_bench(mass_kg=MASS, rolling_m=0.00405, airborne=True), seconds=2
    )
    rest = run_coast(build_coast_bench(mass_kg=MASS, rolling_m=0.00405), speed=0, seconds=2)
    reverse = run_coast(build_coast_bench(mass_kg=MASS, rolling_m=0.00405), speed=-2)
    assert airborne["samples"][-1]["speed_mps"] == pytest.approx(2, abs=1e-8)
    assert airborne["contact_work_J"] == 0
    assert abs(rest["distance_m"]) < 1e-6
    assert abs(reverse["speed_at_20s_mps"]) < 0.05
    assert reverse["maximum_clutch_power_W"] <= 1e-8


def test_loaded_stationary_bike_is_not_held_up_by_rolling_contact():
    for rolling in (0, 0.00405):
        for initial_roll in (0.001, 0.02):
            result = free_lean(mass_kg=MASS, rolling_m=rolling, initial_roll=initial_roll)
            assert result["exceeded_20deg"]
            assert not result["solver_warnings"]


def test_warp_contact_backend_matches_native_coast_without_zero_friction_warning(monkeypatch):
    import warp as wp
    from mujoco_warp._src import collision_convex, collision_flex

    monkeypatch.setattr(wp.config, "kernel_cache_dir", "/tmp/bicycle-warp-cache")
    monkeypatch.setattr(collision_convex, "MJ_MAX_EPAHORIZON", 24)
    monkeypatch.setattr(collision_flex, "MJ_MAX_EPAHORIZON", 24)
    with wp.ScopedDevice("cpu"):
        m = build_coast_bench(mass_kg=MASS, rolling_m=0.00405)
        native = run_coast(m, seconds=0.2)
        warp = run_coast_warp(m, seconds=0.2, device="cpu")
    np.testing.assert_allclose(
        [s["speed_mps"] for s in warp["samples"]],
        [s["speed_mps"] for s in native["samples"]],
        atol=1e-5,
    )
