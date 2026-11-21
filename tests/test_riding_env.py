"""Goal/state conventions, actuator boundaries and deterministic CPU rollouts."""

import json
from pathlib import Path

import numpy as np
import pytest

from bicycle_sim.rl.bicycle_env import BicycleEnv

ROOT = Path(__file__).resolve().parents[1]
BIKE = json.loads((ROOT / "configs/bicycle_v0.json").read_text())
CFG = json.loads((ROOT / "configs/rl/bicycle_a.json").read_text())


def test_world_velocity_and_heading_are_consistent_at_reset():
    e = BicycleEnv(BIKE, CFG, seed=17)
    state = e.state()
    heading = state["heading"]
    np.testing.assert_allclose(
        state["velocity_world"][:2],
        CFG["initial_speed_mps"] * np.array([np.cos(heading), np.sin(heading)]),
        atol=1e-8,
    )
    assert e.observe().shape == (25,)
    assert e.model.nu == 3
    assert e.model.neq == 0


def test_zero_crank_command_allows_coasting_without_backdrive():
    e = BicycleEnv(BIKE, CFG)
    e.data.qvel[e.v["crank_spin"]] = 0
    e.step([-1, 0])
    assert e.data.actuator_force[e.freewheel_act] == 0
    assert abs(e.data.qvel[e.v["crank_spin"]]) < 0.01
    assert e.data.qvel[e.v["rear_wheel_spin"]] > 0


def test_seeded_rollout_repeats_and_actions_are_finite_and_bounded():
    a, b = BicycleEnv(BIKE, CFG, seed=9), BicycleEnv(BIKE, CFG, seed=9)
    for i in range(10):
        action = [np.sin(i), 0.1 * np.cos(i)]
        oa, ra, ta, xa, _ = a.step(action)
        ob, rb, tb, xb, _ = b.step(action)
        np.testing.assert_array_equal(oa, ob)
        assert (ra, ta, xa) == (rb, tb, xb)
    with pytest.raises(ValueError):
        a.step([np.nan, 0])


def test_fall_is_failure_not_timeout():
    e = BicycleEnv(BIKE, CFG)
    e.data.qpos[3:7] = [np.cos(0.9 / 2), np.sin(0.9 / 2), 0, 0]
    _, _, terminated, truncated, _ = e.step([-1, 0])
    assert terminated and not truncated


def test_command_resampling_and_timeout():
    cfg = dict(CFG, command_seconds=0.04, episode_seconds=0.06, initial_roll_rad=0)
    e = BicycleEnv(BIKE, cfg)
    old = e.command.copy()
    e.step([-1, 0])
    np.testing.assert_array_equal(e.command, old)
    e.step([-1, 0])
    assert not np.array_equal(e.command, old)
    _, _, terminated, truncated, _ = e.step([-1, 0])
    assert truncated and not terminated


def test_velocity_penalty_still_distinguishes_large_tracking_errors():
    # Exponential tracking alone saturates near zero; the added term continues
    # to penalize speeding up beyond the goal, a failure found in pilot training.
    cfg = dict(CFG, velocity_error_penalty=0)
    a, b = BicycleEnv(BIKE, CFG, seed=7), BicycleEnv(BIKE, cfg, seed=7)
    a.data.qvel[:2] *= 2
    b.data.qvel[:2] *= 2
    _, ra, _, _, info = a.step([0, 0])
    _, rb, _, _, _ = b.step([0, 0])
    assert ra == pytest.approx(
        rb - a.dt * CFG["velocity_error_penalty"] * info["speed_error_mps"] ** 2
    )
