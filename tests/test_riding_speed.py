"""Reward ordering and terminal task metrics for sustained forward riding."""

import json
from pathlib import Path

import pytest
import torch

from bicycle_sim.cli.probe_rider_gpu import summarize_episodes
from bicycle_sim.rl.rider_gpu_env import task_reward
from bicycle_sim.rl.rider_recording import episode_is_successful
from bicycle_sim.rl.riding_speed import EpisodeSpeedMetrics, validate_speed_objective

OBJECTIVE = json.loads(
    Path("configs/rl/bicycle_b_whole_body_direct_noassist_rolling_speed_sac.json").read_text()
)["speed_objective"]


def test_target_speed_has_more_reward_than_coasting_slowing_down_or_overspeeding():
    speeds = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -2.0])
    velocity = torch.zeros(len(speeds), 3)
    velocity[:, 0] = speeds
    rotation = torch.eye(3).expand(len(speeds), -1, -1)
    action = torch.zeros(len(speeds), 35)
    rate, _ = task_reward(rotation, velocity, torch.zeros_like(velocity), action, action, OBJECTIVE)
    assert rate[4] > rate[3] > rate[2] > rate[1] > rate[0]
    assert rate[4] > rate[5] > rate[6]
    assert rate[0] + 0.1 + 0.1 < 0  # All existing finger/crank bonuses cannot pay for standing.
    assert rate[-1] < 0
    assert rate[2] != pytest.approx(rate[4])  # Old reward saturated above 1 m/s.


def test_survival_alone_or_a_last_moment_speed_spike_is_not_success():
    metric = EpisodeSpeedMetrics(5, 1000, 0.02, OBJECTIVE, "cpu")
    for step in range(1, 1001):
        speed = torch.tensor(
            [
                2.0,  # Sustained riding.
                2.0 * max(0.0, 1 - step / 750),  # Coasts down to a stop.
                0.0 if step < 950 else 2.0,  # Only accelerates at the end.
                2.0 if step <= 950 else 0.0,  # Stops during final second.
                2.0,  # Falls at the time limit.
            ]
        )
        result = metric.update(
            speed, torch.full((5,), step), torch.tensor([step == 1000] * 4 + [False])
        )
    assert result["speed_qualified_success"].tolist() == [True, False, False, False, False]
    assert result["episode_tail_speed_in_band_fraction"][3] == pytest.approx(0.8)
    snapshot = result["episode_speed_in_band_fraction"].clone()
    metric.reset(torch.tensor([1, 3]))
    assert metric.good[1] == metric.good[3] == 0
    assert metric.good[0] == metric.good[4] == 1000
    assert metric.total[1] == metric.tail_count[3] == 0
    torch.testing.assert_close(result["episode_speed_in_band_fraction"], snapshot)


def test_speed_band_accepts_small_transients_without_changing_survival_semantics():
    metric = EpisodeSpeedMetrics(1, 1000, 0.02, OBJECTIVE, "cpu")
    for step in range(1, 1001):
        # One second out of band, including 0.2 s in the final five seconds.
        speed = torch.tensor([1.4 if step <= 40 or step > 990 else 2.0])
        result = metric.update(speed, torch.tensor([step]), torch.tensor([step == 1000]))
    assert result["speed_qualified_success"].item()
    assert result["episode_speed_in_band_fraction"].item() == pytest.approx(0.95)
    assert result["episode_tail_speed_in_band_fraction"].item() == pytest.approx(0.96)
    episodes = [
        {
            "survived_horizon": True,
            "speed_qualified_success": False,
            "duration_s": 20,
            "tilt_failure": False,
            "ground_failure": False,
        },
        {
            "survived_horizon": True,
            "speed_qualified_success": True,
            "duration_s": 20,
            "tilt_failure": False,
            "ground_failure": False,
        },
    ]
    report = summarize_episodes(episodes)
    assert report["survival_rate"] == 1 and report["speed_qualified_20s_rate"] == 0.5
    assert not episode_is_successful(episodes[0]) and episode_is_successful(episodes[1])
    assert episode_is_successful({"survived_horizon": True})  # Historical recordings.


def test_invalid_speed_objectives_fail_before_simulation():
    validate_speed_objective(OBJECTIVE, 20)
    for key, value in [
        ("minimum_mps", 2.0),
        ("reward_sigma_mps", 0),
        ("success_tail_seconds", 25),
        ("success_fraction", 1.1),
        ("target_mps", float("nan")),
    ]:
        with pytest.raises(ValueError):
            validate_speed_objective(dict(OBJECTIVE, **{key: value}), 20)
