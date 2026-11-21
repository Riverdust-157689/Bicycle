"""Comparison must use complete, matching initial cohorts rather than reset counts."""

import json

import numpy as np
import pytest

from bicycle_sim.cli.compare_rider_controls import compare_results
from bicycle_sim.cli.probe_rider_gpu import summarize_episodes


def test_comparison_checks_initial_states_and_counts_each_world_once(tmp_path):
    for mode, durations in (("zero", (1.0, 2.0)), ("feedback", (6.0, 20.0))):
        path = tmp_path / mode
        path.mkdir()
        (path / "run.json").write_text(json.dumps({"environment": {"num_envs": 2}}))
        np.savez(path / "initial_states.npz", qpos=np.zeros((2, 3)), qvel=np.ones((2, 3)))
        episodes = [
            {
                "env": i,
                "duration_s": t,
                "survived_horizon": t == 20,
                "tilt_failure": t < 20,
                "ground_failure": False,
            }
            for i, t in enumerate(durations)
        ]
        (path / "results.json").write_text(
            json.dumps(
                {
                    **summarize_episodes(episodes),
                    "first_cohort_complete": True,
                    "episodes": episodes,
                }
            )
        )
    report = compare_results(tmp_path)
    assert report["baselines"]["feedback"]["survival_rate"] == 0.5
    assert report["paired_feedback_minus_zero_seconds"]["mean"] == 11.5
    path = tmp_path / "feedback/results.json"
    original = path.read_text()
    bad = json.loads(original)
    bad["episodes"].append(bad["episodes"][0])
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="one completed episode"):
        compare_results(tmp_path)
    path.write_text(original)
    np.savez(tmp_path / "feedback/initial_states.npz", qpos=np.ones((2, 3)), qvel=np.ones((2, 3)))
    with pytest.raises(AssertionError):
        compare_results(tmp_path)
