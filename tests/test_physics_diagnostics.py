import json

import numpy as np
import torch

from bicycle_sim.cli.train_rider import save_physics_failure
from bicycle_sim.rl.physics_diagnostics import RiderPhysicsError, physics_failure_report


def test_epa_overflow_is_distinguished_from_nonfinite_state():
    q, v = torch.zeros(3, 7), torch.zeros(3, 6)
    assert physics_failure_report(q, v, torch.zeros(3, dtype=torch.int32)) is None
    report = physics_failure_report(q, v, torch.tensor([0, 256, 0]))
    assert report["nonfinite_qpos_envs"] == report["nonfinite_qvel_envs"] == []
    assert report["overflows"] == [
        {"env": 1, "code": 256, "flags": ["EPA_HORIZON"], "unknown_bits": 0}
    ]


def test_multiple_overflows_and_nonfinite_velocity_are_all_reported():
    q, v = torch.zeros(2, 7), torch.zeros(2, 6)
    q[0, 0] = float("nan")
    v[1, 3] = float("inf")
    report = physics_failure_report(q, v, torch.tensor([0, 256 | 1 | 1024]))
    assert report["nonfinite_qpos_envs"] == [0]
    assert report["nonfinite_qvel_envs"] == [1]
    assert report["overflows"][0]["flags"] == ["NEFC", "EPA_HORIZON"]
    assert report["overflows"][0]["unknown_bits"] == 1024


def test_failure_preserves_checkpoint_and_marks_run_incomplete(tmp_path):
    class Agent:
        def save(self, path):
            torch.save({"policy": "last valid weights"}, path)

    states = {
        "qpos_before": torch.zeros(2, 7),
        "qpos_after": torch.ones(2, 7),
        "actions": torch.zeros(2, 35),
    }
    error = RiderPhysicsError(
        {"overflows": [{"env": 1, "code": 256, "flags": ["EPA_HORIZON"]}]}, states
    )
    save_physics_failure(tmp_path, Agent(), error, vector_steps=7, num_envs=2, policy_updated=True)
    assert (
        torch.load(tmp_path / "interrupted.pt", weights_only=True)["policy"] == "last valid weights"
    )
    report = json.loads((tmp_path / "physics_failure.json").read_text())
    assert report["transitions_completed"] == 14 and report["failed_vector_step"] == 8
    status = json.loads((tmp_path / "completion.json").read_text())
    assert not status["finished"] and status["physics_failure"]
    with np.load(tmp_path / "physics_failure_states.npz") as data:
        np.testing.assert_array_equal(data["qpos_before"], 0)
        np.testing.assert_array_equal(data["qpos_after"], 1)


def test_checkpoint_write_failure_does_not_lose_physics_diagnostics(tmp_path):
    class Agent:
        def save(self, path):
            raise OSError("simulated write failure")

    error = RiderPhysicsError({"overflows": []}, {"qpos_after": torch.zeros(1, 7)})
    save_physics_failure(tmp_path, Agent(), error, vector_steps=0, num_envs=1)
    report = json.loads((tmp_path / "physics_failure.json").read_text())
    assert "simulated write failure" in report["checkpoint_save_error"]
    assert (tmp_path / "physics_failure_states.npz").is_file()
