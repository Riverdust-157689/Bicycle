import json

import mujoco
import numpy as np
import pytest

from bicycle_sim.cli.analyze_rider_feet import audit_feet
from bicycle_sim.rl.rider_recording import load_recorded_episode


def recording(tmp_path):
    (tmp_path / "run.json").write_text(
        json.dumps({"backend": "mjlab/MuJoCo-Warp", "record_env": 0})
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"env": 0, "survived_horizon": False},
                    {"env": 1, "survived_horizon": True},
                    {"env": 2, "survived_horizon": True},
                ]
            }
        )
    )
    q = np.arange(4 * 3 * 7).reshape(4, 3, 7).astype(float)
    valid = np.ones((4, 3), dtype=bool)
    valid[3, 0] = False
    t = np.tile(np.arange(4)[:, None], (1, 3)) * 0.02
    np.savez(
        tmp_path / "cohort_trajectory.npz",
        qpos=q,
        qvel=q[:, :, :6],
        time_s=t,
        valid_first_episode=valid,
    )
    np.savez(tmp_path / "trajectory.npz", qpos=q[:3, 0], time_s=t[:3, 0], episode_id=np.zeros(3))
    return q


def test_selects_success_from_this_recording_and_keeps_source_metadata(tmp_path):
    q = recording(tmp_path)
    meta, track = load_recorded_episode(tmp_path, successful=True)
    assert meta["record_env"] == 1 and len(track["time_s"]) == 4
    np.testing.assert_array_equal(track["qpos"], q[:, 1])
    assert json.loads((tmp_path / "run.json").read_text())["record_env"] == 0


def test_cohort_selection_excludes_reset_frames_and_preserves_old_format(tmp_path):
    q = recording(tmp_path)
    _meta, track = load_recorded_episode(tmp_path, env_id=0)
    np.testing.assert_array_equal(track["qpos"], q[:3, 0])
    assert np.all(track["episode_id"] == 0)
    _, old = load_recorded_episode(tmp_path)
    np.testing.assert_array_equal(old["qpos"], q[:3, 0])
    with pytest.raises(ValueError, match="out of range"):
        load_recorded_episode(tmp_path, env_id=3)


def test_missing_cohort_or_no_success_is_not_replaced_with_a_failed_video(tmp_path):
    recording(tmp_path)
    (tmp_path / "results.json").write_text(
        json.dumps({"episodes": [{"env": 0, "survived_horizon": False}]})
    )
    with pytest.raises(ValueError, match="No completed successful"):
        load_recorded_episode(tmp_path, successful=True)
    (tmp_path / "cohort_trajectory.npz").unlink()
    with pytest.raises(ValueError, match="record-all-envs"):
        load_recorded_episode(tmp_path, env_id=1)


def test_foot_audit_distinguishes_proximity_loading_and_backward_crank():
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="left_pedal_platform" type="box" size=".07 .045 .012" pos="0 .1 .5"/>
      <geom name="right_pedal_platform" type="box" size=".07 .045 .012" pos="0 -.1 .5"/>
      <body><joint name="crank_spin" axis="0 1 0"/><geom type="sphere" size=".01" mass="1"/></body>
      <body pos="0 .1 .524"><freejoint name="left_root"/><geom name="left_riding_sole" type="box" size=".1 .04 .01"/></body>
      <body pos="0 -.1 .524"><freejoint name="right_root"/><geom name="right_riding_sole" type="box" size=".1 .04 .01"/></body>
    </worldbody></mujoco>""")
    q = np.tile(m.qpos0, (4, 1))
    q[:, m.joint("crank_spin").qposadr[0]] = np.linspace(0, -np.pi, 4)
    q[2:, m.joint("left_root").qposadr[0] + 2] = 0.7
    track = {
        "qpos": q,
        "time_s": np.arange(4, dtype=float),
        "support_normal_force_N": np.array([[0, 2, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0]]),
        "crank_speed_rad_s": np.full(4, -1.0),
        "foot_to_crank_torque_Nm": np.ones(4),
    }
    report, gap = audit_feet(m, track)
    assert gap[0, 0] == pytest.approx(0.002, abs=1e-6)
    assert report["left"]["near_pedal_fraction"] == pytest.approx(1 / 3)
    assert report["left"]["loaded_contact_fraction"] == pytest.approx(1 / 3)
    assert report["left"]["longest_gap_over_2cm_s"] == 2
    assert report["right"]["near_pedal_fraction"] == 1
    assert report["right"]["loaded_contact_fraction"] == 0
    assert report["crank_net_revolutions"] == pytest.approx(-0.5)
    assert report["crank_reverse_fraction"] == 1
    assert report["sampled_mean_foot_to_crank_power_W"] == -1


def test_interrupted_probe_preserves_completed_success_and_valid_partial_frames(tmp_path):
    import torch

    from bicycle_sim.cli.probe_rider_gpu import save_probe_recording
    from bicycle_sim.cli.train_rider import save_physics_failure
    from bicycle_sim.rl.physics_diagnostics import RiderPhysicsError

    recording(tmp_path)
    with np.load(tmp_path / "cohort_trajectory.npz") as data:
        cohort = [{k: data[k][i] for k in data.files} for i in range(4)]
    with np.load(tmp_path / "trajectory.npz") as data:
        frames = [{k: data[k][i] for k in data.files} for i in range(3)]
    error = RiderPhysicsError(
        {"overflows": [{"env": 2, "code": 256, "flags": ["EPA_HORIZON"]}]},
        {"qpos_after": torch.full((3, 7), 999.0)},
    )
    save_physics_failure(tmp_path, None, error, vector_steps=3, num_envs=3, evaluation=True)
    report = {
        "first_cohort_complete": False,
        "physics_failure": error.report,
        "episodes": [{"env": 1, "survived_horizon": True}],
    }
    save_probe_recording(tmp_path, report, frames, cohort)
    meta, track = load_recorded_episode(tmp_path, successful=True)
    assert meta["record_env"] == 1 and len(track["qpos"]) == 4
    assert (track["qpos"] < 999).all()  # Faulty post-step poses only enter diagnostic files.
    assert not json.loads((tmp_path / "completion.json").read_text())["finished"]
    assert json.loads((tmp_path / "results.json").read_text())["physics_failure"]
    assert not (tmp_path / "interrupted.pt").exists()  # Evaluation never rewrites the policy.
