import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from bicycle_sim.cli.probe_hand_grasp import collect_finger_contacts
from bicycle_sim.control.hand_synergy import FingerSynergy, finger_targets_tensor
from bicycle_sim.rl.rider_gpu_env import quaternion_product
from bicycle_sim.rl.rider_model import build_rider_model


def contact_model(tmp_path):
    cfg = json.loads(Path("configs/rl/bicycle_b_contact_sac.json").read_text())
    source = mujoco.MjModel.from_xml_path("assets/scenes/rider_fit_diagnostic.xml")
    d = mujoco.MjData(source)
    mujoco.mj_resetDataKeyframe(source, d, 0)
    reference = tmp_path / "phase.npz"
    np.savez(reference, qpos=np.tile(d.qpos, (25, 1)), phases=np.linspace(0, 2 * np.pi, 25))
    return cfg, reference, build_rider_model(cfg, reference)


def test_natural_rider_has_no_palm_connections_and_starts_in_loaded_grasp(tmp_path):
    cfg, _, (m, q0, ctrl0, maps) = contact_model(tmp_path)
    assert m.neq == 1 and m.equality(0).name == "diagnostic_seat_connection"
    assert (m.nq, m.nv, m.nu) == (63, 61, 45)
    d = mujoco.MjData(m)
    d.qpos[:], d.ctrl[:] = q0, ctrl0
    mujoco.mj_forward(m, d)
    contacts, deepest, _ = collect_finger_contacts(m, d)
    assert all(v["normal_force_N"] > 0.5 for v in contacts.values())
    assert deepest > -0.001
    target = finger_targets_tensor(
        torch.full((1, 6), cfg["hand_nominal_closure"]),
        torch.tensor(maps["hand_open"]),
        torch.tensor(maps["hand_closed"]),
        torch.tensor(maps["hand_columns"]),
        torch.tensor(m.actuator_ctrlrange[maps["hand_act"]]),
    )
    np.testing.assert_allclose(target[0], ctrl0[maps["hand_act"]], atol=1e-6)


def test_batched_hand_targets_match_the_validated_single_hand_controller(tmp_path):
    _, _, (m, q0, _, _) = contact_model(tmp_path)
    # Use a separate open reference: the settled q0 is not the open posture.
    opened = q0.copy()
    for j in range(m.njnt):
        if "_hand_" in m.joint(j).name:
            opened[m.jnt_qposadr[j]] = 0
    controller = FingerSynergy(m, opened)
    closure = np.array([[0.8, 0.9, 1, 0.85, 0.95, 1], [0.9] * 6], dtype=np.float32)
    result = finger_targets_tensor(
        torch.tensor(closure),
        torch.tensor(controller.nominal, dtype=torch.float32),
        torch.tensor(controller.closed, dtype=torch.float32),
        torch.tensor(controller.columns),
        torch.tensor(controller.limits, dtype=torch.float32),
    )
    for i in range(2):
        np.testing.assert_allclose(result[i], controller.targets(closure[i]), atol=1e-7)
    assert result.dtype == torch.float32


def test_reset_rotation_preserves_settled_pelvis_and_hand_geometry(tmp_path):
    _, _, (m, q0, _, _) = contact_model(tmp_path)
    d0 = mujoco.MjData(m)
    d0.qpos[:] = q0
    mujoco.mj_kinematics(m, d0)
    quat = np.array([0.9, 0.1, -0.2, 0.3])
    quat /= np.linalg.norm(quat)
    pq = m.joint("floating_base_joint").qposadr[0]
    product = quaternion_product(torch.tensor(quat), torch.tensor(q0[pq + 3 : pq + 7])).numpy()
    expected = np.zeros(4)
    mujoco.mju_mulQuat(expected, quat, q0[pq + 3 : pq + 7])
    np.testing.assert_allclose(product, expected, atol=1e-12)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.qpos[:3] = R @ q0[:3]
    d.qpos[3:7] = quat
    d.qpos[pq : pq + 3] = R @ q0[pq : pq + 3]
    d.qpos[pq + 3 : pq + 7] = product
    mujoco.mj_kinematics(m, d)
    for side in ("left", "right"):
        site = m.site(f"{side}_grasp_reference").id
        np.testing.assert_allclose(d.site_xpos[site], R @ d0.site_xpos[site], atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires visible CUDA")
def test_natural_grasp_gpu_step_and_partial_reset(tmp_path):
    from bicycle_sim.rl.cuda_context import shared_cuda_stream
    from bicycle_sim.rl.rider_gpu_env import RiderGpuEnv

    cfg, reference, _ = contact_model(tmp_path)
    cfg["num_envs"] = 4
    with shared_cuda_stream("cuda:0"):
        env = RiderGpuEnv(cfg, reference_path=reference)
        assert env.num_actions == 20 and env.num_observations == 148
        assert torch.all(env.finger_forces.sum(-1) > 0.5)
        old_q, old_v = env.q[1:].clone(), env.v[1:].clone()
        env.reset(torch.tensor([0], device="cuda:0"))
        torch.testing.assert_close(env.q[1:], old_q, atol=0, rtol=0)
        torch.testing.assert_close(env.v[1:], old_v, atol=0, rtol=0)
        obs, reward, _, _, info = env.step(torch.zeros((4, 20), device="cuda:0"))
        assert obs.shape == (4, 148) and torch.isfinite(obs).all()
        assert reward.shape == (4, 1) and info["hand_closure"].shape == (4, 6)
        torch.testing.assert_close(
            env.sim.data.ctrl[:, env.maps["arm_act"]], env.arm_nominal.expand(4, -1)
        )
