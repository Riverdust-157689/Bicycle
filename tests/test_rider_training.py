"""CPU-verifiable parts of B; CUDA and optional agents have separate smoke checks."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from bicycle_sim.rl.rider_gpu_env import phase_reference, task_reward
from bicycle_sim.rl.rider_model import build_rider_model

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/rl/bicycle_b.json").read_text())


def model_with_structural_reference(tmp_path):
    # Independent structural fixture, not a valid cycling trajectory.
    source = mujoco.MjModel.from_xml_path(str(ROOT / "assets/scenes/rider_fit_diagnostic.xml"))
    d = mujoco.MjData(source)
    mujoco.mj_resetDataKeyframe(source, d, 0)
    path = tmp_path / "reference.npz"
    np.savez(path, qpos=np.tile(d.qpos, (25, 1)), phases=np.linspace(0, 2 * np.pi, 25))
    return build_rider_model(CFG, path)


def test_b_has_free_vehicle_and_no_steering_actuator(tmp_path):
    m, q0, ctrl, maps = model_with_structural_reference(tmp_path)
    assert (m.nq, m.nv, m.nu) == (63, 61, 45)
    assert m.joint("bike_root").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert m.body_mass.sum() == pytest.approx(43.3942352)
    assert len(maps["arm_act"]) == 14 and maps["leg_table"].shape == (25, 12)
    assert not any(
        m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
        and m.actuator_trnid[a, 0] == m.joint("steer").id
        for a in range(m.nu)
    )
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.ctrl[:] = ctrl
    for _ in range(20):
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all() and not d.warning.number.any()
    # With both free roots rotated, every local connection remains satisfied.
    assert m.neq == 3  # seat weld + two ideal palm point constraints


def test_candidate_tires_preserve_rider_state_inertias_and_leg_reference(tmp_path):
    baseline, q0, ctrl, maps = model_with_structural_reference(tmp_path)
    changed, q1, ctrl1, maps1 = build_rider_model(
        dict(CFG, tire_collision_model="ellipsoid"), tmp_path / "reference.npz"
    )
    for name in ("body_mass", "body_inertia", "geom_friction", "geom_solref", "geom_solimp"):
        np.testing.assert_allclose(getattr(changed, name), getattr(baseline, name), atol=1e-10)
    np.testing.assert_array_equal(q0, q1)
    np.testing.assert_array_equal(ctrl, ctrl1)
    np.testing.assert_array_equal(maps["leg_table"], maps1["leg_table"])
    assert (changed.nq, changed.nv, changed.nu) == (baseline.nq, baseline.nv, baseline.nu)
    for side in ("rear", "front"):
        assert baseline.geom(f"{side}_wheel_tire").type[0] == mujoco.mjtGeom.mjGEOM_MESH
        assert changed.geom(f"{side}_wheel_tire").type[0] == mujoco.mjtGeom.mjGEOM_ELLIPSOID
    d = mujoco.MjData(changed)
    d.qpos[:], d.ctrl[:] = q1, ctrl1
    for _ in range(20):
        mujoco.mj_step(changed, d)
    assert np.isfinite(d.qpos).all() and not d.warning.number.any()


def test_phase_interpolation_wraps_without_a_reset_jump():
    angle = torch.linspace(0, 2 * torch.pi, 25)
    table = torch.stack((angle.sin(), angle.cos()), -1)
    phase = torch.tensor([-1e-5, 0.0, 2 * torch.pi, 2 * torch.pi + 1e-5])
    reference = phase_reference(phase, table)
    torch.testing.assert_close(reference[1], reference[2], atol=1e-6, rtol=0)
    torch.testing.assert_close(reference[0], reference[3], atol=3e-5, rtol=0)


def test_moving_balance_reward_does_not_reward_standing_or_a_world_heading():
    R = torch.eye(3).repeat(2, 1, 1)
    a = torch.zeros(2, 14)
    omega = torch.zeros(2, 3)
    velocity = torch.tensor([[2.0, 0, 0], [0.0, 0, 0]])
    reward, tilt = task_reward(R, velocity, omega, a, a)
    assert reward[0] > reward[1] and torch.all(tilt == 0)


@pytest.mark.parametrize(
    "num_actions,num_observations", [(14, 130), (20, 148), (35, 201), (35, 207)]
)
def test_optional_ppo_and_sac_can_update_on_cpu(tmp_path, num_actions, num_observations):
    pytest.importorskip("skrl", reason="Run after installing the optional pinned training extra")
    from types import SimpleNamespace

    from bicycle_sim.rl.skrl_agents import make_agent

    torch.set_num_threads(1)
    env = SimpleNamespace(
        num_observations=num_observations,
        num_actions=num_actions,
        num_envs=4,
        device="cpu",
    )
    cfg = dict(CFG, ppo_rollouts=8)
    for algorithm in ("ppo", "sac"):
        agent = make_agent(algorithm, env, cfg, tmp_path / algorithm, smoke=True)
        agent.init(trainer_cfg={"timesteps": 32})
        agent.set_running_mode("train")
        initial = [p.detach().clone() for p in agent.models["policy"].parameters()]
        states = torch.randn(4, env.num_observations) * 0.05
        for i in range(32):
            agent.pre_interaction(timestep=i, timesteps=32)
            with torch.no_grad():
                actions, _, _ = agent.act(states, timestep=i, timesteps=32)
                assert actions.shape == (4, num_actions)
                next_states = torch.randn_like(states) * 0.05
                agent.record_transition(
                    states=states,
                    actions=actions,
                    rewards=torch.ones(4, 1) * 0.02,
                    next_states=next_states,
                    terminated=torch.zeros(4, 1, dtype=torch.bool),
                    truncated=torch.zeros(4, 1, dtype=torch.bool),
                    infos={},
                    timestep=i,
                    timesteps=32,
                )
            agent.post_interaction(timestep=i, timesteps=32)
            states = next_states
        assert any(
            not torch.equal(a, b)
            for a, b in zip(initial, agent.models["policy"].parameters(), strict=True)
        )


def test_waist_and_fingers_have_restoring_pd_but_no_policy_actions(tmp_path):
    m, q0, ctrl, maps = model_with_structural_reference(tmp_path)
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.ctrl[:] = ctrl
    waist = m.joint("waist_pitch_joint")
    finger = m.joint("left_hand_index_0_joint")
    for joint in (waist, finger):
        act = m.actuator(joint.name).id
        assert act not in maps["arm_act"]
        assert joint.qposadr[0] in maps["robot_q"]
        assert joint.dofadr[0] in maps["robot_v"]
        assert ctrl[act] == pytest.approx(q0[joint.qposadr[0]])
        d.qpos[joint.qposadr[0]] += 0.02
    mujoco.mj_forward(m, d)
    assert d.qfrc_actuator[waist.dofadr[0]] < -1
    assert d.qfrc_actuator[finger.dofadr[0]] < 0


def test_pedals_are_unpowered_free_bearings_and_sole_covers_the_platform(tmp_path):
    m, q0, ctrl, _ = model_with_structural_reference(tmp_path)
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d)
    for side in ("left", "right"):
        joint = m.joint(f"{side}_pedal_spin")
        assert not joint.limited[0] and m.jnt_stiffness[joint.id] == 0
        assert not any(
            m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
            and m.actuator_trnid[a, 0] == joint.id
            for a in range(m.nu)
        )
        sole = m.geom(f"{side}_riding_sole").id
        pedal = m.geom(f"{side}_pedal_platform").id
        assert m.geom_contype[sole] != 0
        distance = mujoco.mj_geomDistance(m, d, sole, pedal, 0.1, None)
        assert abs(distance) < 0.003


def test_leg_reference_velocity_cancels_native_position_servo_drag():
    from bicycle_sim.rl.rider_gpu_env import leg_servo_reference

    angles = torch.linspace(0, 2 * torch.pi, 25)
    table = torch.stack((angles.sin(), angles.cos()), -1)
    phase = torch.tensor([0.31, 1.83])
    omega = torch.tensor([3.0, 4.0])
    kp = torch.tensor([500.0, 500.0])
    kv = torch.tensor([10.0, 15.0])
    target = phase_reference(phase, table)
    command = leg_servo_reference(phase, omega, table, kv / kp)
    eps = 1e-3
    target_vel = (
        (phase_reference(phase + eps, table) - phase_reference(phase - eps, table))
        / (2 * eps)
        * omega[:, None]
    )
    torque = kp * (command - target) - kv * target_vel
    torch.testing.assert_close(torque, torch.zeros_like(torque), atol=0.02, rtol=0)


def test_raw_ppo_keeps_sample_and_log_probability_consistent_at_action_bounds():
    pytest.importorskip("skrl")
    from bicycle_sim.rl.skrl_agents import Policy

    policy = Policy(130, 14, "cpu", (16,), raw_gaussian=True)
    with torch.no_grad():
        for p in policy.net.parameters():
            p.zero_()
        policy.net[-1].bias.fill_(2.0)
        policy.log_std.fill_(-3.0)
    states = torch.zeros(8, 130)
    actions, logp, _ = policy.act({"states": states}, role="policy")
    assert torch.all(actions > 1)  # PPO stores the raw draw; only env execution clips.
    expected = (
        torch.distributions.Normal(
            torch.full_like(actions, 2), torch.full_like(actions, float(np.exp(-3)))
        )
        .log_prob(actions)
        .sum(-1, keepdim=True)
    )
    torch.testing.assert_close(logp, expected)
    _, logp_again, _ = policy.act({"states": states, "taken_actions": actions}, role="policy")
    torch.testing.assert_close(logp, logp_again)


def test_probe_visual_model_preserves_free_roots_and_inertia():
    from bicycle_sim.cli.render_rider_probe import visual_model

    scene = ROOT / "assets/scenes/rider_fit_diagnostic.xml"
    original = mujoco.MjModel.from_xml_path(str(scene))
    rendered = visual_model(scene)
    assert (rendered.nq, rendered.nv, rendered.nu) == (original.nq, original.nv, original.nu)
    assert rendered.joint("bike_root").type[0] == mujoco.mjtJoint.mjJNT_FREE
    np.testing.assert_array_equal(rendered.body_mass, original.body_mass)
    np.testing.assert_array_equal(rendered.body_inertia, original.body_inertia)


def test_deterministic_probe_actions_match_evaluator_means_without_sampling():
    pytest.importorskip("skrl")
    from bicycle_sim.rl.skrl_agents import Policy, SquashedPolicy, deterministic_actions

    states = torch.zeros(3, 130)
    for cls, algorithm, raw in (
        (Policy, "ppo", False),
        (Policy, "ppo", True),
        (SquashedPolicy, "sac", True),
    ):
        policy = cls(130, 14, "cpu", (16,), raw_gaussian=raw)
        with torch.no_grad():
            for parameter in policy.net.parameters():
                parameter.zero_()
            policy.net[-1].bias.fill_(2)
            rng = torch.random.get_rng_state().clone()
            actual = deterministic_actions(policy, states, algorithm)
            assert torch.equal(rng, torch.random.get_rng_state())
            _, _, outputs = policy.act({"states": states}, role="policy")
            torch.testing.assert_close(actual, outputs["mean_actions"])
            if algorithm == "sac":
                assert torch.all(actual < 1)  # Never replay unsquashed SAC means.
            elif raw:
                assert torch.all(actual > 1)
