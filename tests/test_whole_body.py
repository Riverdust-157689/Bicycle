"""Full-body action ownership, real motor-off physics, and curriculum semantics."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from bicycle_sim.cli.settle_whole_body import fixed_bench, sitting_contacts
from bicycle_sim.control.seated_leg_reference import leg_pose_ik, refit_leg_cycle
from bicycle_sim.rl.rider_gpu_env import RiderGpuEnv, quaternion_product
from bicycle_sim.rl.rider_model import build_rider_model, model_fingerprint
from bicycle_sim.rl.whole_body import (
    ARM,
    HAND,
    LEG,
    WAIST,
    advance_offset,
    assistance_at,
    assistance_torque,
    contact_crank_torque,
    direct_joint_targets,
    normalized_joint_targets,
    pedal_joint_features,
    relative_pelvis_features,
)


@pytest.fixture
def whole_source(tmp_path):
    cfg = json.loads(Path("configs/rl/bicycle_b_whole_body_sac.json").read_text())
    source = mujoco.MjModel.from_xml_path("assets/scenes/rider_fit_diagnostic.xml")
    d = mujoco.MjData(source)
    mujoco.mj_resetDataKeyframe(source, d, 0)
    phase = tmp_path / "phase.npz"
    np.savez(phase, qpos=np.tile(d.qpos, (25, 1)), phases=np.linspace(0, 2 * np.pi, 25))
    result = build_rider_model(cfg, phase, load_settled=False)
    return cfg, phase, result


def structural_reference(cfg, phase, result, path):
    # This temporary fixture exercises schema/topology, not pose acceptance.
    m, q, ctrl, maps = result
    ref = {
        "backend": "mjlab/MuJoCo-Warp",
        "passed": True,
        "model_fingerprint": model_fingerprint(m),
        "phase_reference_sha256": hashlib.sha256(phase.read_bytes()).hexdigest(),
        "joint_qpos": {},
        "control_targets": {},
        "leg_reference": maps["leg_table"].tolist(),
    }
    for j in range(m.njnt):
        a = m.jnt_qposadr[j]
        n = 7 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        ref["joint_qpos"][m.joint(j).name] = q[a : a + n].tolist()
    ref["control_targets"] = {m.actuator(a).name: float(v) for a, v in enumerate(ctrl)}
    path.write_text(json.dumps(ref))
    cfg["whole_body_reference_path"] = str(path)
    return ref


def test_all_body_joints_are_owned_and_seat_is_contact_only(whole_source):
    _, _, (m, q, ctrl, maps) = whole_source
    assert m.neq == 0 and m.nq == 63 and m.nv == 61
    main = np.r_[maps["arm_act"], maps["leg_act"], maps["waist_act"]]
    assert len(main) == len(set(main)) == 29
    assert len(maps["hand_act"]) == 14
    assert len(set(np.r_[main, maps["hand_act"]])) == 43
    seat = m.geom("saddle").id
    post = m.geom("seat_post").id
    assert m.geom_pos[seat, 2] + m.geom_size[seat, 2] == pytest.approx(0.7871279513)
    assert m.geom_pos[post, 2] + m.geom_size[post, 1] + m.geom_size[post, 0] == pytest.approx(
        m.geom_pos[seat, 2] - m.geom_size[seat, 2]
    )
    # Physical actuator force is zero for both crank directions, even at speed.
    a = m.actuator("crank_drive").id
    j = m.joint("crank_spin")
    for speed in [-8.0, 8.0]:
        d = mujoco.MjData(m)
        d.qpos[:] = q
        d.ctrl[:] = ctrl
        d.qvel[j.dofadr[0]] = speed
        d.ctrl[a] = 0
        mujoco.mj_forward(m, d)
        assert d.actuator_force[a] == 0
        np.testing.assert_array_equal(m.actuator_biasprm[a], 0)


def test_reference_rejects_unverified_or_incomplete_state(whole_source, tmp_path):
    cfg, phase, result = whole_source
    p = tmp_path / "settled.json"
    ref = structural_reference(cfg, phase, result, p)
    loaded = build_rider_model(cfg, phase)
    np.testing.assert_array_equal(loaded[1], result[1])
    ref["backend"] = "native CPU MuJoCo"
    p.write_text(json.dumps(ref))
    with pytest.raises(ValueError, match="accepted GPU"):
        build_rider_model(cfg, phase)
    ref["backend"] = "mjlab/MuJoCo-Warp"
    del ref["joint_qpos"]["crank_spin"]
    p.write_text(json.dumps(ref))
    with pytest.raises(ValueError, match="every joint"):
        build_rider_model(cfg, phase)


def test_reference_rejects_changed_geometry_and_phase_table(whole_source, tmp_path):
    cfg, phase, result = whole_source
    p = tmp_path / "settled.json"
    structural_reference(cfg, phase, result, p)
    with pytest.raises(ValueError, match="physical model"):
        build_rider_model(dict(cfg, saddle_raise_m=0.02), phase)
    ref = json.loads(p.read_text())
    ref["phase_reference_sha256"] = "stale"
    p.write_text(json.dumps(ref))
    with pytest.raises(ValueError, match="phase table changed"):
        build_rider_model(cfg, phase)


def test_initialization_has_no_seat_or_hand_attachment(whole_source, tmp_path):
    _, _, (m, q, c, _) = whole_source
    p = tmp_path / "source.xml"
    mujoco.mj_saveLastXML(str(p), m)
    b, state, ctrl = fixed_bench(m, q, c, p.read_text())
    assert b.neq == 0 and b.nq == m.nq - 7
    d = mujoco.MjData(b)
    d.qpos[:] = state
    d.ctrl[:] = ctrl
    mujoco.mj_forward(b, d)
    load, _, _ = sitting_contacts(b, d)
    assert load[0] == 0  # The intended initial 0.5 mm gap is not a weld.
    d.qpos[b.joint("floating_base_joint").qposadr[0] + 2] -= 0.002
    mujoco.mj_forward(b, d)
    assert sitting_contacts(b, d)[0][0] > 0


def test_tilted_pelvis_foot_initialization_and_refitted_cycle(whole_source):
    _, _, (m, q, ctrl, maps) = whole_source
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d)
    _, _, old_depth, details = sitting_contacts(m, d, with_details=True)
    assert old_depth < -0.002
    assert details["right_foot"]["deepest_contact"]["bodies"] == [
        "right_pedal",
        "right_ankle_roll_link",
    ]
    # Removing the weld permits pelvis pitch; the old cycle assumes zero pitch.
    pq = m.joint("floating_base_joint").qposadr[0]
    q[pq + 3 : pq + 7] = [np.cos(0.01), 0, np.sin(0.01), 0]
    cq = m.joint("crank_spin").qposadr[0]
    q[cq] = 0.25
    for side in ["left", "right"]:
        q[m.joint(side + "_pedal_spin").qposadr[0]] = -0.25 + 0.08
    pose, check = leg_pose_ik(m, q)
    assert check["max_position_error_m"] < 1e-5 and check["max_angle_error_rad"] < 1e-4
    q[maps["leg_q"]] = pose
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    _, _, _, after = sitting_contacts(m, d, with_details=True)
    assert after["left_foot"]["penetration_m"] < 1e-5
    assert after["right_foot"]["penetration_m"] < 1e-5
    table, check = refit_leg_cycle(m, q)
    assert table.shape == (25, 12) and check["max_position_error_m"] < 1e-5
    np.testing.assert_allclose(table[0], table[-1], atol=1e-8)


def test_loaded_task_uses_fitted_cycle_and_rejects_malformed_cycle(whole_source, tmp_path):
    cfg, phase, result = whole_source
    p = tmp_path / "settled.json"
    ref = structural_reference(cfg, phase, result, p)
    ref["leg_reference"][1][0] += 0.01
    p.write_text(json.dumps(ref))
    _, _, _, maps = build_rider_model(cfg, phase)
    assert maps["leg_table"][1, 0] == pytest.approx(ref["leg_reference"][1][0])
    ref["leg_reference"] = ref["leg_reference"][:-1]
    p.write_text(json.dumps(ref))
    with pytest.raises(ValueError, match="Invalid fitted"):
        build_rider_model(cfg, phase)


def test_action_layout_and_residual_rates_do_not_depend_on_curriculum():
    a = torch.arange(35, dtype=torch.float32)[None] / 35
    assert [a[:, s].shape[1] for s in [ARM, HAND, LEG, WAIST]] == [14, 6, 12, 3]
    scales = torch.ones(12) * 0.4
    offset = advance_offset(torch.zeros(1, 12), a[:, LEG], scales, 4.0, 0.02)
    assert torch.all(offset > 0) and offset.max() <= 0.08
    cfg = {
        "crank_assist_full_until_transitions": 100,
        "crank_assist_zero_after_transitions": 200,
        "crank_servo_kv": 2.0,
        "crank_speed_rad_s": 3.7,
        "max_crank_torque_Nm": 8.0,
    }
    assert [assistance_at(i, cfg) for i in [0, 100, 150, 200, 1000]] == [1, 1, 0.5, 0, 0]
    speed = torch.tensor([-100.0, 0, 3.7, 100.0])
    assert assistance_torque(speed, 0, cfg).abs().sum() == 0
    torch.testing.assert_close(
        assistance_torque(speed, 0.5, cfg), assistance_torque(speed, 1, cfg) * 0.5
    )
    assert assistance_at(0, dict(cfg, crank_assistance_override=0)) == 0
    with pytest.raises(ValueError):
        assistance_at(0, dict(cfg, crank_assistance_override=1.1))


def test_direct_targets_span_entire_control_range_without_phase_override(whole_source, monkeypatch):
    from types import SimpleNamespace

    import bicycle_sim.rl.rider_gpu_env as module

    _cfg, _, (m, _q, ctrl, maps) = whole_source
    limits = torch.tensor(m.actuator_ctrlrange[maps["leg_act"]], dtype=torch.float32)
    action = torch.zeros(3, 35)
    action[0, LEG] = -1
    action[2, LEG] = 1
    targets = direct_joint_targets(action[:, LEG], limits)
    torch.testing.assert_close(targets[0], limits[:, 0])
    torch.testing.assert_close(targets[2], limits[:, 1])
    torch.testing.assert_close(targets[1], limits.mean(-1))
    torch.testing.assert_close(normalized_joint_targets(targets, limits), action[:, LEG])
    env = RiderGpuEnv.__new__(RiderGpuEnv)
    env.direct_legs = True
    env.leg_limits = limits
    env.leg_target = torch.zeros(3, 12)
    env.maps = {"leg_act": torch.tensor(maps["leg_act"])}
    env.sim = SimpleNamespace(
        data=SimpleNamespace(ctrl=torch.tensor(ctrl, dtype=torch.float32)[None].repeat(3, 1))
    )

    def forbid(*args, **kwargs):
        raise AssertionError("Direct legs must never invoke a phase controller")

    monkeypatch.setattr(module, "phase_reference", forbid)
    monkeypatch.setattr(module, "leg_servo_reference", forbid)
    # No scale/rate/phase fields are provided: only the direct command can run.
    env.set_leg_actions(action)
    env.apply_leg_control()
    env.apply_leg_control()
    torch.testing.assert_close(env.sim.data.ctrl[:, env.maps["leg_act"]], targets)


def test_pedal_observation_is_periodic_and_contains_both_velocities():
    angle = torch.tensor([[0.2, -3.1], [2.0, -1.0]], dtype=torch.float64)
    speed = torch.tensor([[3.0, -4.0], [0.0, 15.0]], dtype=torch.float64)
    features = pedal_joint_features(angle, speed)
    assert features.shape == (2, 6)
    torch.testing.assert_close(features, pedal_joint_features(angle + 2 * torch.pi, speed))
    torch.testing.assert_close(features[:, -2:], speed / 15)
    assert not torch.allclose(features, pedal_joint_features(angle + 0.1, speed))


def test_direct_observation_explicitly_exposes_pedal_hinges(whole_source):
    from types import SimpleNamespace

    _, _, (m, q, ctrl, maps) = whole_source
    cfg = json.loads(Path("configs/rl/bicycle_b_whole_body_direct_sac.json").read_text())
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d)
    env = RiderGpuEnv.__new__(RiderGpuEnv)
    env.cfg = cfg
    env.whole_body = True
    env.natural_grasp = True
    env.direct_legs = True
    env.observe_pedal_joints = True
    env.num_envs = 1
    env.m = m
    env.maps = {
        k: torch.tensor(v, dtype=torch.long)
        for k, v in maps.items()
        if k in ["robot_q", "robot_v", "pedal_q", "pedal_v"]
    }
    env.q0 = torch.tensor(q, dtype=torch.float32)
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            **{
                k: torch.tensor(getattr(d, k)[None].copy(), dtype=torch.float32)
                for k in ["qpos", "qvel", "xmat", "geom_xpos", "sensordata"]
            }
        )
    )
    env.frame = m.body("bike_frame").id
    env.pelvis_body = m.body("pelvis").id
    env.pelvis_q = m.joint("floating_base_joint").qposadr[0]
    env.pelvis_v = m.joint("floating_base_joint").dofadr[0]
    env.crank_q = m.joint("crank_spin").qposadr[0]
    env.steer_q = m.joint("steer").qposadr[0]
    env.steer_v = m.joint("steer").dofadr[0]
    env.spins = torch.tensor(
        [m.joint(n).dofadr[0] for n in ["crank_spin", "rear_wheel_spin", "front_wheel_spin"]]
    )
    adr = m.sensor("bike_gyro").adr[0]
    env.gyro_slice = slice(adr, adr + 3)
    env.previous = torch.zeros(1, 35)
    env.offset = torch.zeros(1, 14)
    env.closure = torch.full((1, 6), 0.9)
    env.finger_forces = torch.zeros(1, 6)
    env.leg_target = torch.tensor(ctrl[maps["leg_act"]][None], dtype=torch.float32)
    env.leg_limits = torch.tensor(m.actuator_ctrlrange[maps["leg_act"]], dtype=torch.float32)
    env.waist_offset = torch.zeros(1, 3)
    env.waist_scale = torch.tensor(cfg["waist_offset_limit_rad"])
    env.foot_geoms = torch.tensor([m.geom(f"{s}_riding_sole").id for s in ["left", "right"]])
    env.pedal_geoms = torch.tensor([m.geom(f"{s}_pedal_platform").id for s in ["left", "right"]])
    env.support_forces = torch.zeros(1, 3)
    env.assistance = torch.ones(1)
    before = env.observe()
    assert before.shape == (1, 207) and torch.isfinite(before).all()
    # Isolate the explicit hinge features from all other observation tensors.
    env.q[:, env.maps["pedal_q"]] += 0.3
    env.v[:, env.maps["pedal_v"]] += 2
    after = env.observe()
    torch.testing.assert_close(before[:, :201], after[:, :201])
    assert not torch.allclose(before[:, -6:], after[:, -6:])
    torch.testing.assert_close(
        after[:, -6:],
        pedal_joint_features(env.q[:, env.maps["pedal_q"]], env.v[:, env.maps["pedal_v"]]),
    )


def test_direct_experiment_only_changes_leg_interface_and_pedal_observation():
    baseline = json.loads(Path("configs/rl/bicycle_b_whole_body_sac.json").read_text())
    direct = json.loads(Path("configs/rl/bicycle_b_whole_body_direct_sac.json").read_text())
    allowed = {
        "leg_control_mode",
        "observe_pedal_joints",
        "leg_velocity_feedforward",
        "leg_offset_limit_rad",
        "leg_target_rate_rad_s",
    }
    for k in set(baseline) | set(direct):
        if k not in allowed:
            assert baseline.get(k) == direct.get(k)
    assert direct["leg_control_mode"] == "direct_position" and direct["observe_pedal_joints"]
    assert not direct["leg_velocity_feedforward"]
    assert "leg_offset_limit_rad" not in direct and "leg_target_rate_rad_s" not in direct


def test_contact_torque_sign_and_change_of_geom_order():
    # Downward 100 N at the front of a 0.1 m crank produces +10 Nm about +y.
    position = torch.tensor([[0.1, 0, 0], [0.1, 0, 0]])
    force = torch.tensor([[0.0, 0, -100, 0, 0, 0], [0.0, 0, 100, 0, 0, 0]])
    value = contact_crank_torque(
        force,
        torch.eye(3).repeat(2, 1, 1),
        position,
        torch.zeros(2, 3),
        torch.tensor([[0.0, 1, 0]]).repeat(2, 1),
        torch.tensor([True, False]),
    )
    torch.testing.assert_close(value, torch.tensor([10.0, 10.0]))


def test_tensor_contact_aggregation_matches_native_contacts_and_masks_padding(whole_source):
    from types import SimpleNamespace

    _, _, (m, q, ctrl, _) = whole_source
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[:] = ctrl
    d.qpos[m.joint("floating_base_joint").qposadr[0] + 2] -= 0.002
    mujoco.mj_forward(m, d)
    n = d.ncon
    size = n + 2
    geom = np.zeros((size, 2), dtype=np.int64)
    frame = np.zeros((size, 3, 3))
    pos = np.zeros((size, 3))
    forces = np.zeros((size, 6))
    world = np.zeros(size, dtype=np.int64)
    expected = np.zeros(3)
    expected_torque = 0.0
    pelvis = m.body("pelvis").id
    seat = m.geom("saddle").id
    crank = m.body("crank").id
    pedals = np.array([m.geom(f"{s}_pedal_platform").id for s in ["left", "right"]])
    foot_lut = np.array(
        [
            next(
                (
                    i
                    for i, s in enumerate(["left", "right"])
                    if m.body(m.geom_bodyid[g]).name == f"{s}_ankle_roll_link"
                ),
                -1,
            )
            for g in range(m.ngeom)
        ]
    )
    for i, ct in enumerate(d.contact):
        geom[i] = ct.geom
        frame[i] = ct.frame.reshape(3, 3)
        pos[i] = ct.pos
        mujoco.mj_contactForce(m, d, i, forces[i])
        g1, g2 = geom[i]
        if (g1 == seat and m.geom_bodyid[g2] == pelvis) or (
            g2 == seat and m.geom_bodyid[g1] == pelvis
        ):
            expected[0] += max(0, forces[i, 0])
        for side in range(2):
            if (g1 == pedals[side] and foot_lut[g2] == side) or (
                g2 == pedals[side] and foot_lut[g1] == side
            ):
                expected[side + 1] += max(0, forces[i, 0])
                force = frame[i].T @ forces[i, :3] * (1 if g2 == pedals[side] else -1)
                expected_torque += (
                    np.cross(pos[i] - d.xpos[crank], force) @ d.xmat[crank].reshape(3, 3)[:, 1]
                )
    # Invalid capacity slots carry deliberately huge plausible contact values.
    geom[n:] = [seat, next(g for g in range(m.ngeom) if m.geom_bodyid[g] == pelvis)]
    forces[n:, 0] = 1e6
    world[n] = -1
    env = RiderGpuEnv.__new__(RiderGpuEnv)
    env.m = m
    env.num_envs = 1
    env.seat_geom = seat
    env.crank_body = crank
    env.pelvis_geom = torch.tensor(m.geom_bodyid == pelvis)
    env.pedal_geoms = torch.tensor(pedals)
    env.foot_body_lut = torch.tensor(foot_lut)
    env._forces = torch.tensor(forces, dtype=torch.float32)
    env._contact_slots = torch.arange(size)
    env.support_forces = torch.zeros(1, 3)
    env.pedal_torque = torch.zeros(1)
    contact = SimpleNamespace(
        frame=torch.tensor(frame, dtype=torch.float32), pos=torch.tensor(pos, dtype=torch.float32)
    )
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            contact=contact,
            nacon=torch.tensor([n]),
            xpos=torch.tensor(d.xpos[None].copy(), dtype=torch.float32),
            xmat=torch.tensor(d.xmat[None].copy(), dtype=torch.float32),
        )
    )
    env.update_support_contacts(torch.tensor(geom), torch.tensor(world))
    np.testing.assert_allclose(env.support_forces[0], expected, rtol=1e-5, atol=1e-4)
    assert float(env.pedal_torque[0]) == pytest.approx(expected_torque, abs=1e-4)


def test_pelvis_observation_is_invariant_to_common_world_yaw():
    q = torch.tensor([[0.0, 0, 0, 1, 0, 0, 0, 0.24, 0.01, 0.94, 1, 0, 0, 0]])
    v = torch.tensor([[2.0, 0, 0, 0.1, 0.2, 0.3, 2.1, 0.2, 0.1, 0.2, 0.3, 0.4]])
    R = torch.eye(3)[None]
    before = relative_pelvis_features(q, v, R, R, 7, 6)
    yaw = torch.tensor([[np.cos(0.6), 0.0, 0.0, np.sin(0.6)]], dtype=q.dtype)
    quat = yaw[0].double().numpy()
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    rotation = torch.tensor(matrix.reshape(1, 3, 3), dtype=q.dtype)
    rotated = q.clone()
    rotated[:, :3] = q[:, :3] @ rotation[0].T
    rotated[:, 7:10] = q[:, 7:10] @ rotation[0].T
    rotated[:, 3:7] = quaternion_product(yaw, q[:, 3:7])
    rotated[:, 10:14] = quaternion_product(yaw, q[:, 10:14])
    velocity = v.clone()
    velocity[:, :3] = v[:, :3] @ rotation[0].T
    velocity[:, 6:9] = v[:, 6:9] @ rotation[0].T
    after = relative_pelvis_features(rotated, velocity, rotation, rotation, 7, 6)
    for x, y in zip(before, after, strict=True):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("direct_legs", [False, True])
def test_gpu_whole_body_action_targets_partial_reset_and_motor_off(
    whole_source, tmp_path, direct_legs
):
    from bicycle_sim.rl.cuda_context import shared_cuda_stream

    cfg, phase, result = whole_source
    structural_reference(cfg, phase, result, tmp_path / "structural.json")
    cfg.update(num_envs=2, crank_assistance_override=0)
    if direct_legs:
        cfg.update(
            leg_control_mode="direct_position",
            observe_pedal_joints=True,
            leg_velocity_feedforward=False,
        )
        cfg.pop("leg_offset_limit_rad")
        cfg.pop("leg_target_rate_rad_s")
    with shared_cuda_stream("cuda:0"):
        env = RiderGpuEnv(cfg, device="cuda:0", reference_path=phase)
        assert env.num_actions == 35 and env.num_observations == (207 if direct_legs else 201)
        if direct_legs:
            assert env.v[:, env.maps["leg_v"]].abs().sum() == 0
        torch.testing.assert_close(
            env.v[:, env.maps["pedal_v"]], -env.v[:, env.spins[0], None].expand(-1, 2)
        )
        obs, _, _, _, info = env.step(torch.full((2, 35), 0.2, device="cuda:0"))
        assert torch.isfinite(obs).all()
        torch.testing.assert_close(
            env.sim.data.ctrl[:, env.maps["waist_act"]], env.waist_nominal[None] + env.waist_offset
        )
        assert env.waist_offset.abs().sum() > 0
        if direct_legs:
            torch.testing.assert_close(
                env.sim.data.ctrl[:, env.maps["leg_act"]],
                direct_joint_targets(torch.full((2, 12), 0.2, device="cuda:0"), env.leg_limits),
            )
            assert env.leg_offset.abs().sum() == 0
        else:
            assert env.leg_offset.abs().sum() > 0
        assert env.closure.min() > 0.9
        assert info["external_crank_power_W"].abs().sum() == 0
        other_q, other_v = env.q[1].clone(), env.v[1].clone()
        other_legs = env.leg_offset[1].clone()
        other_target = env.leg_target[1].clone()
        env.reset(torch.tensor([0], device="cuda:0"))
        torch.testing.assert_close(env.q[1], other_q, atol=0, rtol=0)
        torch.testing.assert_close(env.v[1], other_v, atol=0, rtol=0)
        torch.testing.assert_close(env.leg_offset[1], other_legs, atol=0, rtol=0)
        torch.testing.assert_close(env.leg_target[1], other_target, atol=0, rtol=0)
        assert env.leg_offset[0].abs().sum() == 0


def test_rolling_contact_is_pair_specific_and_invalidates_old_reference(whole_source, tmp_path):
    cfg, phase, original = whole_source
    path = tmp_path / "settled.json"
    structural_reference(cfg, phase, original, path)
    rolling_cfg = dict(cfg, wheel_ground_rolling_friction_m=0.00405)
    model, *_ = build_rider_model(rolling_cfg, phase, load_settled=False)
    assert model.npair == 2 and np.all(model.pair_dim == 6)
    assert np.all(model.pair_friction[:, 2] >= 1e-5)
    np.testing.assert_allclose(model.pair_friction[:, 3:], 0.00405)
    assert {model.geom(g).name for g in model.pair_geom1} == {"ground"}
    assert {model.geom(g).name for g in model.pair_geom2} == {"front_wheel_tire", "rear_wheel_tire"}
    np.testing.assert_array_equal(model.geom_condim, original[0].geom_condim)
    np.testing.assert_array_equal(model.geom_friction, original[0].geom_friction)
    fingerprint = model_fingerprint(model)
    assert fingerprint != model_fingerprint(original[0])
    model.pair_friction[0, 3] *= 2
    assert model_fingerprint(model) != fingerprint
    with pytest.raises(ValueError, match="physical model"):
        build_rider_model(rolling_cfg, phase)
