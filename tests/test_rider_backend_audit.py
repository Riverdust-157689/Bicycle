"""Validate state-based backend diagnostics without requiring CUDA."""

from pathlib import Path

import mujoco
import numpy as np
import torch

from bicycle_sim.cli.audit_rider_backend import (
    com_velocity,
    ground_force_acceleration,
    leg_control,
    leg_control_tensors,
    variant_model,
)
from bicycle_sim.rl.rider_gpu_env import leg_servo_reference, phase_reference


def test_whole_system_com_includes_both_free_roots():
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    <body pos="1 0 0"><freejoint/><geom type="sphere" size=".1" mass="3"/></body>
    </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qvel[0] = 2
    np.testing.assert_allclose(com_velocity(m, d.qpos, d.qvel), [0.5, 0, 0], atol=1e-12)
    d.qvel[0], d.qvel[6] = -2, 2
    np.testing.assert_allclose(com_velocity(m, d.qpos, d.qvel), [1, 0, 0], atol=1e-12)


def test_ground_force_sign_handles_both_contact_geom_orders():
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom name="ground" type="plane" size="1 1 .1"/>
    <body><freejoint/><geom type="sphere" size=".1" mass="2"/></body>
    </worldbody></mujoco>""")
    data = {
        "contact_world": np.array([0, 1]),
        "contact_geom": np.array([[0, 1], [1, 0]]),
        "contact_force_world": np.array([[2, 0, 20, 0, 0, 0], [-4, 0, -20, 0, 0, 0]]),
    }
    actual = ground_force_acceleration(m, data, np.array([[1, 0, 0], [1, 0, 0]]))
    np.testing.assert_allclose(actual, [1, 2])


def test_native_diagnostic_reconstructs_the_gpu_leg_command():
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body>
    <joint name="crank_spin"/><geom type="sphere" size=".1"/>
    <body pos="0 0 1"><joint name="leg1"/><geom type="sphere" size=".1"/>
    <body pos="0 0 1"><joint name="leg2"/><geom type="sphere" size=".1"/>
    </body></body></body></worldbody><actuator>
    <position joint="leg1" kp="100" kv="2" ctrlrange="-.4 .4"/>
    <position joint="leg2" kp="100" kv="2" ctrlrange="-.4 .4"/>
    </actuator></mujoco>""")
    theta = np.linspace(0, 2 * np.pi, 25)
    table = np.stack((0.4 * np.sin(theta), 0.4 * np.cos(theta)), -1)
    maps = {"leg_table": table, "leg_act": np.array([0, 1])}
    for feedforward in (False, True):
        for phase in (-0.1, 0.3, 2 * np.pi - 0.001):
            q, v = np.array([phase, 0, 0]), np.array([3.4, 0, 0])
            actual = leg_control(m, maps, {"leg_velocity_feedforward": feedforward}, q, v)
            angle = torch.tensor([phase], dtype=torch.float64)
            ref = torch.as_tensor(table)
            expected = (
                (
                    leg_servo_reference(angle, torch.tensor([3.4]), ref, 0.02)
                    if feedforward
                    else phase_reference(angle, ref)
                )[0]
                .clamp(-0.4, 0.4)
                .numpy()
            )
            np.testing.assert_allclose(actual, expected, atol=1e-7)
            # Exercise the same indexed assignment as the WarpBridge control
            # buffer: float64 MuJoCo limits must not promote this target.
            buffer = torch.zeros((1, m.nu), dtype=torch.float32)
            gpu_table, kvkp, limits = leg_control_tensors(m, maps, buffer)
            gpu_angle = angle.float()
            gpu_ref = (
                leg_servo_reference(gpu_angle, torch.tensor([3.4]), gpu_table, kvkp)
                if feedforward
                else phase_reference(gpu_angle, gpu_table)
            )
            command = gpu_ref.clamp(limits[:, 0], limits[:, 1])
            buffer[:, maps["leg_act"]] = command
            assert command.dtype == buffer.dtype
            np.testing.assert_allclose(buffer[0].numpy(), actual, atol=1e-6)


def test_contact_ablations_preserve_inertia_and_joint_parameters(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("""<mujoco><worldbody><geom name="ground" type="plane" size="1 1 .1"/>
    <body name="wheel"><freejoint/><geom name="rear_wheel_tire" type="sphere" size=".1"/>
    <geom name="front_wheel_tire" type="sphere" size=".1" pos="1 0 0"/>
    <geom name="other" type="box" size=".1 .1 .1" pos="0 0 1"/></body></worldbody>
    <equality><weld body1="wheel"/></equality></mujoco>""")
    baseline = variant_model(scene, "baseline")
    for variant in ("wheel_contacts_only", "no_contacts", "no_equalities"):
        m = variant_model(scene, variant)
        np.testing.assert_array_equal(m.body_mass, baseline.body_mass)
        np.testing.assert_array_equal(m.body_inertia, baseline.body_inertia)
        np.testing.assert_array_equal(m.dof_damping, baseline.dof_damping)
        assert (m.nq, m.nv, m.nu) == (baseline.nq, baseline.nv, baseline.nu)
        if variant == "wheel_contacts_only":
            assert m.geom_contype[m.geom("other").id] == 0
            assert m.geom_contype[m.geom("rear_wheel_tire").id] != 0
        elif variant == "no_contacts":
            assert m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT
        else:
            assert not m.eq_active0.any()


def test_analytic_tire_diagnostic_preserves_mass_dimensions_and_contact_settings():
    scene = Path("assets/scenes/rider_fit_diagnostic.xml")
    baseline = variant_model(scene, "baseline")
    for variant in ("ellipsoid_tires", "ellipsoid_wheel_contacts_only"):
        changed = variant_model(scene, variant)
        assert (changed.nq, changed.nv, changed.nu) == (baseline.nq, baseline.nv, baseline.nu)
        for name in ("body_mass", "body_inertia", "geom_friction", "geom_solref", "geom_solimp"):
            np.testing.assert_allclose(getattr(changed, name), getattr(baseline, name), atol=1e-10)
        for side in ("rear", "front"):
            g = changed.geom(f"{side}_wheel_tire")
            assert g.type[0] == mujoco.mjtGeom.mjGEOM_ELLIPSOID
            np.testing.assert_allclose(g.size, [0.27, 0.018, 0.27], atol=1e-7)
            d = mujoco.MjData(changed)
            mujoco.mj_forward(changed, d)
            # Bounds must be recompiled in the wheel frame; no principal-mesh
            # rotation may turn the thin axle dimension into the rolling radius.
            np.testing.assert_allclose(d.geom_xmat[g.id].reshape(3, 3), np.eye(3), atol=1e-10)
