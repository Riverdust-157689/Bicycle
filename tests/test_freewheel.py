"""Freewheel directionality, reaction torque and dissipative power."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from bicycle_sim.models.bicycle import build_bicycle

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/bicycle_v0.json").read_text())


def rig():
    root = build_bicycle(CONFIG, fixed_frame=True)
    root.find("worldbody").remove(root.find("./worldbody/geom[@name='ground']"))
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    m.opt.gravity[:] = 0
    return m, mujoco.MjData(m)


@pytest.mark.parametrize("crank,rear", [(-5, 0), (-5, 10), (0, 10), (2, 10)])
def test_backpedaling_and_overrunning_transmit_no_force(crank, rear):
    m, d = rig()
    c, r = m.joint("crank_spin").dofadr[0], m.joint("rear_wheel_spin").dofadr[0]
    d.qvel[c], d.qvel[r] = crank, rear
    for _ in range(500):
        mujoco.mj_step(m, d)
        assert d.actuator_force[m.actuator("freewheel").id] == 0
    if crank == 0:
        assert abs(d.qvel[c]) < 1e-12
    if rear == 0:
        assert abs(d.qvel[r]) < 1e-12
    assert not d.warning.number.any()


@pytest.mark.parametrize("crank,rear", [(1, 0), (10, 0), (3, 5), (-2, -10)])
def test_engaged_clutch_reacts_on_crank_and_cannot_supply_energy(crank, rear):
    m, d = rig()
    c, r = m.joint("crank_spin").dofadr[0], m.joint("rear_wheel_spin").dofadr[0]
    d.qvel[c], d.qvel[r] = crank, rear
    mujoco.mj_forward(m, d)
    gain = CONFIG["transmission"]["rear_wheel_revolutions_per_crank_revolution"]
    torque = d.qfrc_actuator[r]
    assert 0 < torque <= CONFIG["transmission"]["freewheel_max_rear_torque_Nm"]
    assert d.qfrc_actuator[c] == pytest.approx(-gain * torque)
    assert np.dot(d.qfrc_actuator, d.qvel) <= 0
    # Even arbitrary ctrl values cannot power the native gain=0 clutch.
    before = d.qfrc_actuator.copy()
    d.ctrl[:] = 100
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(d.qfrc_actuator, before)


def test_forward_drive_step_size_sensitivity():
    states = []
    for dt in (0.002, 0.001, 0.0005):
        m, d = rig()
        m.opt.timestep = dt
        c, r = m.joint("crank_spin").dofadr[0], m.joint("rear_wheel_spin").dofadr[0]
        d.qfrc_applied[c] = 1
        for _ in range(round(1 / dt)):
            mujoco.mj_step(m, d)
        states.append(d.qvel[[c, r]])
        assert not d.warning.number.any()
    np.testing.assert_allclose(states[0], states[-1], rtol=0.05)
