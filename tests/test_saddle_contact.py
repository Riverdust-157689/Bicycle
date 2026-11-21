"""Seat bench preserves the robot and uses unilateral contact, not attachment."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from bicycle_sim.cli.audit_saddle_contact import make_bench, pelvis_hull, run_case
from bicycle_sim.rl.cuda_context import shared_cuda_stream
from bicycle_sim.rl.rider_model import LEG_PARTS, build_rider_model


@pytest.fixture
def source(tmp_path):
    cfg = json.loads(Path("configs/rl/bicycle_b_contact_sac.json").read_text())
    original = mujoco.MjModel.from_xml_path("assets/scenes/rider_fit_diagnostic.xml")
    d = mujoco.MjData(original)
    mujoco.mj_resetDataKeyframe(original, d, 0)
    reference = tmp_path / "reference.npz"
    np.savez(reference, qpos=np.tile(d.qpos, (25, 1)), phases=np.linspace(0, 2 * np.pi, 25))
    m, q, ctrl, _ = build_rider_model(cfg, reference)
    path = tmp_path / "source.xml"
    mujoco.mj_saveLastXML(str(path), m)
    return m, path.read_text(), q, ctrl, pelvis_hull(m, q)


@pytest.mark.parametrize("shape", ["raised", "ellipsoid", "pads"])
def test_seat_support_can_separate_without_changing_robot_or_inertia(source, shape):
    old, xml, q, ctrl, vertices = source
    m, initial, _, rise = make_bench(old, xml, q, ctrl, {"shape": shape}, vertices)
    assert old.neq == 1 and m.neq == 0
    assert sum(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 1
    np.testing.assert_array_equal(m.body_mass, old.body_mass)
    np.testing.assert_array_equal(m.body_inertia, old.body_inertia)
    assert 0.016 < rise < 0.019
    d = mujoco.MjData(m)
    pq = m.joint("floating_base_joint").qposadr[0]
    seat = {g for g in range(m.ngeom) if m.geom(g).name.startswith("saddle")}
    pelvis = m.body("pelvis").id

    def seat_contacts():
        return [
            c
            for c in d.contact
            if any(g in seat for g in c.geom) and any(m.geom_bodyid[g] == pelvis for g in c.geom)
        ]

    d.qpos[:] = initial
    d.qpos[pq + 2] -= 0.002
    mujoco.mj_forward(m, d)
    contacts = seat_contacts()
    assert contacts and any(c.dist < 0 for c in contacts)
    d.qpos[:] = initial
    d.qpos[pq + 2] += 0.05
    mujoco.mj_forward(m, d)
    assert not seat_contacts()


def test_repeated_bench_compilation_keeps_original_root_mapping(source):
    m, xml, q, ctrl, vertices = source
    first, q1, c1, _ = make_bench(m, xml, q, ctrl, {"shape": "raised"}, vertices)
    second, q2, c2, _ = make_bench(m, xml, q, ctrl, {"shape": "raised"}, vertices)
    assert first.nq == second.nq == m.nq - 7
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(c1, c2)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_bench_measures_real_contact_support(source, tmp_path, device):
    from contextlib import nullcontext

    if device != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    old, xml, q, ctrl, vertices = source
    m, q0, c, _ = make_bench(old, xml, q, ctrl, {"shape": "raised"}, vertices)
    leg_q = [
        q[old.joint(f"{s}_{p}_joint").qposadr[0]] for s in ("left", "right") for p in LEG_PARTS
    ]
    table = np.tile(leg_q, (25, 1))
    output = tmp_path / "short_bench"
    context = nullcontext() if device == "cpu" else shared_cuda_stream(device)
    with context:
        result = run_case(m, q0, c, table, "static", 0.1, output, device=device)
    assert result["equality_count"] == 0
    trace = json.loads((output / "results.json").read_text())["trace"]
    assert max(t["pelvis_seat_force_z_N"] for t in trace) > 1
    assert np.isfinite(np.load(output / "trajectory.npz")["qpos"]).all()
