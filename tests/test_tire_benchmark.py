"""Ensure comparison fixtures do not silently change mass or inertia."""

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from bicycle_sim.cli.benchmark_tires import roller, whole_bike

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/bicycle_v0.json").read_text())


@pytest.mark.parametrize("segments", [0, 48, 96, -192])
def test_tire_comparison_holds_mass_inertia_and_configuration(segments):
    before = copy.deepcopy(CONFIG)
    baseline = whole_bike(CONFIG, 0)
    candidate = whole_bike(CONFIG, segments)
    assert CONFIG == before
    assert (candidate.nq, candidate.nv, candidate.nu) == (baseline.nq, baseline.nv, baseline.nu)
    np.testing.assert_allclose(candidate.body_mass, baseline.body_mass)
    np.testing.assert_allclose(candidate.body_inertia, baseline.body_inertia)
    body = mujoco.mj_name2id(candidate, mujoco.mjtObj.mjOBJ_BODY, "rear_wheel")
    geoms = np.where(candidate.geom_bodyid == body)[0]
    assert len(geoms) == (segments if segments > 0 else 1)
    for g in geoms:
        assert candidate.geom_condim[g] == 3
        np.testing.assert_allclose(candidate.geom_friction[g], [0.8, 0, 0])


@pytest.mark.parametrize("segments", [0, 96, -192])
def test_guided_roller_has_specified_total_mass_and_unactuated_spin(segments):
    m = roller(CONFIG, segments, 21.6971176, 0.001)
    assert m.body_mass.sum() == pytest.approx(21.6971176)
    assert (m.nq, m.nv, m.nu) == (3, 3, 0)
