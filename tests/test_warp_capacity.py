"""Exercise the actual backend allocator and collision kernels on Warp CPU."""

import mujoco
import numpy as np
import pytest
import warp as wp

from bicycle_sim.rl.warp_capacity import configure_epa_horizon


def test_expanded_epa_workspace_reaches_collision_kernel_without_changing_contact(monkeypatch):
    import mujoco_warp as mw
    from mujoco_warp._src import collision_convex, collision_flex

    monkeypatch.setattr(wp.config, "kernel_cache_dir", "/tmp/bicycle-warp-cache")
    monkeypatch.setattr(collision_convex, "MJ_MAX_EPAHORIZON", 24)
    monkeypatch.setattr(collision_flex, "MJ_MAX_EPAHORIZON", 24)
    with wp.ScopedDevice("cpu"):
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <geom type="box" size=".025 .025 .025"/>
          <body pos="0 0 .049"><freejoint/>
          <geom type="box" size=".025 .025 .025"/></body>
        </worldbody></mujoco>""")
        wm = mw.put_model(model)
        wd = mw.put_data(model, mujoco.MjData(model), nworld=1, nconmax=8, njmax=64)
        mw.forward(wm, wd)
        ncon = int(wd.nacon.numpy()[0])
        assert ncon > 0
        depths = wd.contact.dist.numpy()[:ncon].copy()
        configure_epa_horizon(128)
        allocations = []
        original = wp.empty

        def track_allocation(*args, **kwargs):
            shape = kwargs.get("shape", args[0] if args else None)
            if shape == (wd.naccdmax, 128):
                allocations.append(shape)
            return original(*args, **kwargs)

        monkeypatch.setattr(wp, "empty", track_allocation)
        mw.forward(wm, wd)
        assert allocations, "The real convex collision allocator did not use the override"
        assert not wd.overflow.numpy().any()
        assert int(wd.nacon.numpy()[0]) == ncon
        np.testing.assert_allclose(wd.contact.dist.numpy()[:ncon], depths, atol=1e-7)


def test_invalid_capacity_does_not_partially_change_backend():
    from mujoco_warp._src import collision_convex, collision_flex

    previous = (collision_convex.MJ_MAX_EPAHORIZON, collision_flex.MJ_MAX_EPAHORIZON)
    for invalid in (5, 257, True, 128.0):
        with pytest.raises(ValueError):
            configure_epa_horizon(invalid)
        assert previous == (collision_convex.MJ_MAX_EPAHORIZON, collision_flex.MJ_MAX_EPAHORIZON)
