"""Regression for native Warp graph capture and ordered Torch/physics access."""

import pytest
import torch

from bicycle_sim.rl.cuda_context import shared_cuda_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a visible CUDA device")
def test_warp_graph_capture_and_torch_tensor_access_share_native_stream():
    import mujoco
    from mjlab.sim import Simulation, SimulationCfg

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/>"
        '<geom type="sphere" size=".1" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    with shared_cuda_stream("cuda:0"):
        # Constructor captures step/forward/reset graphs where the failure occurred.
        sim = Simulation(2, SimulationCfg(nconmax=8, njmax=32), model, device="cuda:0")
        sim.reset()
        height = torch.tensor([1.0, 2.0], device="cuda:0")
        sim.data.qpos[:, 2] = height
        sim.forward()
        torch.testing.assert_close(sim.data.xpos[:, 1, 2], height)
        sim.step()
        assert torch.all(sim.data.qvel[:, 2] < 0)
        assert torch.all(sim.data.qpos[:, 2] < height)
