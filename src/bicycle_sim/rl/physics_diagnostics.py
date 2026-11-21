"""Distinguish invalid states from MuJoCo-Warp collision/solver overflows."""

import json

import torch


class RiderPhysicsError(FloatingPointError):
    def __init__(self, report, states):
        self.report = report
        self.states = states
        super().__init__("GPU physics stopped: " + json.dumps(report, separators=(",", ":")))


def physics_failure_report(qpos, qvel, overflow):
    bad_q = (~torch.isfinite(qpos).all(-1)).nonzero().flatten().tolist()
    bad_v = (~torch.isfinite(qvel).all(-1)).nonzero().flatten().tolist()
    flags = overflow.detach().cpu().flatten().tolist()
    from mujoco_warp._src.types import OverflowType

    entries = []
    for world, value in enumerate(flags):
        value = int(value)
        if value:
            names = [bit.name for bit in OverflowType if value & int(bit)]
            known = sum(int(bit) for bit in OverflowType)
            entries.append(
                {"env": world, "code": value, "flags": names, "unknown_bits": value & ~known}
            )
    if not bad_q and not bad_v and not entries:
        return None
    return {"nonfinite_qpos_envs": bad_q, "nonfinite_qvel_envs": bad_v, "overflows": entries}
