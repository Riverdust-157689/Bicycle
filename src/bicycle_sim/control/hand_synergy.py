"""Six finger closure inputs for the existing, fully actuated Dex3 motors.

This is software synergy with compliant joint PD, not a mechanical tendon model.
Contact can prevent joints reaching the targets; original torque limits remain.
"""

import numpy as np


def finger_targets_tensor(closure, opened, closed, columns, limits):
    """Batched equivalent of the six-input controller, on the caller's device."""
    import torch

    fraction = closure.clamp(0, 1)[:, columns.clamp_min(0)]
    fraction = torch.where(columns[None] >= 0, fraction, torch.zeros_like(fraction))
    return (opened + fraction * (closed - opened)).clamp(limits[:, 0], limits[:, 1])


class FingerSynergy:
    order = (
        "left_thumb",
        "left_index",
        "left_middle",
        "right_thumb",
        "right_index",
        "right_middle",
    )

    def __init__(self, model, nominal_qpos, *, kp=5.0, kv=0.15, closure_rate=0.5):
        if kp <= 0 or kv < 0 or closure_rate <= 0:
            raise ValueError("Invalid finger controller gains/rate")
        self.rate = closure_rate
        self.closure = np.zeros(6)
        names, closed, columns = [], [], []
        for side_index, side in enumerate(("left", "right")):
            sign = 1 if side == "left" else -1
            for finger_index, finger in enumerate(("thumb", "index", "middle")):
                parts = (0, 1, 2) if finger == "thumb" else (0, 1)
                angles = (
                    (0.0, sign * 1.0, sign * 1.2)
                    if finger == "thumb"
                    else (-sign * 1.5, -sign * 1.6)
                )
                for part, angle in zip(parts, angles, strict=True):
                    names.append(f"{side}_hand_{finger}_{part}_joint")
                    closed.append(angle)
                    columns.append(
                        side_index * 3 + finger_index
                        if not (finger == "thumb" and part == 0)
                        else -1
                    )
        joints = [model.joint(name).id for name in names]
        self.qids = model.jnt_qposadr[joints]
        self.vids = model.jnt_dofadr[joints]
        self.aids = np.array([model.actuator(name).id for name in names])
        self.nominal = np.asarray(nominal_qpos)[self.qids].copy()
        self.limits = model.jnt_range[joints].copy()
        self.closed = np.clip(closed, self.limits[:, 0], self.limits[:, 1])
        self.columns = np.array(columns)
        # Only hand motors are changed. Keep mass, joint friction and limits.
        model.actuator_gainprm[self.aids, 0] = kp
        model.actuator_biasprm[self.aids, 1] = -kp
        model.actuator_biasprm[self.aids, 2] = -kv

    def targets(self, closure):
        closure = np.asarray(closure, dtype=float)
        if closure.shape != (6,) or not np.isfinite(closure).all():
            raise ValueError("Expected six finite finger closures")
        u = np.clip(closure, 0, 1)
        fraction = np.where(self.columns >= 0, u[np.maximum(self.columns, 0)], 0)
        return np.clip(
            self.nominal + fraction * (self.closed - self.nominal),
            self.limits[:, 0],
            self.limits[:, 1],
        )

    def apply(self, data, closure, dt):
        # Validate before clipping; invalid commands must not mutate state.
        self.targets(closure)
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and positive")
        self.closure += np.clip(
            np.clip(closure, 0, 1) - self.closure, -self.rate * dt, self.rate * dt
        )
        data.ctrl[self.aids] = self.targets(self.closure)


def remove_palm_connections(root):
    """Remove only the two ideal palm connections, keeping the seat weld."""
    equality = root.find("equality")
    if equality is None:
        raise ValueError("Expected diagnostic equality section")
    for side in ("left", "right"):
        node = equality.find(f"connect[@name='{side}_ideal_grasp']")
        if node is None:
            raise ValueError(f"Missing {side} ideal palm connection")
        equality.remove(node)
