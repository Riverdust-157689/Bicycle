"""Fixed 35-action layout and simple, observable crank-assistance curriculum."""

import torch

ARM = slice(0, 14)
HAND = slice(14, 20)
LEG = slice(20, 32)
WAIST = slice(32, 35)


def direct_joint_targets(action, limits):
    """Absolute joint-position targets over the physical control range."""
    return limits[:, 0] + (action.clamp(-1, 1) + 1) * 0.5 * (limits[:, 1] - limits[:, 0])


def normalized_joint_targets(target, limits):
    return (2 * (target - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1).clamp(-1, 1)


def pedal_joint_features(angle, velocity):
    """Two unbounded hinge angles, represented without a 2pi discontinuity."""
    return torch.cat((angle.cos(), angle.sin(), velocity / 15), -1)


def assistance_at(transitions, cfg):
    override = cfg.get("crank_assistance_override")
    if override is not None:
        if not 0 <= override <= 1:
            raise ValueError("Crank assistance must lie in [0,1]")
        return float(override)
    start = cfg["crank_assist_full_until_transitions"]
    end = cfg["crank_assist_zero_after_transitions"]
    if not 0 <= start < end:
        raise ValueError("Invalid assistance schedule")
    return max(0.0, min(1.0, (end - transitions) / (end - start)))


def advance_offset(offset, action, limits, rate, dt):
    return offset + (action * limits - offset).clamp(-rate * dt, rate * dt)


def assistance_torque(speed, assistance, cfg):
    desired = cfg["crank_servo_kv"] * (cfg["crank_speed_rad_s"] - speed)
    return assistance * desired.clamp(-cfg["max_crank_torque_Nm"], cfg["max_crank_torque_Nm"])


def contact_crank_torque(local_force, frame, position, crank_position, axis, pedal_is_geom2):
    world_force = torch.bmm(frame.transpose(1, 2), local_force[:, :3, None]).squeeze(-1)
    force_on_pedal = world_force * torch.where(pedal_is_geom2, 1.0, -1.0)[:, None]
    return (torch.cross(position - crank_position, force_on_pedal, dim=-1) * axis).sum(-1)


def relative_pelvis_features(q, v, bike_rotation, pelvis_rotation, pq, pv):
    from bicycle_sim.rl.rider_gpu_env import quaternion_product

    inv = q[:, 3:7] * q.new_tensor([1, -1, -1, -1])
    relative_quat = quaternion_product(inv, q[:, pq + 3 : pq + 7])
    # Fix the quaternion sign ambiguity without changing the represented pose.
    relative_quat = torch.where(relative_quat[:, :1] < 0, -relative_quat, relative_quat)
    inv_R = bike_rotation.transpose(1, 2)
    position = torch.bmm(inv_R, (q[:, pq : pq + 3] - q[:, :3]).unsqueeze(-1)).squeeze(-1)
    velocity = torch.bmm(inv_R, (v[:, pv : pv + 3] - v[:, :3]).unsqueeze(-1)).squeeze(-1)
    pelvis_world_omega = torch.bmm(pelvis_rotation, v[:, pv + 3 : pv + 6].unsqueeze(-1))
    omega = torch.bmm(inv_R, pelvis_world_omega).squeeze(-1) - v[:, 3:6]
    return position, relative_quat, velocity, omega
