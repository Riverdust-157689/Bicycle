"""B: batched GPU physics via mjlab Simulation / MuJoCo Warp.

Tensor task math also runs on CPU in tests. Training explicitly requires CUDA.
Construct and use this environment inside shared_cuda_stream().
The environment returns terminal observations BEFORE resets for SAC replay.
"""

import math

import torch

from bicycle_sim.control.hand_synergy import finger_targets_tensor
from bicycle_sim.rl.rider_model import build_rider_model


def phase_reference(phase, table):
    coordinate = torch.remainder(phase, 2 * math.pi) * 24 / (2 * math.pi)
    left = coordinate.floor().long().clamp(0, 23)
    alpha = (coordinate - left).unsqueeze(-1)
    return table[left] * (1 - alpha) + table[left + 1] * alpha


def leg_servo_reference(phase, angular_velocity, table, kv_over_kp):
    """Encode PD(q_ref, dq_ref) in a native position actuator's control input.

    Native torque is kp*(ctrl-q)-kv*dq, so ctrl=q_ref+kv/kp*dq_ref.
    This cancels reference-motion damping; it is not a separate phase command.
    """
    velocity = phase_velocity(phase, angular_velocity, table)
    return phase_reference(phase, table) + kv_over_kp * velocity


def phase_velocity(phase, angular_velocity, table):
    coordinate = torch.remainder(phase, 2 * math.pi) * 24 / (2 * math.pi)
    left = coordinate.floor().long().clamp(0, 23)
    return (table[left + 1] - table[left]) * (24 / (2 * math.pi)) * angular_velocity.unsqueeze(-1)


def task_reward(rotation, velocity_body, omega_body, action, previous, speed_objective=None):
    # Legacy moving balance, or an explicitly configured fixed speed objective.
    tilt = torch.acos(rotation[:, 2, 2].clamp(-1, 1))
    if speed_objective is None:
        moving = velocity_body[:, 0].clamp(0, 1)
        low_speed_penalty = 0
    else:
        from bicycle_sim.rl.riding_speed import speed_reward_terms

        moving, low_speed_penalty = speed_reward_terms(velocity_body[:, 0], speed_objective)
    reward = (1 + 0.5 * torch.exp(-((tilt / 0.3) ** 2))) * moving
    reward -= low_speed_penalty
    reward -= 0.01 * omega_body.square().sum(-1) + 0.005 * action.square().sum(-1)
    reward -= 0.02 * (action - previous).square().sum(-1)
    return reward, tilt


def quaternion_product(q, p):
    """WXYZ product, preserving a settled pelvis orientation under scene rotation."""
    w, x, y, z = q.unbind(-1)
    a, b, c, d = p.unbind(-1)
    return torch.stack(
        (
            w * a - x * b - y * c - z * d,
            w * b + x * a + y * d - z * c,
            w * c - x * d + y * a + z * b,
            w * d + x * c - y * b + z * a,
        ),
        -1,
    )


class RiderGpuEnv:
    num_actions = 14

    def __init__(self, cfg, *, device="cuda:0", reference_path=None):
        import warp as wp
        from mjlab.sim import Simulation, SimulationCfg

        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                "Experiment B GPU training requires visible CUDA; no CPU fallback is used"
            )
        torch.cuda.set_device(device)
        wp.set_device(device)
        self.cfg, self.device, self.num_envs = cfg, device, cfg["num_envs"]
        from bicycle_sim.rl.riding_speed import validate_speed_objective

        self.speed_objective = cfg.get("speed_objective")
        validate_speed_objective(self.speed_objective, cfg["episode_seconds"])
        self.natural_grasp = cfg.get("hand_control_enabled", False)
        self.whole_body = cfg.get("whole_body_control", False)
        self.leg_control_mode = cfg.get("leg_control_mode", "phase_residual")
        if self.leg_control_mode not in ("phase_residual", "direct_position"):
            raise ValueError("Unknown leg control mode")
        self.direct_legs = self.leg_control_mode == "direct_position"
        self.observe_pedal_joints = cfg.get("observe_pedal_joints", False)
        if (self.direct_legs or self.observe_pedal_joints) and not self.whole_body:
            raise ValueError("Direct leg targets/pedal observations require the whole-body task")
        self.num_actions = 35 if self.whole_body else 20 if self.natural_grasp else 14
        self.training_transitions = 0
        kwargs = {} if reference_path is None else {"reference_path": reference_path}
        self.m, q0, ctrl0, maps = build_rider_model(cfg, **kwargs)
        self.q0 = torch.tensor(q0, dtype=torch.float32, device=device)
        self.ctrl0 = torch.tensor(ctrl0, dtype=torch.float32, device=device)
        self.maps = {
            k: torch.as_tensor(
                v,
                device=device,
                dtype=torch.float32
                if k in ("leg_table", "hand_open", "hand_closed")
                else torch.long,
            )
            for k, v in maps.items()
        }
        self.arm_nominal = (
            self.ctrl0[self.maps["arm_act"]].clone()
            if self.natural_grasp
            else self.q0[self.maps["arm_q"]].clone()
        )
        self.arm_limits = torch.as_tensor(
            self.m.jnt_range[self.m.actuator_trnid[maps["arm_act"], 0]],
            dtype=torch.float32,
            device=device,
        )
        leg_act = maps["leg_act"]
        self.leg_kv_over_kp = torch.as_tensor(
            -self.m.actuator_biasprm[leg_act, 2] / self.m.actuator_gainprm[leg_act, 0],
            dtype=torch.float32,
            device=device,
        )
        self.leg_limits = torch.as_tensor(
            self.m.actuator_ctrlrange[leg_act],
            dtype=torch.float32,
            device=device,
        )
        sim_cfg = SimulationCfg(nconmax=cfg["nconmax"], njmax=cfg["njmax"])
        sim_cfg.mujoco.timestep = self.m.opt.timestep
        sim_cfg.mujoco.iterations = self.m.opt.iterations
        from bicycle_sim.rl.warp_capacity import configure_epa_horizon

        self.epa_horizon_capacity = configure_epa_horizon(cfg.get("epa_horizon_capacity", 24))
        self.sim = Simulation(self.num_envs, sim_cfg, self.m, device=device)
        self.dt = self.m.opt.timestep * cfg["control_decimation"]
        self.max_steps = round(cfg["episode_seconds"] / self.dt)
        self.length = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        from bicycle_sim.rl.riding_speed import EpisodeSpeedMetrics

        self.speed_metrics = (
            EpisodeSpeedMetrics(
                self.num_envs, self.max_steps, self.dt, self.speed_objective, device
            )
            if self.speed_objective is not None
            else None
        )
        self.returns = torch.zeros(self.num_envs, device=device)
        self.previous = torch.zeros(self.num_envs, self.num_actions, device=device)
        self.offset = torch.zeros(self.num_envs, 14, device=device)
        if self.whole_body:
            self.leg_offset = torch.zeros(self.num_envs, 12, device=device)
            self.leg_target = self.ctrl0[self.maps["leg_act"]].expand(self.num_envs, -1).clone()
            self.waist_offset = torch.zeros(self.num_envs, 3, device=device)
            if not self.direct_legs:
                self.leg_scale = torch.tensor(cfg["leg_offset_limit_rad"], device=device)
            self.waist_scale = torch.tensor(cfg["waist_offset_limit_rad"], device=device)
            self.waist_nominal = self.ctrl0[self.maps["waist_act"]].clone()
            self.waist_limits = torch.tensor(
                self.m.actuator_ctrlrange[maps["waist_act"]], dtype=torch.float32, device=device
            )
            self.assistance = torch.ones(self.num_envs, device=device)
            self.support_forces = torch.zeros(self.num_envs, 3, device=device)
            self.pedal_torque = torch.zeros(self.num_envs, device=device)
            self.pelvis_body = self.m.body("pelvis").id
            self.crank_body = self.m.body("crank").id
            self.seat_geom = self.m.geom("saddle").id
            self.foot_geoms = torch.tensor(
                [self.m.geom(f"{s}_riding_sole").id for s in ("left", "right")], device=device
            )
            self.pedal_geoms = torch.tensor(
                [self.m.geom(f"{s}_pedal_platform").id for s in ("left", "right")], device=device
            )
            self.pelvis_geom = torch.tensor(
                [self.m.geom_bodyid[g] == self.pelvis_body for g in range(self.m.ngeom)],
                device=device,
            )
            self.foot_body_lut = torch.tensor(
                [
                    next(
                        (
                            i
                            for i, s in enumerate(("left", "right"))
                            if self.m.body(self.m.geom_bodyid[g]).name == f"{s}_ankle_roll_link"
                        ),
                        -1,
                    )
                    for g in range(self.m.ngeom)
                ],
                device=device,
            )
        self.frame = self.m.body("bike_frame").id
        self.pelvis_q = int(self.m.joint("floating_base_joint").qposadr[0])
        self.pelvis_v = int(self.m.joint("floating_base_joint").dofadr[0])
        self.crank_q = int(self.m.joint("crank_spin").qposadr[0])
        self.steer_q = int(self.m.joint("steer").qposadr[0])
        self.steer_v = int(self.m.joint("steer").dofadr[0])
        self.spins = torch.tensor(
            [
                self.m.joint(n).dofadr[0]
                for n in ("crank_spin", "rear_wheel_spin", "front_wheel_spin")
            ],
            device=device,
        )
        self.wheel_v = self.spins[1:]
        self.crank_act = self.m.actuator("crank_drive").id
        self.tires = torch.tensor(
            [self.m.geom(f"{s}_wheel_tire").id for s in ("rear", "front")], device=device
        )
        self.floor = self.m.geom("ground").id
        gyro = self.m.sensor("bike_gyro")
        self.gyro_slice = slice(int(gyro.adr[0]), int(gyro.adr[0]) + 3)
        if self.natural_grasp:
            self.closure = torch.full(
                (self.num_envs, 6), cfg["hand_nominal_closure"], device=device
            )
            self.finger_forces = torch.zeros_like(self.closure)
            self.hand_limits = torch.as_tensor(
                self.m.actuator_ctrlrange[maps["hand_act"]], dtype=torch.float32, device=device
            )
            self.grips = [self.m.geom(f"{s}_grip_surface").id for s in ("left", "right")]
            finger_lut = [-1] * self.m.ngeom
            for g in range(self.m.ngeom):
                body = self.m.body(self.m.geom_bodyid[g]).name
                for s, side in enumerate(("left", "right")):
                    for f, finger in enumerate(("thumb", "index", "middle")):
                        if body.startswith(f"{side}_hand_{finger}_"):
                            finger_lut[g] = s * 3 + f
            self.finger_lut = torch.tensor(finger_lut, device=device)
            capacity = self.sim.wp_data.naconmax
            self._contact_ids_tensor = torch.arange(capacity, dtype=torch.int32, device=device)
            self._contact_ids_wp = wp.from_torch(self._contact_ids_tensor)
            self._contact_slots = self._contact_ids_tensor.long()
            self._forces_wp = wp.zeros(capacity, dtype=wp.spatial_vector, device=device)
            self._forces = wp.to_torch(self._forces_wp)
        self.reset(torch.arange(self.num_envs, device=device))
        self.num_observations = self.observe().shape[-1]

    @property
    def model(self):
        return self.m

    @property
    def q(self):
        return self.sim.data.qpos[:]

    @property
    def v(self):
        return self.sim.data.qvel[:]

    def reset(self, ids):
        ids = ids.long()
        if ids.numel() == 0:
            return self.observe()
        self.sim.reset(ids)
        count = ids.numel()
        q = self.q0.expand(count, -1).clone()
        yaw = (torch.rand(count, device=self.device) * 2 - 1) * math.pi
        roll = (torch.rand(count, device=self.device) * 2 - 1) * self.cfg["initial_roll_rad"]
        cy, sy, cr, sr = (
            torch.cos(yaw / 2),
            torch.sin(yaw / 2),
            torch.cos(roll / 2),
            torch.sin(roll / 2),
        )
        quat = torch.stack((cy * cr, cy * sr, sy * sr, sy * cr), -1)
        q[:, 3:7] = quat
        # Rotate both free roots together: preserve the local seat/grip geometry.
        p = self.q0[self.pelvis_q : self.pelvis_q + 3]
        px = p[0].expand(count)
        py = p[1] * torch.cos(roll) - p[2] * torch.sin(roll)
        pz = p[1] * torch.sin(roll) + p[2] * torch.cos(roll)
        q[:, self.pelvis_q : self.pelvis_q + 3] = torch.stack(
            (
                torch.cos(yaw) * px - torch.sin(yaw) * py,
                torch.sin(yaw) * px + torch.cos(yaw) * py,
                pz,
            ),
            -1,
        )
        q[:, self.pelvis_q + 3 : self.pelvis_q + 7] = (
            quaternion_product(quat, self.q0[self.pelvis_q + 3 : self.pelvis_q + 7])
            if self.natural_grasp
            else quat
        )
        v = torch.zeros(count, self.m.nv, device=self.device)
        speed = self.cfg["initial_speed_mps"]
        v[:, 0], v[:, 1] = speed * torch.cos(yaw), speed * torch.sin(yaw)
        v[:, self.pelvis_v : self.pelvis_v + 3] = v[:, :3]
        v[:, self.wheel_v] = speed / 0.27
        v[:, self.spins[0]] = self.cfg["crank_speed_rad_s"]
        if self.whole_body:
            if not self.direct_legs:
                v[:, self.maps["leg_v"]] = phase_velocity(
                    q[:, self.crank_q], v[:, self.spins[0]], self.maps["leg_table"]
                )
            # Match a stationary pedal platform orientation while the crank
            # already rotates; the pedal hinges stay passive after reset.
            v[:, self.maps["pedal_v"]] = -v[:, self.spins[0], None]
        self.sim.data.qpos[ids] = q
        self.sim.data.qvel[ids] = v
        self.sim.data.ctrl[ids] = self.ctrl0.expand(count, -1)
        self.offset[ids], self.previous[ids] = 0, 0
        self.length[ids], self.returns[ids] = 0, 0
        if self.speed_metrics is not None:
            self.speed_metrics.reset(ids)
        if self.whole_body:
            from bicycle_sim.rl.whole_body import assistance_at

            self.leg_offset[ids], self.waist_offset[ids] = 0, 0
            self.leg_target[ids] = self.ctrl0[self.maps["leg_act"]]
            self.assistance[ids] = assistance_at(self.training_transitions, self.cfg)
        if self.natural_grasp:
            self.closure[ids] = self.cfg["hand_nominal_closure"]
        self.sim.forward()
        if self.natural_grasp:
            self.update_finger_contacts()
        return self.observe()

    def state(self):
        R = self.sim.data.xmat[:, self.frame].reshape(-1, 3, 3)
        velocity = torch.bmm(R.transpose(1, 2), self.v[:, :3].unsqueeze(-1)).squeeze(-1)
        omega = self.sim.data.sensordata[:, self.gyro_slice]
        return R, velocity, omega

    def observe(self):
        R, velocity, omega = self.state()
        phase = self.q[:, self.crank_q]
        qidx, vidx = self.maps["robot_q"], self.maps["robot_v"]
        pieces = [
            -R[:, 2, :],
            velocity / 4,
            omega / 10,
            self.q[:, qidx] - self.q0[qidx],
            self.v[:, vidx] / 10,
            self.q[:, self.steer_q, None],
            self.v[:, self.steer_v, None] / 10,
            torch.cos(phase)[:, None],
            torch.sin(phase)[:, None],
            self.v[:, self.spins] / 15,
            self.previous,
            self.offset / self.cfg["arm_offset_limit_rad"],
        ]
        if self.natural_grasp:
            pieces.extend(
                (self.closure, (self.finger_forces / self.cfg["finger_force_scale_N"]).clamp(0, 4))
            )
        if self.whole_body:
            from bicycle_sim.rl.whole_body import normalized_joint_targets, relative_pelvis_features

            pelvis_R = self.sim.data.xmat[:, self.pelvis_body].reshape(-1, 3, 3)
            pos, quat, vel, omega_rel = relative_pelvis_features(
                self.q, self.v, R, pelvis_R, self.pelvis_q, self.pelvis_v
            )
            foot_delta = (
                self.sim.data.geom_xpos[:, self.foot_geoms]
                - self.sim.data.geom_xpos[:, self.pedal_geoms]
            )
            foot_delta = torch.bmm(foot_delta, R).reshape(self.num_envs, 6)
            pieces.extend(
                (
                    normalized_joint_targets(self.leg_target, self.leg_limits)
                    if self.direct_legs
                    else self.leg_offset / self.leg_scale,
                    self.waist_offset / self.waist_scale,
                    pos,
                    quat,
                    vel / 4,
                    omega_rel / 10,
                    (self.support_forces / self.cfg["contact_force_scale_N"]).clamp(0, 4),
                    foot_delta,
                    self.assistance[:, None],
                )
            )
            if self.observe_pedal_joints:
                from bicycle_sim.rl.whole_body import pedal_joint_features

                pieces.append(
                    pedal_joint_features(
                        self.q[:, self.maps["pedal_q"]], self.v[:, self.maps["pedal_v"]]
                    )
                )
        return torch.cat(pieces, -1).clone()

    def update_finger_contacts(self):
        import mujoco_warp as mjwarp

        self._forces.zero_()
        mjwarp.contact_force(
            self.sim.wp_model, self.sim.wp_data, self._contact_ids_wp, False, self._forces_wp
        )
        contact = self.sim.data.contact
        geom = contact.geom[:].long()
        world = contact.worldid[:].long()
        left = (geom[:, 0] == self.grips[0]) | (geom[:, 1] == self.grips[0])
        right = (geom[:, 0] == self.grips[1]) | (geom[:, 1] == self.grips[1])
        other = torch.where(
            (geom[:, 0] == self.grips[0]) | (geom[:, 0] == self.grips[1]), geom[:, 1], geom[:, 0]
        )
        finger = self.finger_lut[other.clamp(0, self.m.ngeom - 1)]
        valid = (
            (self._contact_slots < self.sim.data.nacon[0])
            & (world >= 0)
            & (world < self.num_envs)
            & (other >= 0)
        )
        valid &= (finger >= 0) & ((left & (finger < 3)) | (right & (finger >= 3)))
        target = world.clamp(0, self.num_envs - 1) * 6 + finger.clamp_min(0)
        self.finger_forces.zero_()
        self.finger_forces.flatten().scatter_add_(
            0, target, torch.where(valid, self._forces[:, 0].clamp_min(0), 0)
        )
        if self.whole_body:
            self.update_support_contacts(geom, world)

    def update_support_contacts(self, geom, world):
        """Seat/foot normal loads and robot-to-pedal generalized crank torque."""
        from bicycle_sim.rl.whole_body import contact_crank_torque

        contact = self.sim.data.contact
        valid = (
            (self._contact_slots < self.sim.data.nacon[0]) & (world >= 0) & (world < self.num_envs)
        )
        slots = world.clamp(0, self.num_envs - 1)
        g1, g2 = geom[:, 0].clamp(0, self.m.ngeom - 1), geom[:, 1].clamp(0, self.m.ngeom - 1)
        normal = self._forces[:, 0].clamp_min(0)
        seat = ((g1 == self.seat_geom) & self.pelvis_geom[g2]) | (
            (g2 == self.seat_geom) & self.pelvis_geom[g1]
        )
        channels = [seat]
        foot_masks = []
        for side in range(2):
            mask = ((g1 == self.pedal_geoms[side]) & (self.foot_body_lut[g2] == side)) | (
                (g2 == self.pedal_geoms[side]) & (self.foot_body_lut[g1] == side)
            )
            channels.append(mask)
            foot_masks.append(mask)
        self.support_forces.zero_()
        for i, mask in enumerate(channels):
            self.support_forces[:, i].scatter_add_(0, slots, torch.where(valid & mask, normal, 0))
        frame = contact.frame[:].reshape(-1, 3, 3)
        pedal_second = (g2 == self.pedal_geoms[0]) | (g2 == self.pedal_geoms[1])
        axis = self.sim.data.xmat[slots, self.crank_body].reshape(-1, 3, 3)[:, :, 1]
        torque = contact_crank_torque(
            self._forces,
            frame,
            contact.pos[:],
            self.sim.data.xpos[slots, self.crank_body],
            axis,
            pedal_second,
        )
        self.pedal_torque.zero_()
        self.pedal_torque.scatter_add_(
            0, slots, torch.where(valid & (foot_masks[0] | foot_masks[1]), torque, 0)
        )

    def bad_ground_contact(self):
        contacts = self.sim.data.contact
        geom = contacts.geom[:]
        world = contacts.worldid[:].long()
        valid = torch.arange(geom.shape[0], device=self.device) < self.sim.data.nacon[0]
        other = torch.where(geom[:, 0] == self.floor, geom[:, 1], geom[:, 0])
        bad = valid & ((geom[:, 0] == self.floor) | (geom[:, 1] == self.floor))
        bad &= (other != self.tires[0]) & (other != self.tires[1]) & (contacts.dist[:] <= 0)
        out = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        out.scatter_add_(0, world.clamp(0, self.num_envs - 1), bad.int())
        return out > 0

    def set_leg_actions(self, action):
        from bicycle_sim.rl.whole_body import LEG, advance_offset, direct_joint_targets

        if self.direct_legs:
            self.leg_target = direct_joint_targets(action[:, LEG], self.leg_limits)
        else:
            self.leg_offset = advance_offset(
                self.leg_offset,
                action[:, LEG],
                self.leg_scale,
                self.cfg["leg_target_rate_rad_s"],
                self.dt,
            )

    def apply_leg_control(self):
        if self.direct_legs:
            self.sim.data.ctrl[:, self.maps["leg_act"]] = self.leg_target
            return
        reference = phase_reference(self.q[:, self.crank_q], self.maps["leg_table"])
        if self.cfg.get("leg_velocity_feedforward", False):
            reference = leg_servo_reference(
                self.q[:, self.crank_q],
                self.v[:, self.spins[0]],
                self.maps["leg_table"],
                self.leg_kv_over_kp,
            )
        if self.whole_body:
            reference = reference + self.leg_offset
        self.sim.data.ctrl[:, self.maps["leg_act"]] = reference.clamp(
            self.leg_limits[:, 0], self.leg_limits[:, 1]
        )

    def step(self, actions):
        if actions.shape != (self.num_envs, self.num_actions) or not torch.isfinite(actions).all():
            raise ValueError(f"Expected finite (num_envs,{self.num_actions}) actions")
        # Retain the last valid control boundary for rare physics failures.
        before_state = {
            "qpos_before": self.q.detach().clone(),
            "qvel_before": self.v.detach().clone(),
            "ctrl_before": self.sim.data.ctrl[:].detach().clone(),
        }
        action = actions.clamp(-1, 1)
        if self.whole_body:
            from bicycle_sim.rl.whole_body import WAIST, advance_offset

            self.set_leg_actions(action)
            self.waist_offset = advance_offset(
                self.waist_offset,
                action[:, WAIST],
                self.waist_scale,
                self.cfg["waist_target_rate_rad_s"],
                self.dt,
            )
            waist_target = self.waist_nominal + self.waist_offset
            self.sim.data.ctrl[:, self.maps["waist_act"]] = waist_target.clamp(
                self.waist_limits[:, 0], self.waist_limits[:, 1]
            )
        desired = action[:, :14] * self.cfg["arm_offset_limit_rad"]
        delta = self.cfg["arm_target_rate_rad_s"] * self.dt
        self.offset += (desired - self.offset).clamp(-delta, delta)
        target = self.arm_nominal + self.offset
        self.sim.data.ctrl[:, self.maps["arm_act"]] = target.clamp(
            self.arm_limits[:, 0], self.arm_limits[:, 1]
        )
        if self.natural_grasp:
            desired_closure = (
                self.cfg["hand_nominal_closure"]
                + self.cfg["hand_closure_action_scale"] * action[:, 14:20]
            ).clamp(0, 1)
            change = self.cfg["hand_closure_rate_per_s"] * self.dt
            self.closure += (desired_closure - self.closure).clamp(-change, change)
            self.sim.data.ctrl[:, self.maps["hand_act"]] = finger_targets_tensor(
                self.closure,
                self.maps["hand_open"],
                self.maps["hand_closed"],
                self.maps["hand_columns"],
                self.hand_limits,
            )
        bad_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        external_work = torch.zeros(self.num_envs, device=self.device)
        leg_work = torch.zeros_like(external_work)
        for _ in range(self.cfg["control_decimation"]):
            self.apply_leg_control()
            if self.whole_body:
                from bicycle_sim.rl.whole_body import assistance_torque

                self.sim.data.ctrl[:, self.crank_act] = assistance_torque(
                    self.v[:, self.spins[0]], self.assistance, self.cfg
                )
            else:
                self.sim.data.ctrl[:, self.crank_act] = self.cfg["crank_speed_rad_s"]
            self.sim.step()
            if self.whole_body:
                external_work += (
                    self.sim.data.actuator_force[:, self.crank_act] * self.v[:, self.spins[0]]
                ).clamp_min(0) * self.m.opt.timestep
                leg_work += (
                    self.sim.data.qfrc_actuator[:, self.maps["leg_v"]]
                    * self.v[:, self.maps["leg_v"]]
                ).clamp_min(0).sum(-1) * self.m.opt.timestep
            bad_contact |= self.bad_ground_contact()
        self.sim.forward()
        if (
            not torch.isfinite(self.q).all()
            or not torch.isfinite(self.v).all()
            or torch.any(self.sim.data.overflow[:] != 0)
        ):
            from bicycle_sim.rl.physics_diagnostics import RiderPhysicsError, physics_failure_report

            report = physics_failure_report(self.q, self.v, self.sim.data.overflow[:])
            report["capacities"] = {
                name: int(getattr(self.sim.wp_data, name))
                for name in ("naconmax", "naccdmax", "njmax")
                if hasattr(self.sim.wp_data, name)
            }
            report["capacities"]["epa_horizon"] = self.epa_horizon_capacity
            raise RiderPhysicsError(
                report,
                dict(
                    before_state,
                    qpos_after=self.q.detach().clone(),
                    qvel_after=self.v.detach().clone(),
                    ctrl_after=self.sim.data.ctrl[:].detach().clone(),
                    actions=action.detach().clone(),
                ),
            )
        R, velocity, omega = self.state()
        rate, tilt = task_reward(R, velocity, omega, action, self.previous, self.speed_objective)
        if self.natural_grasp:
            self.update_finger_contacts()
            rate += self.cfg["finger_contact_reward_weight"] * (
                self.finger_forces > 0.5
            ).float().mean(-1)
        self.length += 1
        terminated = (tilt > self.cfg["fall_angle_rad"]) | bad_contact
        rider_failure = torch.zeros_like(terminated)
        if self.whole_body:
            pelvis_R = self.sim.data.xmat[:, self.pelvis_body].reshape(-1, 3, 3)
            rider_tilt = torch.acos(pelvis_R[:, 2, 2].clamp(-1, 1))
            separation = torch.linalg.vector_norm(
                self.sim.data.xpos[:, self.pelvis_body]
                - self.sim.data.geom_xpos[:, self.seat_geom],
                dim=-1,
            )
            rider_failure = (rider_tilt > self.cfg["rider_fall_angle_rad"]) | (
                separation > self.cfg["rider_separation_limit_m"]
            )
            terminated |= rider_failure
            rate += self.cfg["pedaling_reward_weight"] * (
                self.v[:, self.spins[0]] / self.cfg["crank_speed_rad_s"]
            ).clamp(0, 1)
        truncated = (self.length >= self.max_steps) & ~terminated
        reward = rate * self.dt - 5 * terminated.float()
        self.returns += reward
        self.previous.copy_(action)
        info = {
            "tilt": tilt,
            "tilt_failure": tilt > self.cfg["fall_angle_rad"],
            "ground_failure": bad_contact,
            "steer_angle": self.q[:, self.steer_q].clone(),
            "forward_speed": velocity[:, 0],
            "episode_return": self.returns.clone(),
            "episode_seconds": self.length * self.dt,
        }
        if self.speed_metrics is not None:
            info.update(self.speed_metrics.update(velocity[:, 0], self.length, truncated))
            from bicycle_sim.rl.riding_speed import speed_reward_terms

            score, penalty = speed_reward_terms(velocity[:, 0], self.speed_objective)
            info.update(speed_target_score=score, low_speed_penalty_rate=penalty)
        if self.natural_grasp:
            loaded = self.finger_forces > 0.5
            info.update(
                finger_contact_fraction=loaded.float().mean(-1),
                both_hands_contact=loaded[:, :3].any(-1) & loaded[:, 3:].any(-1),
                finger_normal_force_N=self.finger_forces.clone(),
                hand_closure=self.closure.clone(),
            )
        if self.whole_body:
            info.update(
                rider_failure=rider_failure,
                seat_normal_force_N=self.support_forces[:, 0].clone(),
                foot_normal_force_N=self.support_forces[:, 1:].clone(),
                crank_assistance=self.assistance.clone(),
                crank_speed_rad_s=self.v[:, self.spins[0]].clone(),
                external_crank_power_W=external_work / self.dt,
                leg_positive_power_W=leg_work / self.dt,
                foot_to_crank_torque_Nm=self.pedal_torque.clone(),
                foot_to_crank_power_W=self.pedal_torque * self.v[:, self.spins[0]],
            )
            self.training_transitions += self.num_envs
        return (
            self.observe(),
            reward[:, None],
            terminated[:, None],
            truncated[:, None],
            info,
        )
