"""CPU MuJoCo reference for experiment A. No stabilizing force or rear motor.

The policy commands crank speed and steering angle; finite torque servos realize
these commands. The native freewheel handles internal power transmission.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from bicycle_sim.models.bicycle import build_bicycle, numbers, sub


def training_tree(bike_config, cfg):
    root = build_bicycle(bike_config)
    actuator = root.find("actuator")
    if actuator is None:
        actuator = sub(root, "actuator")
    sub(
        actuator,
        "velocity",
        name="crank_drive",
        joint="crank_spin",
        kv=cfg["crank_servo_kv"],
        ctrllimited="true",
        ctrlrange=numbers([0, cfg["max_crank_speed_rad_s"]]),
        forcelimited="true",
        forcerange=numbers([-cfg["max_crank_torque_Nm"], cfg["max_crank_torque_Nm"]]),
    )
    steer_limit = bike_config["geometry"]["steer_limit_rad"]
    sub(
        actuator,
        "position",
        name="steer_drive",
        joint="steer",
        kp=cfg["steer_servo_kp"],
        kv=cfg["steer_servo_kv"],
        ctrllimited="true",
        ctrlrange=numbers([-steer_limit, steer_limit]),
        forcelimited="true",
        forcerange=numbers([-cfg["max_steer_torque_Nm"], cfg["max_steer_torque_Nm"]]),
    )
    return root


class BicycleEnv:
    """One non-autoresetting environment with normalized two-dimensional actions."""

    num_actions = 2

    def __init__(self, bike_config, cfg, *, seed=42, model=None):
        self.bike_config, self.cfg = copy.deepcopy(bike_config), copy.deepcopy(cfg)
        if cfg.get("experiment") != "A":
            raise ValueError("This environment implements experiment A only")
        if bike_config["transmission"]["model"] != "one_way_viscous_freewheel":
            raise ValueError("Riding baseline requires the freewheel model")
        self.model = (
            model
            if model is not None
            else mujoco.MjModel.from_xml_string(
                ET.tostring(training_tree(bike_config, cfg), encoding="unicode")
            )
        )
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.frame = self.model.body("bike_frame").id
        self.floor = self.model.geom("ground").id
        self.tires = {self.model.geom(f"{s}_wheel_tire").id for s in ("rear", "front")}
        names = ("steer", "crank_spin", "rear_wheel_spin", "front_wheel_spin")
        self.q = {name: self.model.joint(name).qposadr[0] for name in names}
        self.v = {name: self.model.joint(name).dofadr[0] for name in names}
        self.crank_act = self.model.actuator("crank_drive").id
        self.steer_act = self.model.actuator("steer_drive").id
        self.freewheel_act = self.model.actuator("freewheel").id
        self.decimation = bike_config["simulation_initial"]["control_decimation"]
        self.dt = self.model.opt.timestep * self.decimation
        self.max_steps = round(cfg["episode_seconds"] / self.dt)
        self.command_steps = round(cfg["command_seconds"] / self.dt)
        if min(self.max_steps, self.command_steps) < 1:
            raise ValueError("Episode and command intervals must exceed a control step")
        self.steer_limit = bike_config["geometry"]["steer_limit_rad"]
        self.steps = 0
        self.previous_action = np.zeros(2)
        self.steer_target = 0.0
        self.command = np.zeros(2)  # speed magnitude, absolute world heading
        self.reset()

    def state(self):
        m, d = self.model, self.data
        # mj_step's derived fields can precede integration; always refresh before reading.
        mujoco.mj_forward(m, d)
        rotation = d.xmat[self.frame].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.frame, velocity, 0)
        return {
            "velocity_world": velocity[3:].copy(),
            "omega_world": velocity[:3].copy(),
            "gravity_body": rotation.T @ np.array([0.0, 0.0, -1.0]),
            "heading": float(np.arctan2(rotation[1, 0], rotation[0, 0])),
            "roll": float(np.arctan2(rotation[2, 1], rotation[2, 2])),
            "pitch": float(np.arcsin(np.clip(-rotation[2, 0], -1, 1))),
        }

    def sample_command(self):
        # A world-frame heading held constant until the next command interval.
        heading = self.state()["heading"]
        self.command[:] = [
            self.rng.uniform(*self.cfg["speed_range_mps"]),
            heading
            + self.rng.uniform(-self.cfg["heading_change_rad"], self.cfg["heading_change_rad"]),
        ]
        self.command[1] = np.arctan2(np.sin(self.command[1]), np.cos(self.command[1]))

    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        yaw = self.rng.uniform(-np.pi, np.pi)
        roll = self.rng.uniform(-self.cfg["initial_roll_rad"], self.cfg["initial_roll_rad"])
        cy, sy, cr, sr = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(roll / 2), np.sin(roll / 2)
        d.qpos[3:7] = [cy * cr, cy * sr, sy * sr, sy * cr]
        speed = self.cfg["initial_speed_mps"]
        d.qvel[:2] = speed * np.array([np.cos(yaw), np.sin(yaw)])
        spin = speed / self.bike_config["geometry"]["wheel_radius"]
        for name in ("rear_wheel_spin", "front_wheel_spin"):
            d.qvel[self.v[name]] = spin
        d.qvel[self.v["crank_spin"]] = (
            spin / self.bike_config["transmission"]["rear_wheel_revolutions_per_crank_revolution"]
        )
        self.steps = 0
        self.previous_action[:] = 0
        self.steer_target = 0.0
        self.sample_command()
        return self.observe()

    def observe(self):
        s, d = self.state(), self.data
        phase = d.qpos[self.q["crank_spin"]]
        obs = np.r_[
            s["velocity_world"] / 4,
            s["omega_world"] / 10,
            np.cos(s["heading"]),
            np.sin(s["heading"]),
            s["gravity_body"],
            d.qpos[self.q["steer"]] / self.steer_limit,
            d.qvel[self.v["steer"]] / 10,
            np.cos(phase),
            np.sin(phase),
            [d.qvel[self.v[n]] / 20 for n in ("crank_spin", "rear_wheel_spin", "front_wheel_spin")],
            self.command[0] / 4,
            np.cos(self.command[1]),
            np.sin(self.command[1]),
            self.previous_action,
            self.steer_target / self.steer_limit,
            (self.command_steps - self.steps % self.command_steps) / self.command_steps,
        ]
        return obs.astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite pair [crank speed, steering angle]")
        action = np.clip(action, -1, 1)
        desired = action[1] * self.steer_limit
        change = self.cfg["max_steer_rate_rad_s"] * self.dt
        self.steer_target += np.clip(desired - self.steer_target, -change, change)
        self.data.ctrl[self.crank_act] = (action[0] + 1) / 2 * self.cfg["max_crank_speed_rad_s"]
        self.data.ctrl[self.steer_act] = self.steer_target
        mujoco.mj_step(self.model, self.data, nstep=self.decimation)
        if not np.isfinite(self.data.qpos).all() or self.data.warning.number.any():
            raise FloatingPointError("Non-finite state or MuJoCo warning; stop and inspect physics")
        self.steps += 1
        s = self.state()
        target_velocity = self.command[0] * np.array(
            [np.cos(self.command[1]), np.sin(self.command[1])]
        )
        speed_error = np.linalg.norm(s["velocity_world"][:2] - target_velocity)
        heading_error = np.arctan2(
            np.sin(s["heading"] - self.command[1]), np.cos(s["heading"] - self.command[1])
        )
        bad_contact = any(
            self.floor in (c.geom1, c.geom2)
            and (c.geom2 if c.geom1 == self.floor else c.geom1) not in self.tires
            for c in self.data.contact
        )
        terminated = bool(
            abs(s["roll"]) > self.cfg["fall_roll_rad"] or abs(s["pitch"]) > 0.6 or bad_contact
        )
        truncated = self.steps >= self.max_steps and not terminated
        terms = {
            "velocity": 2 * np.exp(-((speed_error / 0.6) ** 2)),
            "velocity_error": -self.cfg.get("velocity_error_penalty", 0.0) * speed_error**2,
            "heading": 0.5 * np.exp(-((heading_error / 0.5) ** 2)),
            "alive": 0.2,
            "lean": -0.2 * max(0, abs(s["roll"]) - 0.26) ** 2,
            "roll_rate": -0.01 * float(s["omega_world"] @ s["omega_world"]),
            "smooth": -0.01 * float(np.sum((action - self.previous_action) ** 2)),
        }
        reward = self.dt * sum(terms.values()) - float(terminated)
        self.previous_action[:] = action
        info = {
            "speed_error_mps": float(speed_error),
            "heading_error_rad": float(abs(heading_error)),
            "roll_rad": s["roll"],
            "reward_terms": terms,
            "freewheel_rear_torque_Nm": float(-self.data.actuator_force[self.freewheel_act]),
            "termination": "fall_or_nonwheel_ground"
            if terminated
            else "timeout"
            if truncated
            else "",
        }
        # Reward is for the command just executed; the next observation sees the new command.
        if self.steps % self.command_steps == 0 and not (terminated or truncated):
            self.sample_command()
        return self.observe(), float(reward), terminated, truncated, info
