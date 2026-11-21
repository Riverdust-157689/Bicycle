"""CPU snapshot diagnostics for B: no-action, random, policy and coordinated arms.

This is a controllability/integration probe, not a GPU validation or an algorithm
benchmark. A simple feedback controller selects a steering reference; bounded
arm IK targets transmit it through the same local palm constraints.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.special import ndtr

from bicycle_sim.cli.check_rider_steering import build_rig
from bicycle_sim.rl.rider_model import ARM_PARTS, build_rider_model
from bicycle_sim.rl.skrl_agents import Policy, SquashedPolicy, deterministic_actions


def arm_references(steer_limit=0.25, arm_limit=0.25):
    m, q0, _ = build_rig()
    joints = [m.joint(f"{s}_{p}_joint").id for s in ("left", "right") for p in ARM_PARTS]
    qi = m.jnt_qposadr[joints]
    steer = m.joint("steer").qposadr[0]
    sites = [m.site(f"{s}_grasp_reference").id for s in ("left", "right")]
    anchors = [m.site(f"{s}_ideal_grasp_anchor").id for s in ("left", "right")]
    physical_limit = min(-m.jnt_range[m.joint("steer").id, 0], m.jnt_range[m.joint("steer").id, 1])
    if not 0 < steer_limit <= physical_limit + 1e-9 or arm_limit <= 0:
        raise ValueError(
            "IK steering limit must be within the vehicle joint range; arm limit must be positive"
        )
    angles = np.array([-1, -0.6, -0.2, 0, 0.2, 0.6, 1]) * steer_limit
    offsets = []
    residuals = []
    for angle in angles:
        d = mujoco.MjData(m)
        d.qpos[:] = q0
        d.qpos[steer] = angle
        mujoco.mj_kinematics(m, d)
        target = d.site_xpos[anchors].copy()

        def residual(x, d=d, target=target):
            d.qpos[qi] = x
            mujoco.mj_kinematics(m, d)
            return np.r_[(d.site_xpos[sites] - target).ravel(), 0.002 * (x - q0[qi])]

        lo = np.maximum(m.jnt_range[joints, 0] + 1e-6, q0[qi] - arm_limit)
        hi = np.minimum(m.jnt_range[joints, 1] - 1e-6, q0[qi] + arm_limit)
        fit = least_squares(residual, q0[qi], bounds=(lo, hi), max_nfev=150)
        offsets.append(fit.x - q0[qi])
        residuals.append(float(np.linalg.norm(fit.fun[:6])))
    return angles, np.asarray(offsets), residuals


class NativeProbe:
    def __init__(self, run, checkpoint, reference):
        self.meta = json.loads((run / "run.json").read_text())
        self.cfg = self.meta["environment"]
        if self.cfg.get("hand_control_enabled", False):
            raise ValueError(
                "Use probe_rider_gpu --controller policy to record the 20-action natural-grasp task; this native probe targets the legacy 14-action task"
            )
        if hashlib.sha256(reference.read_bytes()).hexdigest() != self.meta["reference_sha256"]:
            raise ValueError("Phase reference differs from the saved run")
        template, self.q0, self.ctrl, self.maps = build_rider_model(self.cfg, reference)
        self.m = mujoco.MjModel.from_xml_path(str((run / "scene.xml").resolve()))
        if self.m.nq != template.nq or any(
            self.m.joint(j).name != template.joint(j).name for j in range(template.njnt)
        ):
            raise ValueError("Snapshot topology differs from current task mapping")
        self.algorithm = self.meta.get("algorithm", "ppo")
        policy_type = SquashedPolicy if self.algorithm == "sac" else Policy
        self.policy = policy_type(
            130,
            14,
            "cpu",
            self.cfg["hidden_dims"],
            raw_gaussian=self.cfg.get("ppo_raw_gaussian", False),
        )
        saved = torch.load(run / checkpoint, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(saved["policy"])
        self.policy.eval()
        self.frame = self.m.body("bike_frame").id
        self.crq = self.m.joint("crank_spin").qposadr[0]
        self.crv = self.m.joint("crank_spin").dofadr[0]
        self.stq = self.m.joint("steer").qposadr[0]
        self.stv = self.m.joint("steer").dofadr[0]
        self.spins = [
            self.m.joint(n).dofadr[0] for n in ("crank_spin", "rear_wheel_spin", "front_wheel_spin")
        ]
        self.gyro = int(self.m.sensor("bike_gyro").adr[0])
        self.pq = self.m.joint("floating_base_joint").qposadr[0]
        self.pv = self.m.joint("floating_base_joint").dofadr[0]
        self.arm_limits = self.m.jnt_range[self.m.actuator_trnid[self.maps["arm_act"], 0]]
        self.kvkp = (
            -self.m.actuator_biasprm[self.maps["leg_act"], 2]
            / self.m.actuator_gainprm[self.maps["leg_act"], 0]
        )
        self.floor = self.m.geom("ground").id
        self.tires = {self.m.geom(f"{s}_wheel_tire").id for s in ("rear", "front")}
        self.dt = self.m.opt.timestep * self.cfg["control_decimation"]
        self.angles, self.offsets, self.ik_errors = arm_references(
            self.cfg.get("steering_reference_limit_rad", 0.25), self.cfg["arm_offset_limit_rad"]
        )

    def rollout(self, mode, seed, *, record=False):
        m = self.m
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        d = mujoco.MjData(m)
        d.qpos[:] = self.q0
        d.ctrl[:] = self.ctrl
        yaw = rng.uniform(-np.pi, np.pi)
        roll = rng.uniform(-cfg["initial_roll_rad"], cfg["initial_roll_rad"])
        cy, sy, cr, sr = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(roll / 2), np.sin(roll / 2)
        q = np.array([cy * cr, cy * sr, sy * sr, sy * cr])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        d.qpos[3:7] = q
        d.qpos[self.pq : self.pq + 3] = R @ self.q0[self.pq : self.pq + 3]
        d.qpos[self.pq + 3 : self.pq + 7] = q
        d.qvel[:3] = R @ np.array([cfg["initial_speed_mps"], 0, 0])
        d.qvel[self.pv : self.pv + 3] = d.qvel[:3]
        d.qvel[self.spins] = [
            cfg["crank_speed_rad_s"],
            cfg["initial_speed_mps"] / 0.27,
            cfg["initial_speed_mps"] / 0.27,
        ]
        previous = np.zeros(14)
        offset = np.zeros(14)
        trace = []
        means = []
        mujoco.mj_forward(m, d)
        replay = []

        def capture(desired=float("nan"), terminated=False, truncated=False):
            R = d.xmat[self.frame].reshape(3, 3)
            phi = float(np.arctan2(R[2, 1], R[2, 2]))
            roll_rate = float(d.sensordata[self.gyro])
            pd_reference = np.clip(
                -cfg.get("controller_roll_gain", 3.0) * phi
                - cfg.get("controller_roll_rate_gain", 0.7) * roll_rate,
                -cfg.get("steering_reference_limit_rad", 0.25),
                cfg.get("steering_reference_limit_rad", 0.25),
            )
            replay.append(
                {
                    "qpos": d.qpos.copy(),
                    "qvel": d.qvel.copy(),
                    "time_s": float(d.time),
                    "desired_steer_rad": desired,
                    "diagnostic_pd_steer_rad": float(pd_reference),
                    "actual_steer_rad": float(d.qpos[self.stq]),
                    "roll_rad": phi,
                    "roll_rate_rad_s": roll_rate,
                    "action": previous.copy(),
                    "arm_requested_target_rad": self.q0[self.maps["arm_q"]]
                    + previous * cfg["arm_offset_limit_rad"],
                    "arm_executed_target_rad": d.ctrl[self.maps["arm_act"]].copy(),
                    "arm_actual_rad": d.qpos[self.maps["arm_q"]].copy(),
                    "arm_actual_torque_Nm": d.qfrc_actuator[self.maps["arm_v"]].copy(),
                    "palm_anchor_error_m": np.array(
                        [
                            np.linalg.norm(
                                d.site_xpos[m.site(f"{s}_grasp_reference").id]
                                - d.site_xpos[m.site(f"{s}_ideal_grasp_anchor").id]
                            )
                            for s in ("left", "right")
                        ]
                    ),
                    "speed_mps": float((R.T @ d.qvel[:3])[0]),
                    "episode_id": 0,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )

        if record:
            capture()
        for step in range(round(cfg["episode_seconds"] / self.dt)):
            target = float("nan")
            R = d.xmat[self.frame].reshape(3, 3)
            velocity = R.T @ d.qvel[:3]
            omega = d.sensordata[self.gyro : self.gyro + 3]
            obs = np.r_[
                -R[2, :],
                velocity / 4,
                omega / 10,
                d.qpos[self.maps["robot_q"]] - self.q0[self.maps["robot_q"]],
                d.qvel[self.maps["robot_v"]] / 10,
                d.qpos[self.stq],
                d.qvel[self.stv] / 10,
                np.cos(d.qpos[self.crq]),
                np.sin(d.qpos[self.crq]),
                d.qvel[self.spins] / 15,
                previous,
                offset / cfg["arm_offset_limit_rad"],
            ]
            if mode == "policy":
                with torch.no_grad():
                    raw = deterministic_actions(
                        self.policy, torch.as_tensor(obs, dtype=torch.float32)[None], self.algorithm
                    )[0].numpy()
                means.append(raw.copy())
            elif mode == "random":
                raw = rng.normal(0, np.exp(-1), 14)
            elif mode == "coordinated":
                # Diagnostic gains selected by a small CPU search, not learned.
                steer_limit = self.cfg.get("steering_reference_limit_rad", 0.25)
                target = np.clip(
                    -self.cfg.get("controller_roll_gain", 3.0) * np.arctan2(R[2, 1], R[2, 2])
                    - self.cfg.get("controller_roll_rate_gain", 0.7) * omega[0],
                    -steer_limit,
                    steer_limit,
                )
                raw = (
                    np.array(
                        [np.interp(target, self.angles, self.offsets[:, j]) for j in range(14)]
                    )
                    / cfg["arm_offset_limit_rad"]
                )
            else:
                raw = np.zeros(14)
            action = np.clip(raw, -1, 1)
            limit = cfg["arm_target_rate_rad_s"] * self.dt
            offset += np.clip(action * cfg["arm_offset_limit_rad"] - offset, -limit, limit)
            d.ctrl[self.maps["arm_act"]] = np.clip(
                self.q0[self.maps["arm_q"]] + offset, self.arm_limits[:, 0], self.arm_limits[:, 1]
            )
            bad = False
            for _ in range(cfg["control_decimation"]):
                coordinate = (d.qpos[self.crq] % (2 * np.pi)) * 24 / (2 * np.pi)
                i = min(23, int(coordinate))
                table = self.maps["leg_table"]
                ref = table[i] * (1 - (coordinate - i)) + table[i + 1] * (coordinate - i)
                if cfg.get("leg_velocity_feedforward", False):
                    ref += (
                        self.kvkp * (table[i + 1] - table[i]) * 24 / (2 * np.pi) * d.qvel[self.crv]
                    )
                d.ctrl[self.maps["leg_act"]] = ref
                mujoco.mj_step(m, d)
                bad |= any(
                    self.floor in (c.geom1, c.geom2)
                    and (c.geom2 if c.geom1 == self.floor else c.geom1) not in self.tires
                    and c.dist <= 0
                    for c in d.contact
                )
            mujoco.mj_forward(m, d)
            R = d.xmat[self.frame].reshape(3, 3)
            tilt = float(np.arccos(np.clip(R[2, 2], -1, 1)))
            previous = action.copy()
            trace.append(np.r_[d.time, tilt, d.qpos[self.stq], (R.T @ d.qvel[:3])[0], action])
            if record:
                capture(
                    float(target),
                    bool(tilt > cfg["fall_angle_rad"] or bad),
                    bool(step + 1 == round(cfg["episode_seconds"] / self.dt)),
                )
            if d.warning.number.any() or not np.isfinite(d.qpos).all():
                raise FloatingPointError("Native simulation warning/nonfinite state")
            if tilt > cfg["fall_angle_rad"] or bad:
                break
        if record:
            self.last_replay = {key: np.asarray([r[key] for r in replay]) for key in replay[0]}
        return (
            {
                "mode": mode,
                "seed": seed,
                "duration_s": float(d.time),
                "survived_horizon": bool(
                    step + 1 == round(cfg["episode_seconds"] / self.dt)
                    and tilt <= cfg["fall_angle_rad"]
                    and not bad
                ),
                "tilt_rad": tilt,
                "nonwheel_ground": bool(bad),
            },
            np.array(trace),
            means,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--checkpoint", default="final.pt")
    p.add_argument(
        "--reference", type=Path, default=Path("output/phase1/rider_kinematic_reference.npz")
    )
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--output", type=Path, default=Path("output/phase2/b_diagnosis"))
    args = p.parse_args()
    if args.episodes < 1:
        p.error("episodes must be positive")
    torch.set_num_threads(1)
    probe = NativeProbe(args.run, args.checkpoint, args.reference)
    results = []
    traces = {}
    means = []
    for mode in ("zero", "random", "policy", "coordinated"):
        for i in range(args.episodes):
            row, trace, mu = probe.rollout(mode, args.seed + i)
            results.append(row)
            traces[f"{mode}_{args.seed + i}"] = trace
            means.extend(mu)
        selected = [r for r in results if r["mode"] == mode]
        print(
            mode,
            {
                "mean_seconds": float(np.mean([r["duration_s"] for r in selected])),
                "survival_rate": float(np.mean([r["survived_horizon"] for r in selected])),
            },
            flush=True,
        )
    report = {
        "backend": "native_cpu_saved_scene",
        "run": str(args.run),
        "episodes": results,
        "coordinated_gains": {"roll": -3, "roll_rate": -0.7, "not_a_learned_policy": True},
        "coordinated_static_ik_error_m": probe.ik_errors,
    }
    if means and probe.algorithm == "ppo" and not probe.cfg.get("ppo_raw_gaussian", False):
        mu = np.array(means)
        std = probe.policy.log_std.detach().exp().numpy()
        clipped = ndtr((-1 - mu) / std) + 1 - ndtr((1 - mu) / std)
        report["legacy_policy_clipping_prediction"] = {
            "per_joint_fraction": float(clipped.mean()),
            "any_joint_fraction": float((1 - np.prod(1 - clipped, axis=1)).mean()),
            "mean_saturation_fraction": float((np.abs(mu) > 0.95).mean()),
            "scope": "Gaussian tail probabilities at deterministic-policy probe states, not direct GPU sample counts",
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "traces.npz", **traces)


if __name__ == "__main__":
    main()
