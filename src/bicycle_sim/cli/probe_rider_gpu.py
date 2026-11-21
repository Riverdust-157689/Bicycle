"""GPU controllability probe: roll feedback -> arm IK -> physical handlebar.

No learning and no direct steering actuator. Separates task/backend failures
from exploration and policy optimization before another expensive training run.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize_episodes(episodes):
    durations = [r["duration_s"] for r in episodes]
    report = {
        "completed_episodes": len(episodes),
        "survival_rate": sum(r["survived_horizon"] for r in episodes) / max(1, len(episodes)),
        "mean_episode_seconds": float(np.mean(durations)) if durations else None,
        "median_episode_seconds": float(np.median(durations)) if durations else None,
        "min_episode_seconds": min(durations, default=None),
        "max_episode_seconds": max(durations, default=None),
        "tilt_failures": sum(r["tilt_failure"] for r in episodes),
        "ground_failures": sum(r["ground_failure"] for r in episodes),
    }
    if episodes and all("final_forward_speed_mps" in r for r in episodes):
        report["mean_final_forward_speed_mps"] = float(
            np.mean([r["final_forward_speed_mps"] for r in episodes])
        )
    if episodes and all("rider_failure" in r for r in episodes):
        report["rider_failures"] = sum(r["rider_failure"] for r in episodes)
    if episodes and all("speed_qualified_success" in r for r in episodes):
        report["speed_qualified_20s_rate"] = sum(
            r["speed_qualified_success"] for r in episodes
        ) / len(episodes)
    return report


def save_probe_recording(output, report, frames, cohort_frames):
    """Persist valid frames on both normal completion and physics interruption."""
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    for filename, values in (("trajectory.npz", frames), ("cohort_trajectory.npz", cohort_frames)):
        if values:
            np.savez_compressed(
                output / filename,
                **{key: np.asarray([frame[key] for frame in values]) for key in values[0]},
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rl/bicycle_b.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--crank-assistance", type=float, help="Whole-body fixed assistance [0,1], default zero"
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1100)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--controller", choices=("feedback", "zero", "policy"), default="feedback")
    parser.add_argument(
        "--checkpoint", type=Path, help="Policy checkpoint, default source-run/final.pt"
    )
    parser.add_argument("--source-run", type=Path, help="Use this run's saved environment config")
    parser.add_argument("--tire-collision", choices=("mesh", "ellipsoid"))
    parser.add_argument(
        "--epa-horizon",
        type=int,
        choices=range(24, 257),
        metavar="24..256",
        help="EPA boundary workspace capacity; does not change contact physics",
    )
    parser.add_argument(
        "--first-episodes-only", action="store_true", help="Evaluate one initial episode per world"
    )
    parser.add_argument("--record-env", type=int, default=0)
    parser.add_argument(
        "--record-all-envs",
        action="store_true",
        help="Also save every world's first-episode poses and foot forces",
    )
    parser.add_argument(
        "--record-only",
        action="store_true",
        help="Save a short replay without applying the 20-second survival gate",
    )
    parser.add_argument("--output", type=Path, default=Path("output/phase2/b_gpu_controller_probe"))
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        parser.error("Requires visible CUDA; no CPU fallback")
    if args.num_envs < 1 or args.steps < 1:
        parser.error("num-envs and steps must be positive")
    if not 0 <= args.record_env < args.num_envs:
        parser.error("record-env must select an existing environment")
    if args.record_all_envs and not args.first_episodes_only:
        parser.error("--record-all-envs requires --first-episodes-only")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory")
    from bicycle_sim.cli.diagnose_rider import arm_references
    from bicycle_sim.rl.cuda_context import shared_cuda_stream
    from bicycle_sim.rl.rider_gpu_env import RiderGpuEnv
    from bicycle_sim.rl.whole_body import normalized_joint_targets

    source_meta = (
        json.loads((args.source_run / "run.json").read_text()) if args.source_run else None
    )
    cfg = source_meta["environment"] if source_meta else json.loads(args.config.read_text())
    algorithm, checkpoint = None, None
    if args.controller == "policy":
        if source_meta is None or source_meta.get("algorithm") not in ("ppo", "sac"):
            parser.error(
                "Policy recording requires --source-run pointing to a PPO/SAC training run"
            )
        algorithm = source_meta["algorithm"]
        checkpoint = args.checkpoint or args.source_run / "final.pt"
        if not checkpoint.is_file():
            parser.error(f"Missing checkpoint: {checkpoint}")
    elif args.checkpoint:
        parser.error("--checkpoint requires --controller policy")
    cfg.update(num_envs=args.num_envs, seed=args.seed)
    if cfg.get("whole_body_control"):
        from bicycle_sim.rl.whole_body import assistance_at

        cfg["crank_assistance_override"] = (
            args.crank_assistance if args.crank_assistance is not None else 0.0
        )
        assistance_at(0, cfg)
        if (
            source_meta
            and source_meta.get("whole_body_reference_sha256")
            != hashlib.sha256(Path(cfg["whole_body_reference_path"]).read_bytes()).hexdigest()
        ):
            parser.error("The trained whole-body initialization reference has changed")
    elif args.crank_assistance is not None:
        parser.error("--crank-assistance requires a whole-body configuration")
    if args.tire_collision is not None:
        cfg["tire_collision_model"] = args.tire_collision
    if args.epa_horizon is not None:
        cfg["epa_horizon_capacity"] = args.epa_horizon
    steer_limit = cfg.get("steering_reference_limit_rad", 0.25)
    if cfg.get("hand_control_enabled", False) and args.controller == "feedback":
        parser.error(
            "The legacy feedback IK table is not calibrated for the settled contact pose; use policy or zero"
        )
    if args.controller == "feedback":
        angles, offsets, errors = arm_references(steer_limit, cfg["arm_offset_limit_rad"])
    else:
        angles, offsets, errors = [0.0], np.zeros((1, 14)), []
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "environment": cfg,
                "controller": (
                    f"{algorithm}_mean_policy"
                    if args.controller == "policy"
                    else "roll_PD_to_arm_IK"
                    if args.controller == "feedback"
                    else "zero"
                ),
                "checkpoint": str(checkpoint) if checkpoint else None,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                if checkpoint
                else None,
                "backend": "mjlab/MuJoCo-Warp",
                "learning": False,
                "hand_control_enabled": cfg.get("hand_control_enabled", False),
                "whole_body_control": cfg.get("whole_body_control", False),
                "leg_control_mode": cfg.get("leg_control_mode", "phase_residual"),
                "observe_pedal_joints": cfg.get("observe_pedal_joints", False),
                "ik_errors_m": errors,
                "record_env": args.record_env,
                "record_all_envs": args.record_all_envs,
                "record_only": args.record_only,
                "first_episodes_only": args.first_episodes_only,
                "source_run": str(args.source_run) if args.source_run else None,
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path("src/bicycle_sim").rglob("*.py"))
                },
            },
            indent=2,
        )
        + "\n"
    )
    episodes = []
    trace = []
    frames = []
    cohort_frames = []
    recorded_episode = 0
    physics_error = None
    with shared_cuda_stream(args.device), torch.no_grad():
        env = RiderGpuEnv(cfg, device=args.device)
        policy = None
        if args.controller == "policy":
            from bicycle_sim.rl.skrl_agents import Policy, SquashedPolicy, deterministic_actions

            policy_type = SquashedPolicy if algorithm == "sac" else Policy
            policy = policy_type(
                env.num_observations,
                env.num_actions,
                args.device,
                cfg["hidden_dims"],
                raw_gaussian=cfg.get("ppo_raw_gaussian", False),
            ).to(args.device)
            policy.load_state_dict(
                torch.load(checkpoint, map_location=args.device, weights_only=False)["policy"]
            )
            policy.eval()
        first_finished = torch.zeros(args.num_envs, device=args.device, dtype=torch.bool)
        np.savez_compressed(
            args.output / "initial_states.npz",
            qpos=env.q.cpu().numpy(),
            qvel=env.v.cpu().numpy(),
        )
        import mujoco

        mujoco.mj_saveLastXML(str(args.output / "scene.xml"), env.model)
        snapshot = mujoco.MjModel.from_xml_path(str((args.output / "scene.xml").resolve()))
        if (snapshot.nq, snapshot.nv, snapshot.nu) != (env.model.nq, env.model.nv, env.model.nu):
            raise RuntimeError("Saved replay model topology differs from the simulated model")

        def capture(
            desired=float("nan"),
            terminated=False,
            truncated=False,
            raw_desired=float("nan"),
            arm_reference=None,
            normalized_action=None,
        ):
            j = args.record_env
            R, velocity, omega = env.state()
            torque = float(env.sim.data.actuator_force[j, env.crank_act])
            crank_speed = float(env.v[j, env.spins[0]])
            frames.append(
                {
                    "qpos": env.q[j].cpu().numpy().copy(),
                    "qvel": env.v[j].cpu().numpy().copy(),
                    "time_s": float(env.length[j]) * env.dt,
                    "episode_id": recorded_episode,
                    "roll_rad": float(torch.atan2(R[j, 2, 1], R[j, 2, 2])),
                    "roll_rate_rad_s": float(omega[j, 0]),
                    "speed_mps": float(velocity[j, 0]),
                    "desired_steer_rad": desired,
                    "diagnostic_pd_steer_rad": float(
                        (
                            -cfg.get("controller_roll_gain", 3.0)
                            * torch.atan2(R[j, 2, 1], R[j, 2, 2])
                            - cfg.get("controller_roll_rate_gain", 0.7) * omega[j, 0]
                        ).clamp(-steer_limit, steer_limit)
                    ),
                    "action": (
                        torch.zeros(env.num_actions, device=args.device)
                        if normalized_action is None
                        else normalized_action
                    )
                    .cpu()
                    .numpy()
                    .copy(),
                    "raw_steer_demand_rad": raw_desired,
                    "steer_reference_saturated": bool(abs(raw_desired) >= steer_limit),
                    "actual_steer_rad": float(env.q[j, env.steer_q]),
                    (
                        "arm_requested_target_rad"
                        if args.controller == "policy"
                        else "arm_ik_target_rad"
                    ): (
                        env.arm_nominal
                        + (
                            torch.zeros(14, device=args.device)
                            if arm_reference is None
                            else arm_reference
                        )
                    )
                    .cpu()
                    .numpy()
                    .copy(),
                    "arm_executed_target_rad": env.sim.data.ctrl[j, env.maps["arm_act"]]
                    .cpu()
                    .numpy()
                    .copy(),
                    "arm_actual_rad": env.q[j, env.maps["arm_q"]].cpu().numpy().copy(),
                    "arm_actual_torque_Nm": env.sim.data.qfrc_actuator[j, env.maps["arm_v"]]
                    .cpu()
                    .numpy()
                    .copy(),
                    "crank_speed_rad_s": crank_speed,
                    "rear_speed_rad_s": float(env.v[j, env.spins[1]]),
                    "front_speed_rad_s": float(env.v[j, env.spins[2]]),
                    "motor_torque_Nm": torque,
                    "motor_power_W": torque * crank_speed,
                    "freewheel_rear_torque_Nm": -float(
                        env.sim.data.actuator_force[j, env.model.actuator("freewheel").id]
                    ),
                    "terminated": terminated,
                    "truncated": truncated,
                    **(
                        {
                            "hand_closure": env.closure[j].cpu().numpy().copy(),
                            "finger_normal_force_N": env.finger_forces[j].cpu().numpy().copy(),
                        }
                        if env.natural_grasp
                        else {}
                    ),
                    **(
                        {
                            "support_normal_force_N": env.support_forces[j].cpu().numpy().copy(),
                            ("leg_command_normalized" if env.direct_legs else "leg_offset_rad"): (
                                normalized_joint_targets(env.leg_target[j], env.leg_limits)
                                if env.direct_legs
                                else env.leg_offset[j]
                            )
                            .cpu()
                            .numpy()
                            .copy(),
                            "pedal_joint_angle_rad": env.q[j, env.maps["pedal_q"]]
                            .cpu()
                            .numpy()
                            .copy(),
                            "pedal_joint_velocity_rad_s": env.v[j, env.maps["pedal_v"]]
                            .cpu()
                            .numpy()
                            .copy(),
                            "waist_offset_rad": env.waist_offset[j].cpu().numpy().copy(),
                            "crank_assistance": float(env.assistance[j]),
                            "foot_to_crank_torque_Nm": float(env.pedal_torque[j]),
                            "leg_executed_target_rad": env.sim.data.ctrl[j, env.maps["leg_act"]]
                            .cpu()
                            .numpy()
                            .copy(),
                            "waist_executed_target_rad": env.sim.data.ctrl[j, env.maps["waist_act"]]
                            .cpu()
                            .numpy()
                            .copy(),
                            "leg_actual_rad": env.q[j, env.maps["leg_q"]].cpu().numpy().copy(),
                            "waist_actual_rad": env.q[j, env.maps["waist_q"]].cpu().numpy().copy(),
                            "leg_actual_torque_Nm": env.sim.data.qfrc_actuator[j, env.maps["leg_v"]]
                            .cpu()
                            .numpy()
                            .copy(),
                        }
                        if env.whole_body
                        else {}
                    ),
                }
            )

        def capture_cohort(active):
            R, velocity, omega = env.state()
            values = {
                "qpos": env.q,
                "qvel": env.v,
                "time_s": env.length * env.dt,
                "roll_rad": torch.atan2(R[:, 2, 1], R[:, 2, 2]),
                "roll_rate_rad_s": omega[:, 0],
                "speed_mps": velocity[:, 0],
                "actual_steer_rad": env.q[:, env.steer_q],
                "desired_steer_rad": torch.full_like(env.length, float("nan"), dtype=torch.float32),
                "valid_first_episode": active,
                "crank_speed_rad_s": env.v[:, env.spins[0]],
                "motor_power_W": env.sim.data.actuator_force[:, env.crank_act]
                * env.v[:, env.spins[0]],
            }
            if env.whole_body:
                values.update(
                    support_normal_force_N=env.support_forces,
                    foot_to_crank_torque_Nm=env.pedal_torque,
                    crank_assistance=env.assistance,
                )
            cohort_frames.append({k: v.detach().cpu().numpy().copy() for k, v in values.items()})

        capture()
        if args.record_all_envs:
            capture_cohort(~first_finished)
        grid = torch.tensor(angles, dtype=torch.float32, device=args.device)
        table = torch.tensor(offsets, dtype=torch.float32, device=args.device)
        for i in range(args.steps):
            R, _, omega = env.state()
            raw_desired = (
                -cfg.get("controller_roll_gain", 3.0) * torch.atan2(R[:, 2, 1], R[:, 2, 2])
                - cfg.get("controller_roll_rate_gain", 0.7) * omega[:, 0]
            )
            if args.controller == "feedback":
                desired = raw_desired.clamp(-steer_limit, steer_limit)
                right = torch.searchsorted(grid, desired).clamp(1, len(angles) - 1)
                weight = ((desired - grid[right - 1]) / (grid[right] - grid[right - 1]))[:, None]
                reference = table[right - 1] * (1 - weight) + table[right] * weight
            elif args.controller == "policy":
                raw_desired = torch.full_like(raw_desired, float("nan"))
                desired = raw_desired
                reference = deterministic_actions(policy, env.observe(), algorithm)
                policy_action = reference
                reference = reference[:, :14] * cfg["arm_offset_limit_rad"]
            else:
                raw_desired = torch.zeros_like(raw_desired)
                desired = raw_desired
                reference = torch.zeros((args.num_envs, 14), device=args.device)
            action = reference / cfg["arm_offset_limit_rad"]
            if args.controller == "policy":
                action = policy_action
            elif env.natural_grasp:
                action = torch.cat(
                    (action, torch.zeros((env.num_envs, env.num_actions - 14), device=args.device)),
                    -1,
                )
            from bicycle_sim.rl.physics_diagnostics import RiderPhysicsError

            try:
                _, _, terminated, truncated, info = env.step(action)
            except RiderPhysicsError as error:
                from bicycle_sim.cli.train_rider import save_physics_failure

                save_physics_failure(
                    args.output,
                    None,
                    error,
                    vector_steps=i,
                    num_envs=env.num_envs,
                    evaluation=True,
                )
                physics_error = error
                break  # Do not append invalid poses or count the failed step as an episode.
            done = (terminated | truncated).flatten()
            if args.record_all_envs:
                capture_cohort(~first_finished)
            if not args.first_episodes_only or not bool(first_finished[args.record_env]):
                capture(
                    float(desired[args.record_env]),
                    bool(terminated[args.record_env, 0]),
                    bool(truncated[args.record_env, 0]),
                    float(raw_desired[args.record_env]),
                    reference[args.record_env],
                    action[args.record_env],
                )
            if bool(done[args.record_env]):
                recorded_episode += 1
            ids = done.nonzero().flatten()
            if i % 50 == 0:
                active = ~first_finished if args.first_episodes_only else torch.ones_like(done)
                trace.append(
                    {
                        "step": i + 1,
                        "time_s": (i + 1) * env.dt,
                        "active_initial_worlds": int(active.sum()),
                        "mean_tilt_rad": float(info["tilt"][active].mean()),
                        "max_tilt_rad": float(info["tilt"][active].max()),
                        "mean_speed_mps": float(info["forward_speed"][active].mean()),
                    }
                )
            for j in ids.tolist():
                if args.first_episodes_only and bool(first_finished[j]):
                    continue
                episodes.append(
                    {
                        "step": i + 1,
                        "env": j,
                        "duration_s": float(info["episode_seconds"][j]),
                        "survived_horizon": bool(truncated[j, 0]),
                        "tilt_failure": bool(info["tilt_failure"][j]),
                        "ground_failure": bool(info["ground_failure"][j]),
                        "tilt_rad": float(info["tilt"][j]),
                        "steer_rad": float(info["steer_angle"][j]),
                        "final_forward_speed_mps": float(info["forward_speed"][j]),
                        **(
                            {
                                key: bool(info[key][j])
                                if key == "speed_qualified_success"
                                else float(info[key][j])
                                for key in (
                                    "speed_qualified_success",
                                    "episode_mean_forward_speed_mps",
                                    "episode_speed_in_band_fraction",
                                    "episode_tail_speed_in_band_fraction",
                                )
                            }
                            if env.speed_objective is not None
                            else {}
                        ),
                        **(
                            {
                                "rider_failure": bool(info["rider_failure"][j]),
                                "crank_assistance": float(info["crank_assistance"][j]),
                            }
                            if env.whole_body
                            else {}
                        ),
                    }
                )
            first_finished |= done
            if args.first_episodes_only and bool(first_finished.all()):
                break
            if ids.numel():
                env.reset(ids)
            if (i + 1) % 100 == 0:
                print(
                    f"step={i + 1}/{args.steps} completed={len(episodes)} survivors={sum(r['survived_horizon'] for r in episodes)}",
                    flush=True,
                )
    report = {
        **summarize_episodes(episodes),
        "first_cohort_complete": bool(first_finished.all()),
        "uncompleted_first_episodes": int((~first_finished).sum()),
        "episodes": episodes,
        "trace": trace,
        "physics_failure": physics_error.report if physics_error else None,
    }
    save_probe_recording(args.output, report, frames, cohort_frames)
    print(json.dumps({k: v for k, v in report.items() if k not in ("episodes", "trace")}, indent=2))
    if physics_error is not None:
        raise physics_error
    if args.first_episodes_only and not report["first_cohort_complete"]:
        raise SystemExit(
            "Incomplete initial cohort: increase --steps; do not compare partial cohorts"
        )
    success_rate = report.get("speed_qualified_20s_rate", report["survival_rate"])
    if not args.record_only and (not episodes or success_rate < 0.9):
        raise SystemExit(
            "Controller probe below 90% task success; inspect survival/speed results before training"
        )


if __name__ == "__main__":
    main()
