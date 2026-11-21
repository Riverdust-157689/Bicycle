"""Train assisted experiment B with genuine GPU physics and skrl PPO or SAC."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from collections import deque
from pathlib import Path


def save_physics_failure(
    output,
    agent,
    error,
    *,
    vector_steps,
    num_envs,
    evaluation=False,
    policy_updated=False,
    smoke=False,
):
    """Save valid weights without storing the failed transition in SAC replay."""
    import numpy as np

    output = Path(output)
    report = dict(
        error.report,
        vector_steps_completed=vector_steps,
        transitions_completed=vector_steps * num_envs,
        failed_vector_step=vector_steps + 1,
    )
    if not evaluation:
        checkpoint = output / "interrupted.pt"
        try:
            agent.save(str(checkpoint))
            report["saved_checkpoint"] = str(checkpoint)
        except (OSError, RuntimeError) as exc:
            report["checkpoint_save_error"] = repr(exc)
    np.savez_compressed(
        output / "physics_failure_states.npz",
        **{k: v.detach().cpu().numpy() for k, v in error.states.items()},
    )
    (output / "physics_failure.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "completion.json").write_text(
        json.dumps(
            {
                "finished": False,
                "policy_updated": policy_updated,
                "smoke_only": smoke,
                "physics_failure": True,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--config", type=Path, default=Path("configs/rl/bicycle_b.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/bicycle_b_ppo"))
    parser.add_argument(
        "--reference", type=Path, default=Path("output/phase1/rider_kinematic_reference.npz")
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--epa-horizon",
        type=int,
        choices=range(24, 257),
        metavar="24..256",
        help="EPA boundary workspace capacity; does not change contact physics",
    )
    parser.add_argument(
        "--crank-assistance",
        type=float,
        help="Whole-body fixed assistance [0,1]; evaluation defaults to zero",
    )
    parser.add_argument("--transitions", type=int, default=8388608)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="GPU reset/physics and short algorithm-update check; not a trained result",
    )
    parser.add_argument(
        "--evaluate", type=Path, help="Load an agent checkpoint and run without updates"
    )
    args = parser.parse_args()
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error(
            "This command requires visible CUDA for both physics and learning; it never falls back to CPU"
        )
    try:
        version = importlib.metadata.version("skrl")
    except importlib.metadata.PackageNotFoundError:
        parser.error(
            "Install the pinned training extra first: uv add --optional training 'skrl==1.4.2'"
        )
    if version != "1.4.2":
        parser.error(f"Expected skrl 1.4.2, found {version}; use the documented uv training extra")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; use a new directory")
    cfg = json.loads(args.config.read_text())
    if args.evaluate:
        run_file = next(
            (p / "run.json" for p in args.evaluate.parents if (p / "run.json").is_file()), None
        )
        if run_file is None:
            parser.error("Checkpoint has no parent run.json")
        trained = json.loads(run_file.read_text())
        if trained["algorithm"] != args.algorithm:
            parser.error("Evaluation algorithm must match the checkpoint run")
        cfg = trained["environment"]
        if trained.get("grasp_reference_sha256") and (
            hashlib.sha256(Path(cfg["grasp_reference_path"]).read_bytes()).hexdigest()
            != trained["grasp_reference_sha256"]
        ):
            parser.error("Saved grasp initialization reference has changed")
        if (
            trained.get("whole_body_reference_sha256")
            and hashlib.sha256(Path(cfg["whole_body_reference_path"]).read_bytes()).hexdigest()
            != trained["whole_body_reference_sha256"]
        ):
            parser.error("Saved no-weld initialization reference has changed")
    if cfg.get("whole_body_control"):
        if args.crank_assistance is not None:
            cfg["crank_assistance_override"] = args.crank_assistance
        elif args.evaluate:
            cfg["crank_assistance_override"] = 0.0
        from bicycle_sim.rl.whole_body import assistance_at

        assistance_at(0, cfg)
    elif args.crank_assistance is not None:
        parser.error("--crank-assistance is only supported by the whole-body task")
    for name in ("num_envs", "seed"):
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    if args.epa_horizon is not None:
        cfg["epa_horizon_capacity"] = args.epa_horizon
    if args.smoke:
        cfg["num_envs"] = args.num_envs or 4
        cfg["ppo_rollouts"] = 8
    if cfg["num_envs"] < 1 or args.transitions < 1:
        parser.error("num-envs and transitions must be positive")
    from bicycle_sim.rl.cuda_context import shared_cuda_stream

    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    with shared_cuda_stream(args.device):
        run_training(args, cfg, version)


def run_training(args, cfg, version):
    """All Torch allocations, physics calls and optimizer updates share a stream."""
    import torch

    from bicycle_sim.rl.physics_diagnostics import RiderPhysicsError
    from bicycle_sim.rl.rider_gpu_env import RiderGpuEnv
    from bicycle_sim.rl.skrl_agents import make_agent

    env = RiderGpuEnv(cfg, device=args.device, reference_path=args.reference)
    if args.smoke and env.num_envs > 1:
        other_q, other_v = env.q[1:].clone(), env.v[1:].clone()
        env.reset(torch.tensor([0], device=args.device))
        torch.testing.assert_close(env.q[1:], other_q, atol=0, rtol=0)
        torch.testing.assert_close(env.v[1:], other_v, atol=0, rtol=0)
    if args.smoke and env.natural_grasp:
        both_loaded = (env.finger_forces[:, :3].sum(-1) > 0.5) & (
            env.finger_forces[:, 3:].sum(-1) > 0.5
        )
        if not bool(both_loaded.all()):
            raise RuntimeError("Natural-grasp smoke: one or both hands lack initial finger contact")
    steps = 32 if args.smoke else math.ceil(args.transitions / env.num_envs)
    if args.algorithm == "ppo" and not args.evaluate:
        steps = math.ceil(steps / cfg["ppo_rollouts"]) * cfg["ppo_rollouts"]
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "environment": cfg,
        "algorithm": args.algorithm,
        "backend": "mjlab.Simulation/MuJoCo-Warp",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device),
        "skrl": version,
        "num_actions": env.num_actions,
        "num_observations": env.num_observations,
        "success_metric": "speed_qualified_20s_rate"
        if env.speed_objective
        else "survival_20s_rate",
        "assistance": [
            *([] if env.whole_body else ["local seat weld"]),
            *([] if env.natural_grasp else ["local ideal palm connections"]),
            "zero external crank torque"
            if env.whole_body and cfg.get("crank_assistance_override") == 0
            else "scheduled external crank torque"
            if env.whole_body
            else "external crank drive",
            "direct full-range leg position targets"
            if env.direct_legs
            else "policy residuals around phase leg reference"
            if env.whole_body
            else "phase-driven leg PD",
        ],
        "steering_actuator": False,
        "timesteps": steps,
        "transitions": steps * env.num_envs,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("src/bicycle_sim").rglob("*.py"))
        },
        "evaluation_checkpoint": str(args.evaluate) if args.evaluate else None,
    }
    if env.natural_grasp:
        snapshot["hand_control"] = (
            "six closure residuals around a settled contact grasp; no palm connections"
        )
        snapshot["grasp_reference_sha256"] = hashlib.sha256(
            Path(cfg["grasp_reference_path"]).read_bytes()
        ).hexdigest()
    if env.whole_body:
        snapshot.update(
            whole_body_reference_sha256=hashlib.sha256(
                Path(cfg["whole_body_reference_path"]).read_bytes()
            ).hexdigest(),
            action_layout={
                "arms": [0, 14],
                "fingers": [14, 20],
                "legs": [20, 32],
                "waist": [32, 35],
            },
            equality_constraints=env.m.neq,
            leg_control_mode=env.leg_control_mode,
            observe_pedal_joints=env.observe_pedal_joints,
            leg_action_mapping="[-1,1] -> [joint/control lower, upper]; zero is midpoint"
            if env.direct_legs
            else "phase leg reference + bounded joint offsets",
            curriculum="per-episode observed crank assistance; all 35 actions always active",
        )
    (args.output / "run.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    import mujoco

    mujoco.mj_saveLastXML(str(args.output / "scene.xml"), env.model)
    agent = make_agent(
        args.algorithm,
        env,
        cfg,
        args.output,
        smoke=args.smoke,
        evaluation=bool(args.evaluate),
    )
    agent.init(trainer_cfg={"timesteps": steps})
    if args.evaluate:
        agent.load(str(args.evaluate))
    agent.set_running_mode("eval" if args.evaluate else "train")
    obs = env.observe()
    before = [p.detach().clone() for p in agent.models["policy"].parameters()]
    completed, successes, seconds, returns = 0, 0, 0.0, 0.0
    recent = deque(maxlen=1000)
    speed_successes = 0
    recent_speed_successes = deque(maxlen=1000)
    failure_counts = {"tilt": 0, "ground": 0, "rider": 0}
    diagnostic_keys = [
        "raw_action_out_of_bounds_fraction",
        "executed_action_abs_mean",
        "steer_abs_mean_rad",
        "tilt_mean_rad",
    ]
    if env.natural_grasp:
        diagnostic_keys.extend(
            (
                "finger_contact_fraction",
                "both_hands_contact_fraction",
                "finger_normal_force_mean_N",
                "hand_closure_mean",
            )
        )
    if env.whole_body:
        diagnostic_keys.extend(
            (
                "seat_normal_force_mean_N",
                "both_feet_contact_fraction",
                "crank_assistance_mean",
                "crank_speed_mean_rad_s",
                "external_crank_power_mean_W",
                "leg_positive_power_mean_W",
                "foot_to_crank_torque_mean_Nm",
                "foot_to_crank_power_mean_W",
                "forward_speed_mean_mps",
            )
        )
    if env.speed_objective is not None:
        diagnostic_keys.extend(("speed_target_score_mean", "low_speed_penalty_rate_mean"))
    diagnostic_sum = torch.zeros(len(diagnostic_keys), device=args.device)
    diagnostic_steps = 0
    start = time.monotonic()
    log = args.output / "metrics.jsonl"
    for i in range(steps):
        if not args.evaluate:
            agent.pre_interaction(timestep=i, timesteps=steps)
        with torch.no_grad():
            action, _, outputs = agent.act(obs, timestep=i, timesteps=steps)
            if args.evaluate:
                action = outputs.get("mean_actions", action)
            try:
                next_obs, reward, terminated, truncated, info = env.step(action)
            except RiderPhysicsError as error:
                changed = any(
                    not torch.equal(a, b)
                    for a, b in zip(before, agent.models["policy"].parameters(), strict=True)
                )
                save_physics_failure(
                    args.output,
                    agent,
                    error,
                    vector_steps=i,
                    num_envs=env.num_envs,
                    evaluation=bool(args.evaluate),
                    policy_updated=changed,
                    smoke=args.smoke,
                )
                raise
            if (
                args.smoke
                and env.whole_body
                and cfg.get("crank_assistance_override") == 0
                and env.sim.data.actuator_force[:, env.crank_act].abs().max() > 1e-6
            ):
                raise RuntimeError("Zero-assistance smoke: crank motor is still exerting torque")
            diagnostics = [
                (action.abs() > 1).float().mean(),
                action.clamp(-1, 1).abs().mean(),
                info["steer_angle"].abs().mean(),
                info["tilt"].mean(),
            ]
            if env.natural_grasp:
                diagnostics.extend(
                    (
                        info["finger_contact_fraction"].mean(),
                        info["both_hands_contact"].float().mean(),
                        info["finger_normal_force_N"].mean(),
                        info["hand_closure"].mean(),
                    )
                )
            if env.whole_body:
                diagnostics.extend(
                    (
                        info["seat_normal_force_N"].mean(),
                        (info["foot_normal_force_N"] > 1).all(-1).float().mean(),
                        info["crank_assistance"].mean(),
                        info["crank_speed_rad_s"].mean(),
                        info["external_crank_power_W"].mean(),
                        info["leg_positive_power_W"].mean(),
                        info["foot_to_crank_torque_Nm"].mean(),
                        info["foot_to_crank_power_W"].mean(),
                        info["forward_speed"].mean(),
                    )
                )
            if env.speed_objective is not None:
                diagnostics.extend(
                    (info["speed_target_score"].mean(), info["low_speed_penalty_rate"].mean())
                )
            diagnostic_sum += torch.stack(diagnostics)
            diagnostic_steps += 1
            done = (terminated | truncated).squeeze(-1)
            if not args.evaluate:
                agent.record_transition(
                    states=obs,
                    actions=action,
                    rewards=reward,
                    next_states=next_obs,
                    terminated=terminated,
                    truncated=truncated,
                    infos={},
                    timestep=i,
                    timesteps=steps,
                )
        if not args.evaluate:
            agent.post_interaction(timestep=i, timesteps=steps)
        ids = done.nonzero().flatten()
        completed += ids.numel()
        successes += int(truncated.sum().item())
        if env.speed_objective is not None:
            speed_successes += int(info["speed_qualified_success"][ids].sum().item())
            recent_speed_successes.extend(info["speed_qualified_success"][ids].tolist())
        seconds += float(info["episode_seconds"][ids].sum().item())
        returns += float(info["episode_return"][ids].sum().item())
        if ids.numel():
            recent.extend(
                zip(
                    info["episode_seconds"][ids].tolist(),
                    truncated[ids, 0].tolist(),
                    strict=True,
                )
            )
            failure_counts["tilt"] += int(info["tilt_failure"][ids].sum().item())
            failure_counts["ground"] += int(info["ground_failure"][ids].sum().item())
            if env.whole_body:
                failure_counts["rider"] += int(info["rider_failure"][ids].sum().item())
        # Reset only AFTER replay/rollout storage saw the terminal next state.
        obs = env.reset(ids) if ids.numel() else next_obs
        if (i + 1) % 100 == 0 or i + 1 == steps:
            row = {
                "vector_steps": i + 1,
                "transitions": (i + 1) * env.num_envs,
                "completed_episodes": completed,
                "survival_20s_rate": successes / max(1, completed),
                "mean_episode_seconds": seconds / max(1, completed),
                "mean_return": returns / max(1, completed),
                "unfinished_episodes": env.num_envs,
                "wall_seconds": time.monotonic() - start,
                "recent_completed_episodes": len(recent),
                "recent_mean_episode_seconds": sum(x[0] for x in recent) / max(1, len(recent)),
                "recent_survival_20s_rate": sum(x[1] for x in recent) / max(1, len(recent)),
                "tilt_failures": failure_counts["tilt"],
                "nonwheel_ground_failures": failure_counts["ground"],
                **({"rider_failures": failure_counts["rider"]} if env.whole_body else {}),
            }
            if env.speed_objective is not None:
                row.update(
                    speed_qualified_20s_rate=speed_successes / max(1, completed),
                    recent_speed_qualified_20s_rate=sum(recent_speed_successes)
                    / max(1, len(recent_speed_successes)),
                )
            for key, value in zip(
                diagnostic_keys,
                (diagnostic_sum / diagnostic_steps).tolist(),
                strict=True,
            ):
                row[key] = value
            diagnostic_sum.zero_()
            diagnostic_steps = 0
            with log.open("a") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    changed = any(
        not torch.equal(a, b)
        for a, b in zip(before, agent.models["policy"].parameters(), strict=True)
    )
    if args.smoke and not args.evaluate and not changed:
        raise RuntimeError("Smoke check failed: policy weights did not update")
    if not args.evaluate:
        agent.save(str(args.output / "final.pt"))
    (args.output / "completion.json").write_text(
        json.dumps(
            {"finished": True, "policy_updated": changed, "smoke_only": args.smoke}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
