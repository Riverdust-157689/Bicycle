"""Evaluate a frozen policy against a speed-matched no-steering baseline."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from bicycle_sim.cli.train_bicycle import runner_config
from bicycle_sim.rl.bicycle_env import BicycleEnv
from bicycle_sim.rl.vector_env import BicycleVecEnv


def load_run(run, checkpoint="final.pt"):
    meta = json.loads((run / "run.json").read_text())
    cfg, bike = meta["environment"], meta["bike"]
    cfg["num_envs"] = 1
    torch.set_num_threads(1)
    runner = OnPolicyRunner(
        BicycleVecEnv(bike, cfg), meta.get("runner", runner_config(cfg)), device="cpu"
    )
    runner.load(str(run / checkpoint), map_location="cpu")
    return bike, cfg, runner.get_inference_policy()


def evaluate(bike, cfg, policy, episodes, seed):
    results = []
    for i in range(episodes):
        env = BicycleEnv(bike, cfg, seed=seed + i)
        errors, headings, rolls = [], [], []
        while True:
            if policy is None:
                # Positive crank reference, no steering. Same commanded speed.
                G = bike["transmission"]["rear_wheel_revolutions_per_crank_revolution"]
                cadence = env.command[0] / (G * bike["geometry"]["wheel_radius"])
                action = [2 * cadence / cfg["max_crank_speed_rad_s"] - 1, 0.0]
            else:
                from tensordict import TensorDict

                obs = torch.from_numpy(env.observe()).unsqueeze(0)
                with torch.inference_mode():
                    action = policy(TensorDict({"policy": obs}, batch_size=[1]))[0].numpy()
            _, _, terminated, truncated, info = env.step(action)
            errors.append(info["speed_error_mps"])
            headings.append(abs(info["heading_error_rad"]))
            rolls.append(abs(info["roll_rad"]))
            if terminated or truncated:
                break
        # No survivor-only averaging: failed episodes count as unsuccessful.
        results.append(
            {
                "seed": seed + i,
                "survived_horizon": bool(truncated),
                "duration_s": env.steps * env.dt,
                "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(errors)))),
                "heading_mae_rad": float(np.mean(headings)),
                "max_roll_rad": float(max(rolls)),
                "goal_success": bool(
                    truncated
                    and np.sqrt(np.mean(np.square(errors))) < 0.5
                    and np.mean(headings) < 0.2
                ),
            }
        )
    return {
        "episodes": results,
        "survival_rate": float(np.mean([r["survived_horizon"] for r in results])),
        "goal_success_rate": float(np.mean([r["goal_success"] for r in results])),
        "mean_duration_s": float(np.mean([r["duration_s"] for r in results])),
        "mean_velocity_rmse_mps": float(np.mean([r["velocity_rmse_mps"] for r in results])),
        "mean_heading_mae_rad": float(np.mean([r["heading_mae_rad"] for r in results])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    bike, cfg, policy = load_run(args.run, args.checkpoint)
    report = {
        "checkpoint": str(args.run / args.checkpoint),
        "disturbances": False,
        "policy": evaluate(bike, cfg, policy, args.episodes, args.seed),
        "no_steering": evaluate(bike, cfg, None, args.episodes, args.seed),
    }
    output = args.output or args.run / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {a: b for a, b in v.items() if a != "episodes"}
                for k, v in report.items()
                if isinstance(v, dict)
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
