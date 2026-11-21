"""Train experiment A with the existing RSL-RL PPO implementation on CPU."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import torch
from rsl_rl.runners import OnPolicyRunner

from bicycle_sim.rl.vector_env import BicycleVecEnv


def runner_config(cfg):
    # Match the installed RSL-RL MLP API directly. mjlab's richer model config
    # contains CNN/RNN fields that this MLPModel does not accept.
    model = {
        "class_name": "MLPModel",
        "hidden_dims": cfg["hidden_dims"],
        "activation": "elu",
        "obs_normalization": True,
    }
    actor = copy.deepcopy(model)
    actor["distribution_cfg"] = {
        "class_name": "GaussianDistribution",
        "init_std": 0.35,
        "std_type": "log",
    }
    return {
        "seed": cfg["seed"],
        "num_steps_per_env": cfg["num_steps_per_env"],
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": actor,
        "critic": model,
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": cfg["learning_rate"],
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "entropy_coef": 0.002,
            "gamma": 0.99,
            "lam": 0.95,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "save_interval": 50,
        "logger": "tensorboard",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rl/bicycle_a.json"))
    parser.add_argument("--bike-config", type=Path, default=Path("configs/bicycle_v0.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/bicycle_a"))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument(
        "--warm-start", type=Path, help="Load actor/critic weights only for a new task curriculum"
    )
    args = parser.parse_args()
    cfg, bike = json.loads(args.config.read_text()), json.loads(args.bike_config.read_text())
    for key in ("iterations", "num_envs", "seed"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if cfg["iterations"] < 1:
        parser.error("iterations must be positive")
    if args.resume:
        previous = json.loads((args.resume.parent / "run.json").read_text())
        ignored = {"iterations", "num_envs"}
        old = {k: v for k, v in previous["environment"].items() if k not in ignored}
        new = {k: v for k, v in cfg.items() if k not in ignored}
        if old != new or previous["bike"] != bike:
            parser.error("Resume requires the same task, seed and vehicle configuration")
    if (args.output / "run.json").exists():
        parser.error("Output already contains a run; choose another directory (also for resume)")
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    env = BicycleVecEnv(bike, cfg)
    r_cfg = runner_config(cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "environment": cfg,
                "bike": bike,
                "runner": copy.deepcopy(r_cfg),
                "backend": "native_mujoco_cpu",
                "mujoco_version": mujoco.__version__,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path("src/bicycle_sim").rglob("*.py"))
                },
                "resumed_from": str(args.resume) if args.resume else None,
                "warm_started_from": str(args.warm_start) if args.warm_start else None,
            },
            indent=2,
        )
        + "\n"
    )
    runner = OnPolicyRunner(env, r_cfg, log_dir=str(args.output), device="cpu")
    if args.resume:
        runner.load(str(args.resume), map_location="cpu")
    elif args.warm_start:
        runner.load(
            str(args.warm_start),
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            map_location="cpu",
        )
    runner.learn(cfg["iterations"])
    runner.save(str(args.output / "final.pt"))


if __name__ == "__main__":
    main()
