"""Run paired GPU zero-action and feedback/IK baselines using saved B settings."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def compare_results(root):
    configs = [json.loads((root / mode / "run.json").read_text()) for mode in ("zero", "feedback")]
    if configs[0]["environment"] != configs[1]["environment"]:
        raise ValueError("Baseline environment configurations differ")
    with (
        np.load(root / "zero/initial_states.npz") as zero,
        np.load(root / "feedback/initial_states.npz") as feedback,
    ):
        differences = {}
        for key in ("qpos", "qvel"):
            np.testing.assert_allclose(zero[key], feedback[key], rtol=0, atol=1e-7)
            differences[key] = float(np.max(abs(zero[key] - feedback[key])))
    reports = {}
    for mode in ("zero", "feedback"):
        r = json.loads((root / mode / "results.json").read_text())
        if not r["first_cohort_complete"]:
            raise ValueError("An initial cohort is incomplete")
        reports[mode] = {k: v for k, v in r.items() if k not in ("episodes", "trace")}
    zero_rows = json.loads((root / "zero/results.json").read_text())["episodes"]
    feedback_rows = json.loads((root / "feedback/results.json").read_text())["episodes"]
    by_world = [{r["env"]: r["duration_s"] for r in rows} for rows in (zero_rows, feedback_rows)]
    count = configs[0]["environment"]["num_envs"]
    if any(len(rows) != count for rows in (zero_rows, feedback_rows)) or any(
        set(rows) != set(range(count)) for rows in by_world
    ):
        raise ValueError("Expected exactly one completed episode per initial world")
    delta = np.array([by_world[1][i] - by_world[0][i] for i in sorted(by_world[0])])
    return {
        "environment": configs[0]["environment"],
        "initial_state_max_difference": differences,
        "statistics": "One first episode per world; paired initial states; no episode resampling",
        "baselines": reports,
        "paired_feedback_minus_zero_seconds": {
            "mean": float(delta.mean()),
            "median": float(np.median(delta)),
            "positive_fraction": float((delta > 0).mean()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run", type=Path, default=Path("output/phase2/b_sac_v4_eval_seed20000")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1100)
    parser.add_argument("--tire-collision", choices=("mesh", "ellipsoid"))
    parser.add_argument("--output", type=Path, default=Path("output/phase2/b_controls_v4"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory")
    # Let the probe validate CUDA before creating output files.
    for mode in ("zero", "feedback"):
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "bicycle_sim.cli.probe_rider_gpu",
                "--controller",
                mode,
                "--source-run",
                str(args.source_run),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--num-envs",
                str(args.num_envs),
                "--steps",
                str(args.steps),
                "--first-episodes-only",
                "--record-only",
                *(["--tire-collision", args.tire_collision] if args.tire_collision else []),
                "--output",
                str(args.output / mode),
            ],
            check=True,
        )
    report = compare_results(args.output)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
