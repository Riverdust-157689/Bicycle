"""Record the installed environment and compile project assets without training."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import mujoco
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/phase1"))
    parser.add_argument("--render", action="store_true", help="Requires EGL/GL configuration")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = {}
    for name, path in {
        "robot": Path("assets/g1_dex3/g1_with_hands.xml"),
        "bicycle_free": Path("assets/bicycle/scene_free.xml"),
        "bicycle_fixed_diagnostic": Path("assets/bicycle/scene_fixed_diagnostic.xml"),
    }.items():
        m = mujoco.MjModel.from_xml_path(str(path.resolve()))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        models[name] = {"nq": m.nq, "nv": m.nv, "nu": m.nu, "mass_kg": float(m.body_mass.sum())}
        if args.render and name == "bicycle_free":
            import imageio.v3 as iio

            with mujoco.Renderer(m, height=480, width=640) as renderer:
                renderer.update_scene(d, camera="side")
                iio.imwrite(args.output / "bicycle_v0.png", renderer.render())
    try:
        smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False
        )
        gpu_status = {"returncode": smi.returncode, "output": smi.stdout + smi.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu_status = {"error": str(error)}
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "isolated_python": bool(sys.flags.isolated),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("mjlab", "mujoco", "mujoco-warp", "torch", "warp-lang")
        },
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi": gpu_status,
        "models": models,
        "scope": "CPU model compilation only; no rider fit or GPU dynamics acceptance",
    }
    path = args.output / "platform_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
