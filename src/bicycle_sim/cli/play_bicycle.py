"""Replay one deterministic policy episode as a GIF; use MUJOCO_GL=egl headlessly."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from tensordict import TensorDict

from bicycle_sim.cli.evaluate_bicycle import load_run
from bicycle_sim.rl.bicycle_env import BicycleEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("output/phase2/bicycle_a.gif"))
    args = parser.parse_args()
    bike, cfg, policy = load_run(args.run, args.checkpoint)
    env = BicycleEnv(bike, cfg, seed=args.seed)
    # Rendering parameters only; a MuJoCo plane is physically infinite already.
    env.model.geom_size[env.floor, :2] = 100
    env.model.vis.headlight.ambient[:] = 0.6
    env.model.vis.headlight.diffuse[:] = 0.7
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.7, 125, -18
    frames = []
    with mujoco.Renderer(env.model, height=360, width=640) as renderer:
        for i in range(env.max_steps):
            if i % 5 == 0:
                camera.lookat[:] = env.data.xpos[env.frame] + np.array([0.4, 0, 0.5])
                renderer.update_scene(env.data, camera=camera)
                # Target arrow is a render-only user geom, never a simulated force.
                scene = renderer.scene
                geom = scene.geoms[scene.ngeom]
                scene.ngeom += 1
                start = env.data.xpos[env.frame] + np.array([0.4, 0, 1.2])
                end = start + 0.5 * np.array([np.cos(env.command[1]), np.sin(env.command[1]), 0])
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).ravel(),
                    np.array([0.9, 0.25, 0.1, 1]),
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.025, start, end)
                frame = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(frame)
                speed = np.linalg.norm(env.state()["velocity_world"][:2])
                draw.rectangle((0, 0, 640, 26), fill="white")
                draw.text(
                    (8, 7),
                    f"t={env.steps * env.dt:.1f}s  speed={speed:.2f}  target={env.command[0]:.2f} m/s  arrow=target heading",
                    fill="black",
                )
                frames.append(frame)
            obs = torch.from_numpy(env.observe()).unsqueeze(0)
            with torch.inference_mode():
                action = policy(TensorDict({"policy": obs}, batch_size=[1]))[0].numpy()
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=round(env.dt * 5 * 1000),
        loop=0,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "duration_s": env.steps * env.dt,
                "termination": info["termination"],
                "checkpoint": str(args.run / args.checkpoint),
                "arrow": "world target heading; visual only",
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
