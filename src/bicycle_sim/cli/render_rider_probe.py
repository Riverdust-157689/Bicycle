"""Render recorded GPU poses, or an explicitly labelled native-CPU control replay."""

import argparse
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def visual_model(scene):
    root = ET.parse(scene).getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = Path(compiler.get("meshdir"))
        if not meshdir.is_absolute():
            compiler.set("meshdir", str((scene.parent / meshdir).resolve()))
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="replay_sky",
        type="skybox",
        builtin="flat",
        rgb1=".95 .96 .98",
        width="128",
        height="768",
    )
    ET.SubElement(
        asset,
        "texture",
        name="replay_floor_tex",
        type="2d",
        builtin="checker",
        rgb1=".90 .91 .93",
        rgb2=".76 .79 .83",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "material",
        name="replay_floor",
        texture="replay_floor_tex",
        texrepeat="2 2",
        reflectance="0",
    )
    root.find("./worldbody/geom[@name='ground']").set("material", "replay_floor")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    m.geom_size[m.geom("ground").id, :2] = 100
    m.vis.global_.offwidth = 640
    m.vis.global_.offheight = 480
    m.vis.headlight.ambient[:] = 0.6
    m.vis.headlight.diffuse[:] = 0.7
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recording", nargs="?", type=Path)
    p.add_argument("--cpu-source-run", type=Path)
    p.add_argument(
        "--cpu-controller", choices=("coordinated", "policy", "zero"), default="coordinated"
    )
    p.add_argument("--checkpoint", default="final.pt")
    p.add_argument("--slow-motion", type=float, default=1, help="Playback slowdown factor, e.g. 4")
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--episode", type=int, default=0)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument(
        "--env-id", type=int, help="Extract this environment from an all-env recording"
    )
    selection.add_argument(
        "--successful",
        action="store_true",
        help="Render the first actually successful recorded environment",
    )
    p.add_argument("--output", type=Path, default=Path("output/phase2/rider_probe_render"))
    args = p.parse_args()
    if not np.isfinite(args.slow_motion) or args.slow_motion < 1:
        p.error("slow-motion must be finite and at least 1")
    if bool(args.recording) == bool(args.cpu_source_run):
        p.error("Choose a recorded probe directory OR --cpu-source-run")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cpu_source_run:
        from bicycle_sim.cli.diagnose_rider import NativeProbe

        probe = NativeProbe(
            args.cpu_source_run,
            args.checkpoint,
            Path("output/phase1/rider_kinematic_reference.npz"),
        )
        result, _, _ = probe.rollout(args.cpu_controller, args.seed, record=True)
        recording = args.output / "cpu_recording"
        recording.mkdir(exist_ok=True)
        # arm_references() compiles a separate fixed rig. mj_saveLastXML would
        # otherwise reuse that last parsed XML and silently lose the bike root.
        shutil.copyfile(args.cpu_source_run / "scene.xml", recording / "scene.xml")
        np.savez_compressed(recording / "trajectory.npz", **probe.last_replay)
        meta = {
            "backend": "native_cpu_replay",
            "source_run": str(args.cpu_source_run),
            "seed": args.seed,
            "result": result,
            "controller": f"{probe.algorithm}_mean_policy"
            if args.cpu_controller == "policy"
            else args.cpu_controller,
            "checkpoint": args.checkpoint,
        }
        (recording / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    else:
        recording = args.recording
        if not (recording / "trajectory.npz").is_file():
            p.error(
                "This run did not record poses. Re-run the updated probe into a new directory first."
            )
        meta = json.loads((recording / "run.json").read_text())
    from bicycle_sim.rl.rider_recording import load_recorded_episode

    try:
        meta, track = load_recorded_episode(
            recording, env_id=args.env_id, successful=args.successful
        )
    except ValueError as exc:
        p.error(str(exc))
    indices = np.flatnonzero(track["episode_id"] == args.episode)
    if not len(indices):
        p.error("Requested episode is not present")
    is_cpu = meta["backend"] == "native_cpu_replay"
    label = (
        "CPU 原生对照回放 · 非本次 GPU 轨迹" if is_cpu else "GPU 实测轨迹回放 · 仅渲染保存的姿态"
    )
    m = visual_model(recording / "scene.xml")
    d = mujoco.MjData(m)
    if track["qpos"].shape[1] != m.nq:
        p.error("Trajectory and model do not match")
    foot_gaps = None
    if meta.get("whole_body_control") and "support_normal_force_N" in track:
        from bicycle_sim.cli.analyze_rider_feet import audit_feet

        episode_track = {k: v[indices] for k, v in track.items()}
        feet_report, episode_gaps = audit_feet(m, episode_track)
        foot_gaps = np.full((len(track["time_s"]), 2), np.nan)
        foot_gaps[indices] = episode_gaps
        feet_report["record_env"] = meta.get("record_env", 0)
        (args.output / "feet_audit.json").write_text(json.dumps(feet_report, indent=2) + "\n")
    body = m.body("bike_frame").id
    fontpath = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font = ImageFont.truetype(str(fontpath), 22) if fontpath.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(fontpath), 18) if fontpath.exists() else font
    stride = 5
    selected = indices[::stride]
    if selected[-1] != indices[-1]:
        selected = np.r_[selected, indices[-1]]
    width, height = 1120, 630
    fps = 10 / args.slow_motion
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(args.output / "replay.mp4"),
    ]
    frames = []
    with (
        subprocess.Popen(cmd, stdin=subprocess.PIPE) as movie,
        mujoco.Renderer(m, height=480, width=640) as renderer,
    ):
        for index in selected:
            d.qpos[:] = track["qpos"][index]
            if "qvel" in track:
                d.qvel[:] = track["qvel"][index]
            mujoco.mj_forward(m, d)
            R = d.xmat[body].reshape(3, 3)
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            center = d.xpos[body] + np.array(
                [0.35 * np.cos(np.radians(yaw)), 0.35 * np.sin(np.radians(yaw)), 0.80]
            )
            sheet = Image.new("RGB", (width, height), "#f3f5f8")
            draw = ImageDraw.Draw(sheet)
            draw.text(
                (18, 10),
                label + f" · {args.slow_motion:g}×慢放 · 环境{meta.get('record_env', 0)}",
                font=font,
                fill="#a62f32" if is_cpu else "#155e75",
            )
            desired = track["desired_steer_rad"][index]
            desired_text = f"{np.degrees(desired):+.2f}°" if np.isfinite(desired) else "—"
            is_policy = meta.get("controller", "").endswith("_mean_policy")
            steering_text = f"转向目标={desired_text}"
            if is_policy:
                steering_text = (
                    "全身35维策略，无显式转角目标"
                    if meta.get("whole_body_control")
                    else "策略直接控制双臂，无显式转角目标"
                )
            draw.text(
                (18, 47),
                f"t={track['time_s'][index]:.2f}s   侧倾={np.degrees(track['roll_rad'][index]):+.2f}°   侧倾角速度={track['roll_rate_rad_s'][index]:+.3f} rad/s",
                font=small,
                fill="#263445",
            )
            draw.text(
                (18, 77),
                f"速度={track['speed_mps'][index]:.2f} m/s   实际转向={np.degrees(track['actual_steer_rad'][index]):+.2f}°   {steering_text}",
                font=small,
                fill="#263445",
            )
            if foot_gaps is not None:
                force = track["support_normal_force_N"][index, 1:]
                gap = foot_gaps[index] * 100
                draw.text(
                    (18, 99),
                    f"脚踏法向力 左{force[0]:.1f}/右{force[1]:.1f} N；鞋底—踏板几何间距 左{gap[0]:+.1f}/右{gap[1]:+.1f} cm",
                    font=small,
                    fill="#263445",
                )
            for left, azimuth, title, viewwidth in [
                (0, yaw + 90, "侧视：观察踩踏与上肢", 640),
                (640, yaw + 180, "正视：观察侧倾与双手", 480),
            ]:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = center
                camera.distance = 2.65
                camera.azimuth = azimuth
                camera.elevation = -8
                renderer.update_scene(d, camera=camera)
                view = Image.fromarray(renderer.render().copy())
                if viewwidth != 640:
                    view = view.crop((80, 0, 560, 480))
                sheet.paste(view, (left, 120))
                draw.text((left + 14, 121), title, font=small, fill="#263445")
            draw.text(
                (18, 604),
                (
                    (
                        "全身35维策略 → 腿部直接关节目标；无座位／掌把连接"
                        if meta.get("leg_control_mode") == "direct_position"
                        else "均值策略 → 全身35维目标 → 接触骑行；无座位／掌把连接"
                    )
                    if meta.get("whole_body_control") and is_policy
                    else (
                        "均值策略 → 双臂＋六指闭合 → 接触握把；仅保留座位辅助连接"
                        if meta.get("hand_control_enabled")
                        else "均值策略 → 双臂目标 → 关节 PD → 手把作用力；保留座位／掌把辅助连接"
                    )
                    if is_policy
                    else (
                        (
                            "全身零动作 → 腿部关节中点；无座位／掌把连接"
                            if meta.get("leg_control_mode") == "direct_position"
                            else "全身35维零残差 → 接触骑行；无座位／掌把连接"
                        )
                        if meta.get("whole_body_control")
                        else "侧倾反馈 → 转向参考 → 双臂 IK → 关节 PD → 手把作用力；保留座位／掌把辅助连接"
                    )
                ),
                font=small,
                fill="#374151",
            )
            movie.stdin.write(np.asarray(sheet, dtype=np.uint8).tobytes())
            frames.append(sheet)
        movie.stdin.close()
        movie.wait()
        if movie.returncode:
            raise RuntimeError("ffmpeg failed")
    frames[0].save(
        args.output / "replay.gif",
        save_all=True,
        append_images=frames[1:],
        duration=round(100 * args.slow_motion),
        loop=0,
    )
    frames[min(30, len(frames) - 1)].save(args.output / "preview.png")
    (args.output / "render.json").write_text(
        json.dumps(
            {
                "source_backend": meta["backend"],
                "source_recording": str(recording),
                "label": label,
                "episode": args.episode,
                "record_env": meta.get("record_env", 0),
                "cohort_selection": meta.get("cohort_selection"),
                "duration_s": float(track["time_s"][indices[-1]]),
                "render_only_no_physics_steps": True,
                "controller": meta.get("controller"),
                "slow_motion": args.slow_motion,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output / "replay.mp4")
    print(args.output / "replay.gif")


if __name__ == "__main__":
    main()
