"""Render orthographic inspection views of the mounted kinematic keyframe."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v3 as iio
import mujoco
from PIL import Image, ImageDraw, ImageFont

from bicycle_sim.control.hand_contact import audit_hands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path, default=Path("assets/scenes/rider_fit_diagnostic.xml")
    )
    parser.add_argument("--output", type=Path, default=Path("output/phase1/three_views"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = ET.parse(args.scene).getroot()
    geometry = json.loads(root.find("./custom/text[@name='bicycle_geometry']").get("data"))
    # Rendering-only changes: source geometry, joints, pose and physics remain intact.
    compiler = root.find("compiler")
    compiler.set("meshdir", str((args.scene.parent / compiler.get("meshdir")).resolve()))
    ET.SubElement(
        root.find("asset"),
        "texture",
        name="inspection_sky",
        type="skybox",
        builtin="flat",
        rgb1="0.95 0.96 0.98",
        width="512",
        height="3072",
    )
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    m.vis.global_.offwidth = m.vis.global_.offheight = 1024
    m.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    m.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    m.vis.headlight.specular[:] = [0.1, 0.1, 0.1]
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "mounted_kinematic")
    if key < 0:
        raise ValueError("Mounted kinematic keyframe is required")
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    hands_passed = all(h["passed"] for h in audit_hands(m, d).values())
    camera = mujoco.MjvCamera()
    camera.orthographic = 1
    camera.lookat[:] = [geometry["wheelbase"] / 2, 0, 0.76]
    camera.distance = 2.35
    views = [
        ("front", "正视图 · 从车头看向车尾", 0, 0),
        ("side", "左侧视图", 90, 0),
        ("top", "俯视图", 90, -90),
    ]
    frames = []
    with mujoco.Renderer(m, height=1024, width=1024) as renderer:
        for name, title, azimuth, elevation in views:
            camera.azimuth, camera.elevation = azimuth, elevation
            renderer.update_scene(d, camera=camera)
            frame = renderer.render().copy()
            iio.imwrite(args.output / f"rider_{name}.png", frame)
            frames.append((title, Image.fromarray(frame)))
        hand_camera = mujoco.MjvCamera()
        grip = geometry["grip_centers"][0]
        hand_camera.lookat[:] = [grip[0], 0, grip[2] + 0.02]
        hand_camera.distance = 0.9
        hand_camera.azimuth, hand_camera.elevation = 0, -25
        renderer.update_scene(d, camera=hand_camera)
        iio.imwrite(args.output / "rider_hands.png", renderer.render())
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font_path).exists():
        title_font = ImageFont.truetype(font_path, 38)
        label_font = ImageFont.truetype(font_path, 27)
    else:
        title_font = label_font = ImageFont.load_default()
    sheet = Image.new("RGB", (2460, 1020), "#f2f4f8")
    draw = ImageDraw.Draw(sheet)
    draw.text((30, 20), "G1 + Dex3 骑乘姿态三视图", font=title_font, fill="#1f2937")
    draw.text(
        (30, 80),
        f"轴距 {geometry['wheelbase']:.2f} m｜座高 {geometry['saddle_top'][2]:.2f} m｜握位高 {geometry['grip_centers'][0][2]:.2f} m｜转向轴偏竖直 {90 - geometry['head_angle_from_horizontal_deg']:.0f}°",
        font=label_font,
        fill="#475569",
    )
    for i, (title, frame) in enumerate(frames):
        x = 20 + i * 815
        frame = frame.resize((790, 790), Image.Resampling.LANCZOS)
        sheet.paste(frame, (x, 170))
        draw.text((x + 12, 125), title, font=label_font, fill="#1f2937")
    draw.text(
        (30, 974),
        (
            "初始掌心向下、近零间隙；已做几何检查，未做动态平衡。"
            if hands_passed
            else "运动学参考姿态；未做动态平衡。手部接触检查尚未通过。"
        ),
        font=label_font,
        fill="#7c3f20",
    )
    sheet.save(args.output / "rider_three_views.png")
    (args.output / "render_metadata.json").write_text(
        json.dumps(
            {
                "scene": str(args.scene),
                "keyframe": "mounted_kinematic",
                "orthographic": True,
                "camera_distance": camera.distance,
                "geometry_changed_for_render": False,
                "views": [{"name": n, "azimuth": a, "elevation": e} for n, _, a, e in views],
                "scope": "Kinematic inspection only; no dynamic balance or load-bearing validation.",
                "initial_hand_geometry_passed": hands_passed,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output / "rider_three_views.png")


if __name__ == "__main__":
    main()
