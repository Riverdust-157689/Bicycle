"""Draw the production tire sampling and convex-envelope explanation."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from bicycle_sim.models.tire import rounded_envelope_vertices

font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
points = np.array(rounded_envelope_vertices(0.27, 0.018, 192, 24)).reshape(192, 24, 3)
fig = plt.figure(figsize=(16, 7.2), dpi=170, facecolor="#f5f7fb")
ax = fig.add_subplot(131, projection="3d")
for i in range(0, 192, 8):
    p = np.vstack([points[i], points[i, 0]])
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#9cbbd8", lw=0.7)
a = np.arange(192) * 2 * np.pi / 192
ax.scatter(0.252 * np.cos(a), np.zeros_like(a), 0.252 * np.sin(a), s=3, color="#316a9c")
p = np.vstack([points[0], points[0, 0]])
ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#dd7b26", lw=2)
ax.set_title("① 沿车轮圆周，取192个位置", fontproperties=font, fontsize=15, pad=18)
ax.view_init(elev=20, azim=-55)
ax.set_box_aspect((1, 0.3, 1))
ax.set_axis_off()
ax.text2D(
    0.06,
    -0.04,
    "像沿着一个游泳圈，均匀标出192处",
    transform=ax.transAxes,
    fontproperties=font,
    fontsize=12,
)

ax2 = fig.add_subplot(132)
b = np.arange(25) * 2 * np.pi / 24
ax2.plot(18 * np.cos(b), 18 * np.sin(b), color="#dd7b26", lw=2)
ax2.scatter(18 * np.cos(b[:-1]), 18 * np.sin(b[:-1]), s=25, color="#dd7b26")
ax2.annotate(
    "", xy=(-18, -25), xytext=(18, -25), arrowprops={"arrowstyle": "<->", "color": "#475569"}
)
ax2.text(0, -31, "胎宽36 mm", ha="center", fontproperties=font, fontsize=12)
ax2.set(xlim=(-31, 31), ylim=(-38, 30), aspect="equal")
ax2.set_axis_off()
ax2.set_title("② 每处小圆截面，再取24个点", fontproperties=font, fontsize=15, pad=18)
ax2.text(
    0.03,
    -0.04,
    "小圆半径18 mm；图中已放大",
    transform=ax2.transAxes,
    fontproperties=font,
    fontsize=12,
)

ax3 = fig.add_subplot(133, projection="3d")
v = points.reshape(-1, 3)
hull = ConvexHull(v)
faces = v[hull.simplices]
colors = np.tile([0.20, 0.48, 0.63, 0.90], (len(faces), 1))
light = np.array([0.2, -0.5, 0.8])
light /= np.linalg.norm(light)
shade = 0.68 + 0.32 * np.abs(hull.equations[:, :3] @ light)
colors[:, :3] *= shade[:, None]
collection = Poly3DCollection(faces, facecolors=colors, edgecolors="none")
ax3.add_collection3d(collection)
ax3.set(xlim=(-0.29, 0.29), ylim=(-0.025, 0.025), zlim=(-0.29, 0.29))
ax3.set_box_aspect((1, 0.3, 1))
ax3.view_init(elev=20, azim=-55)
ax3.set_axis_off()
ax3.set_title("③ 用这些点的“凸包”作为碰撞外形", fontproperties=font, fontsize=15, pad=18)
ax3.text2D(
    0.03,
    -0.04,
    "外缘圆滑；凸包会填满轮心",
    transform=ax3.transAxes,
    fontproperties=font,
    fontsize=12,
)

fig.suptitle(
    "192 × 24 圆冠轮胎，到底是什么？", fontproperties=font, fontsize=23, y=0.95, color="#203247"
)
fig.text(
    0.5,
    0.16,
    "每轮：192 × 24 = 4608个输入采样点 → 取外包络 → 1个碰撞体",
    ha="center",
    fontproperties=font,
    fontsize=18,
    color="#1f6b58",
)
fig.text(
    0.5,
    0.095,
    "不是4608个小球，也不是192个独立零件；碰撞体仍是刚体，软接触由求解器参数控制。",
    ha="center",
    fontproperties=font,
    fontsize=13,
    color="#475569",
)
fig.text(
    0.5,
    0.045,
    "圆周采样控制滚动方向的细分，截面采样控制胎冠圆滑程度。图中厚度略夸张，轮胎外半径实际为270 mm。",
    ha="center",
    fontproperties=font,
    fontsize=11,
    color="#64748b",
)
fig.subplots_adjust(top=0.78, bottom=0.27, left=0.02, right=0.98, wspace=0.18)
out = Path("output/phase1/rounded_tire_adoption")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "tire_explained.png", bbox_inches="tight")
fig.savefig(out / "tire_explained.svg", bbox_inches="tight")
