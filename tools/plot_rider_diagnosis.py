"""Plot the completed B run and independent native-CPU control probes."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run = Path("runs/b_ppo_v2_seed42")
rows = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
probe = json.loads(Path("output/phase2/b_diagnosis/results.json").read_text())
font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if font.exists():
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
x = np.array([r["transitions"] for r in rows]) / 1e6
y = np.array([r["mean_episode_seconds"] for r in rows])
axes[0].plot(x, y, color="#2563eb", label="训练累计平均")
window = 10
xs = []
ys = []
for j in range(window, len(rows), window):
    a, b = rows[j - window], rows[j]
    n = b["completed_episodes"] - a["completed_episodes"]
    if n:
        xs.append(b["transitions"] / 1e6)
        ys.append(
            (
                b["mean_episode_seconds"] * b["completed_episodes"]
                - a["mean_episode_seconds"] * a["completed_episodes"]
            )
            / n
        )
axes[0].plot(xs, ys, color="#f59e0b", alpha=0.8, label="区间均值（累计量差分）")
axes[0].set(
    xlabel="交互次数（百万）",
    ylabel="回合持续时间 / s",
    title="PPO 训练没有形成改善趋势",
    ylim=(0, 3),
)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.2)
modes = ["zero", "random", "policy", "coordinated"]
labels = ["固定名义手臂", "随机动作", "最终 PPO 均值", "反馈转向＋双臂 IK"]
means = [np.mean([r["duration_s"] for r in probe["episodes"] if r["mode"] == m]) for m in modes]
axes[1].bar(range(4), means, color=["#94a3b8", "#94a3b8", "#ef4444", "#16a34a"])
axes[1].set_xticks(range(4), labels, rotation=12, ha="right", fontsize=9)
axes[1].set(ylabel="回合持续时间 / s", title="相同 CPU 场景：10 个未见初始条件", ylim=(0, 22))
for i, v in enumerate(means):
    axes[1].text(i, v + 0.3, f"{v:.2f}s", ha="center", fontsize=10)
axes[1].grid(axis="y", alpha=0.2)
fig.suptitle("B 首轮诊断：策略失败；结构化双臂控制可持续骑行", fontsize=13)
out = Path("output/phase2/b_diagnosis/summary.png")
fig.savefig(out, dpi=180)
plt.close(fig)
print(out)
