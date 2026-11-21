"""Compare early probe speed before any GPU world resets; no paired-state claim."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

root = Path("output/phase2")
gpu = json.loads((root / "b_gpu_controller_probe/results.json").read_text())
with np.load(root / "pd_ik_cpu_render/cpu_recording/trajectory.npz") as d:
    cpu = {k: d[k] for k in d.files}
first_failure = min(r["step"] for r in gpu["episodes"])
trace = [r for r in gpu["trace"] if r["step"] < first_failure]
t = np.array([r["step"] * 0.02 for r in trace])
font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
mask = cpu["time_s"] <= t[-1]
ax[0].plot(cpu["time_s"][mask], cpu["speed_mps"][mask], label="CPU单回合对照")
ax[0].plot(t, [r["mean_speed_mps"] for r in trace], "o-", label="GPU 32环境平均")
ax[0].set(xlabel="时间 / s", ylabel="前向速度 / (m/s)", title="GPU在约5秒内明显减速")
ax[0].legend()
ax[1].plot(cpu["time_s"][mask], np.rad2deg(abs(cpu["roll_rad"][mask])), label="CPU单回合侧倾")
ax[1].plot(t, np.rad2deg([r["mean_tilt_rad"] for r in trace]), "o-", label="GPU平均整体倾角")
ax[1].plot(t, np.rad2deg([r["max_tilt_rad"] for r in trace]), "--", label="GPU最大整体倾角")
ax[1].set(xlabel="时间 / s", ylabel="角度 / °", title="仅展示GPU首次复位前的数据")
ax[1].legend()
for a in ax:
    a.grid(alpha=0.25)
fig.suptitle("同一反馈＋双臂IK控制器：CPU成功不代表GPU已通过；初始状态非逐一配对")
out = root / "pd_ik_cpu_render/backend_comparison.png"
fig.savefig(out, dpi=170)
print(out)
