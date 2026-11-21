"""Plot the measured coast-down traces from audit_coasting."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("recording", type=Path)
a = p.parse_args()
rows = json.loads((a.recording / "results.json").read_text())["results"]
fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
for row in rows:
    b = row["rolling_friction_m"]
    label = f"b={b*1000:g} mm, sliding={row['sliding_mu']:g}"
    ax.plot([s["time_s"] for s in row["samples"]],
            [s["speed_mps"] for s in row["samples"]],
            label=label, linewidth=2.7 if b == .00405 else 1.6,
            linestyle="--" if row["sliding_mu"] > .8 else "-")
ax.axhline(.1, color="#777777", linewidth=.8, linestyle=":")
ax.set(xlabel="Time (s)", ylabel="Forward speed (m/s)",
       title="Unpowered coast: 43.4 kg total load, initial speed 2 m/s",
       xlim=(0, 25), ylim=(0, 2.1))
ax.grid(alpha=.2)
ax.legend(fontsize=9)
fig.savefig(a.recording / "coast_down.png", dpi=170)
fig.savefig(a.recording / "coast_down.svg")
