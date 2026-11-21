"""Inspect recorded GPU feedback, steering response and control saturation."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(track, cfg, episode=0, controller="roll_PD_to_arm_IK"):
    ids = np.flatnonzero(track["episode_id"] == episode)
    if len(ids) < 2:
        raise ValueError("Episode needs at least two recorded frames")
    d = {k: v[ids] for k, v in track.items()}
    t = d["time_s"]
    target = d["desired_steer_rad"]
    actual = d["actual_steer_rad"]
    limit = cfg.get("steering_reference_limit_rad", 0.25)
    expected = np.clip(
        -cfg.get("controller_roll_gain", 3.0) * d["roll_rad"][:-1]
        - cfg.get("controller_roll_rate_gain", 0.7) * d["roll_rate_rad_s"][:-1],
        -limit,
        limit,
    )
    if controller == "zero":
        expected = np.zeros_like(expected)
    valid = np.isfinite(target) & (t > 0.2)
    unsaturated = valid & (abs(target) < limit * 0.99)
    saturated = valid & (abs(target) >= limit * 0.99)
    error = target - actual
    corr = (
        float(np.corrcoef(target[valid], actual[valid])[0, 1])
        if valid.any() and np.std(target[valid]) > 1e-8 and np.std(actual[valid]) > 1e-8
        else None
    )
    out = {
        "episode": episode,
        "duration_s": float(t[-1]),
        "feedback_formula_max_error_rad": float(np.nanmax(abs(expected - target[1:])))
        if np.isfinite(target[1:]).any()
        else None,
        "target_actual_correlation": corr,
        "tracking_rms_rad": float(np.sqrt(np.mean(error[valid] ** 2))) if valid.any() else None,
        "unsaturated_tracking_rms_rad": float(np.sqrt(np.mean(error[unsaturated] ** 2)))
        if unsaturated.any()
        else None,
        "steer_limit_rad": limit,
        "reference_saturation_fraction": float(saturated[valid].mean()) if valid.any() else None,
        "first_reference_saturation_s": float(t[np.flatnonzero(saturated)[0]])
        if saturated.any()
        else None,
        "actual_steer_max_abs_rad": float(np.max(abs(actual))),
        "initial_speed_mps": float(d["speed_mps"][0]),
        "final_speed_mps": float(d["speed_mps"][-1]),
        "motor_peak_abs_torque_Nm": float(np.max(abs(d["motor_torque_Nm"]))),
        "motor_torque_limit_Nm": cfg["max_crank_torque_Nm"],
        "motor_saturation_fraction": float(
            np.mean(abs(d["motor_torque_Nm"]) >= cfg["max_crank_torque_Nm"] - 1e-3)
        ),
        "terminated": bool(d["terminated"][-1]),
        "truncated": bool(d["truncated"][-1]),
    }
    if "arm_actual_rad" in d:
        out["arm_joint_tracking_rms_rad"] = float(
            np.sqrt(np.mean((d["arm_executed_target_rad"] - d["arm_actual_rad"]) ** 2))
        )
        target_key = (
            "arm_requested_target_rad" if "arm_requested_target_rad" in d else "arm_ik_target_rad"
        )
        metric_key = (
            "requested_to_executed_target_max_difference_rad"
            if target_key == "arm_requested_target_rad"
            else "ik_to_executed_target_max_difference_rad"
        )
        out[metric_key] = float(np.max(abs(d[target_key] - d["arm_executed_target_rad"])))
        out["arm_peak_abs_actual_motor_torque_Nm"] = float(np.max(abs(d["arm_actual_torque_Nm"])))
    return out, d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recording", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    meta = json.loads((args.recording / "run.json").read_text())
    cfg = meta["environment"]
    with np.load(args.recording / "trajectory.npz") as raw:
        track = {k: raw[k] for k in raw.files}
        report, d = summarize(track, cfg, args.episode, meta.get("controller", "roll_PD_to_arm_IK"))
    output = args.output or args.recording / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "control_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    t = d["time_s"]
    axes[0, 0].plot(t, np.rad2deg(d["roll_rad"]), label="侧倾角 / °")
    axes[0, 0].set_title("GPU实测车架侧倾")
    if meta.get("controller", "").endswith("_mean_policy"):
        axes[0, 1].plot(t, d["diagnostic_pd_steer_rad"], label="PD诊断参考（未执行）")
    else:
        axes[0, 1].plot(t, d["desired_steer_rad"], label="反馈转向参考")
    axes[0, 1].plot(t, d["actual_steer_rad"], label="实际转向")
    for sign in [-1, 1]:
        axes[0, 1].axhline(sign * report["steer_limit_rad"], color="gray", ls="--", alpha=0.5)
    axes[0, 1].set_title("参考是否执行、何时达到限幅")
    axes[0, 1].set_ylabel("rad")
    axes[1, 0].plot(t, d["speed_mps"], label="车体系前向速度")
    axes[1, 0].plot(t, 0.27 * d["rear_speed_rad_s"], label="后轮轮缘速度（近似对照）", alpha=0.7)
    axes[1, 0].set_title("低速段与倾倒时刻")
    axes[1, 0].set_ylabel("m/s")
    axes[1, 1].plot(t, d["motor_torque_Nm"], label="曲柄电机扭矩")
    axes[1, 1].plot(t, d["freewheel_rear_torque_Nm"], label="自由轮向后轮传递扭矩")
    axes[1, 1].set_title("驱动链确有输出，检查是否饱和")
    axes[1, 1].set_ylabel("N·m")
    for ax in axes.flat:
        ax.set_xlabel("回合时间 / s")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle("GPU实测：侧倾反馈控制器的转向响应与驱动状态")
    fig.savefig(output / "control_response.png", dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(output / "control_response.png")


if __name__ == "__main__":
    main()
