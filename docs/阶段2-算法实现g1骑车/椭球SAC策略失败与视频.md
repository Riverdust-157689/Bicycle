# 椭球SAC策略失败：确定性评测、CPU回放与动作可执行性

对象：`runs/b_sac_ellipsoid_pilot_seed42/final.pt`。本次仅增加诊断和回放能力，未重训策略。

## GPU独立评测

`output/phase2/b_sac_ellipsoid_eval_seed20000`完成262,144次交互，4003个完成回合，平均1.2887秒，最近1000回合平均1.2865秒，20秒生存率0%，均为倾角超限。`policy_updated=false`，使用确定性SAC动作`tanh(mu)`。去掉探索噪声仍失败。

该评测没有保存逐帧qpos，无法从metrics.jsonl还原原始GPU视频。

## 已生成的真实权重CPU回放

加载原始最终权重和保存场景，使用与GPU评测相同的确定性动作计算方式。CPU重新积分动力学，初始条件由NumPy随机数生成；同一CPU种子下的不同控制方式相互配对，但不是与GPU的Torch随机初始化逐一配对。

- [4倍慢放MP4](../../output/phase2/b_sac_ellipsoid_policy_cpu_render/replay.mp4)
- [GIF](../../output/phase2/b_sac_ellipsoid_policy_cpu_render/replay.gif)
- [预览](../../output/phase2/b_sac_ellipsoid_policy_cpu_render/preview.png)

视频始终标注“CPU原生对照回放·非本次GPU轨迹”，含侧视、正视、侧倾、侧倾角速度、前向速度和实际转角。SAC直接输出双臂目标，没有把人工PD诊断参考当作策略目标。

```bash
MUJOCO_GL=egl uv run --locked --extra training python -I -m bicycle_sim.cli.render_rider_probe \
  --cpu-source-run runs/b_sac_ellipsoid_pilot_seed42 --cpu-controller policy \
  --seed 20000 --slow-motion 4 \
  --output output/phase2/b_sac_ellipsoid_policy_cpu_render
```

无可见GPU的当前环境使用Mesa EGL软件渲染成功。视频为1120×630、6.8秒，仿真回合实际1.52秒，播放含采样端点的时长与回合时长不严格成四倍。

## CPU十组配对对照

|控制方式|平均回合时长|20秒生存率|
|---|---:|---:|
|零动作|1.580秒|0%|
|随机动作|1.694秒|0%|
|最终SAC确定性策略|1.252秒|0%|
|反馈＋IK|20秒|100%|

当前策略在这个CPU对照中甚至比保持名义手臂目标更早失稳。此表不代替GPU评测结果。

在CPU种子20000示例中，0.4秒侧倾约+0.004 rad且角速度−0.096 rad/s，已验证PD的诊断转向参考为+0.056 rad，策略导致实际转向−0.050 rad；0.8秒参考+0.328 rad，实际仍−0.056 rad；1.2秒参考已到+0.35 rad，实际仍为−0.060 rad。最后才出现较大反向动作，侧倾已超限。PD不是唯一可能有效的控制律，此对照用于显示该策略没有及时形成有效的侧倾纠正。

## 最明确的失效机制：目标不相容、力矩饱和、转向纠正失败

为避免只观察倒地后的异常，统一比较十个CPU种子在0.05–0.8秒的早期窗口：

|诊断指标|SAC策略|反馈＋IK|
|---|---:|---:|
|双臂关节目标跟踪RMS|0.242 rad|0.00340 rad|
|触及原力矩上限的关节×时间比例|41.0%|0%|
|目标姿态的两掌合并几何残差|0.143 m|0.00320 m|

几何残差的计算方法：保持记录时的身体姿态，将14个双臂关节放到实际下发的PD目标；再在±0.35 rad内优化车把转角，使两掌到两车把连接点的距离平方和最小。表中为两掌误差合并L2范数的采样均值，不是每只手的实际脱离距离，也不是声称两手真的离开车把14 cm。

这表明策略指令在当前身体姿态及转向范围内通常不满足双掌连接几何。实际仿真由电机、关节和软约束折中：手臂达不到目标、产生较大反力和饱和，未能形成有效的协同转向。完整详细数据在`output/phase2/b_sac_ellipsoid_cpu_diagnosis/failure_analysis.json`。

![CPU失效轨迹及执行诊断](../../output/phase2/b_sac_ellipsoid_cpu_diagnosis/failure_summary.png)

这些证据确认了CPU对照中的控制行为失效机制，还不能唯一归因到SAC探索、熵/奖励尺度、样本量或14维动作表示。下一步优先验证动作的可执行性和协调，再开展有针对性的训练对照，避免直接追加长预算。

## 补录当前策略的GPU轨迹

已扩展原GPU探针，可加载冻结策略而不更新参数。默认加载source-run/final.pt，恢复该次环境配置；在环境初始化后才创建策略，避免干扰首批GPU初始状态。

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.probe_rider_gpu \
  --controller policy --source-run runs/b_sac_ellipsoid_pilot_seed42 \
  --device cuda:0 --num-envs 128 --seed 20000 \
  --first-episodes-only --record-only \
  --output output/phase2/b_sac_ellipsoid_policy_gpu_recorded
```

该命令只运行首批128个回合，全部结束后停止；保存模型、初始状态、环境0首回合姿态、动作、双臂目标/实际角度及力矩。独立PD参考仅作诊断，标记为未执行。可直接渲染真实GPU轨迹：

```bash
MUJOCO_GL=egl uv run --locked --extra training python -I -m bicycle_sim.cli.render_rider_probe \
  output/phase2/b_sac_ellipsoid_policy_gpu_recorded --slow-motion 4 \
  --output output/phase2/b_sac_ellipsoid_policy_gpu_render
```

CPU检查62项通过，1项CUDA测试跳过，包含SAC/PPO诊断动作与原评测mean_actions一致的回归检查。新增GPU策略录制尚未在本会话运行。
