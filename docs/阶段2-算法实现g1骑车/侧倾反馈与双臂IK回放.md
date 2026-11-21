# 侧倾反馈＋双臂 IK：控制方法、回放与GPU结果

日期：2026-11-19。此处是人工设计的可控性探针，不是学到的策略。

## 实际怎么控制

1. 读取自行车车架的侧倾角`phi=atan2(R[2,1],R[2,2])`及车架IMU的侧倾角速度`omega_x`，按当前坐标符号计算：

   `steer_ref = clip(-3*phi - 0.7*omega_x, -0.25, 0.25)`。

   系数是在CPU上小范围调试得到的，不保证任意车速、负载和后端都稳定。

2. 预先求一张双臂IK表，转向采样为`[-0.25,-0.15,-0.05,0,0.05,0.15,0.25] rad`。对每个假想转向角，先用车辆运动学计算两掌连接点应到达的位置，再优化双臂14个关节，使两掌参考点接近目标。残差为两掌的位置误差，加`0.002*(q_arm-q_arm_initial)`的小姿态正则；同时遵守原关节限位和各关节相对名义姿态±0.25 rad的范围。

   这里求的是掌心点位置，不是Dex3抓握；没有强制掌心姿态或优化手指。±0.15 rad内位置残差很小，±0.25 rad两端约有10.5 mm残差，不能当成任意转向角都完全可达。

3. 运行时按`steer_ref`在IK表中线性插值，得到14个手臂关节目标，再经过2 rad/s目标变化率限制和原有有限力矩位置PD执行。手臂通过两掌与车把的理想局部点连接传力，实际改变转向。

```mermaid
flowchart LR
    A[侧倾角与侧倾角速度] --> B[PD反馈及转角限幅]
    B --> C[双臂IK表插值]
    C --> D[14个关节目标与PD]
    D --> E[手掌与车把作用力]
    E --> F[实际转向与车辆运动]
    F --> A
```

只有离线生成IK表时设置了假想车把角度；运行中的控制器不直接设置车把qpos/qvel，没有直接转向执行器。座位局部焊接、掌把理想连接、曲柄速度伺服和腿部相位参考仍然存在，不是完整自然接触骑行。

## 已生成的CPU可视化

- [20秒MP4](../../output/phase2/pd_ik_cpu_render/replay.mp4)
- [GIF](../../output/phase2/pd_ik_cpu_render/replay.gif)
- [预览](../../output/phase2/pd_ik_cpu_render/preview.png)

画面为侧视和正视，标注侧倾、角速度、速度以及目标／实际转向角。**这是原生CPU对照，不是本次GPU实测轨迹**，画面顶端始终保留这一标识。初始侧倾范围仅±0.02 rad，没有运行中扰动，所以稳定后上肢修正幅度很小。

```bash
MUJOCO_GL=egl uv run --locked --extra training python -I -m bicycle_sim.cli.render_rider_probe \
  --cpu-source-run runs/b_ppo_v2_seed42 --seed 20000 \
  --output output/phase2/pd_ik_cpu_render
```

CPU录制保存到`cpu_recording/scene.xml`和`trajectory.npz`。渲染器逐帧设置录制的姿态，没有重新积分动力学。渲染只增加背景和材质，不锁车架、不改变质量或惯量。

## 本次GPU探针实际没有通过

`output/phase2/b_gpu_controller_probe/results.json`：96个完成回合，生存率0%，平均持续约6.149秒，范围5.48–7.98秒，全部为倾角超限，没有非轮部件触地终止。

这比PPO约1.6秒更长，但远未达到20秒。失败时实际转向多接近±0.234 rad，已接近该动作／IK范围下的有效转角上限。

GPU首次复位前的平均前进速度：约0.02秒为2.00 m/s、1.02秒为1.83、3.02秒为1.50、5.02秒为1.16。CPU该对照回合在5秒约1.94 m/s。两者初始状态不是逐一配对，但这个持续减速差异值得优先核查；尚不能确认来自接触、传动、关节阻尼、腿部随动还是其他后端差异。

![首次复位前的CPU/GPU速度对照](../../output/phase2/pd_ik_cpu_render/backend_comparison.png)

此处的0%仅表示未保持20秒，不能据此判断控制器没有执行。后续已补录GPU逐帧轨迹并确认其有响应，见下文。CPU/GPU动力学差异继续记录，不作为开始SAC试验的阻断条件。

## 补录真实GPU轨迹与直接渲染

旧程序只保存了回合统计和各环境平均轨迹，没有逐帧qpos，不能从旧结果文件还原这次GPU画面。新版探针默认录制环境0，另存完整`scene.xml`和`trajectory.npz`，包含姿态、速度、侧倾／角速度、目标／实际转向、前后轮与曲柄转速、电机扭矩／功率、自由轮传动力矩及终止标记。

先在GPU重新录制约8秒，使用新目录；这不是重训：

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.probe_rider_gpu \
  --device cuda:0 --num-envs 32 --steps 400 --seed 20000 \
  --record-env 0 --record-only --output output/phase2/b_gpu_probe_recorded
```

`--record-only`只录制，不把短于20秒的记录拿去做成功率验收；终止事件仍会如实记录。随后执行渲染：

```bash
MUJOCO_GL=egl uv run --locked --extra training python -I -m bicycle_sim.cli.render_rider_probe \
  output/phase2/b_gpu_probe_recorded --episode 0 \
  --output output/phase2/b_gpu_probe_render
```

渲染不需要CUDA物理仿真，只需要可用EGL／OpenGL。输出的MP4和GIF才是补录的真实GPU姿态轨迹，而非CPU重算结果。该记录命令不足20秒，只用于取得倾倒过程，不能用于报告20秒成功率。


## GPU执行证据与扩大控制范围（当前配置）

对补录的环境0首个回合（6.42秒）分析：按上一帧侧倾和角速度重算控制律，与下一帧记录的目标最大差仅1.79e-8 rad；目标与实际转角相关系数0.9988，去除最初0.2秒后全回合跟踪RMS为0.00979 rad。1秒时目标/实际约0.01315/0.01238 rad，5秒约0.10387/0.09984 rad。5.54秒开始目标饱和，6秒实际转角约0.233 rad，6.42秒侧倾超限。这足以确认GPU控制器确实执行并产生转向；不能把失稳解释成“控制器没有工作”。

同回合前向速度由2.0降至约1.06 m/s，曲柄电机扭矩峰值约3.01 N·m，未触及8 N·m上限。转向饱和与低速同时出现，符合进入原控制器可控范围之外的可能性，但还不能确定单一因果。

![GPU目标与实际转向、速度和驱动力矩](../../output/phase2/b_gpu_probe_recorded/analysis/control_response.png)

当前`configs/rl/bicycle_b.json`已调整：

|参数|旧值|当前值|含义|
|---|---:|---:|---|
|转向参考上限|±0.25 rad|±0.35 rad（约20°）|用满现有转向关节物理范围|
|双臂目标相对初始姿态偏移|±0.25 rad|±0.50 rad|让手臂能够配合更大车把转角，仍受原关节限位约束|
|双臂目标变化速度上限|2 rad/s|4 rad/s|提高间接转向响应速度|

前叉转向轴没有独立电机，也没有单独设置角速度裁剪；其实际转速由手臂动作、力矩和接触约束决定。因此4 rad/s是各手臂关节目标的变化速度，不是保证前叉达到的转速。曲柄仍由外部电机保持3.704 rad/s参考，它不是RL动作，也未设置额外的实际转速裁剪。需要区分曲柄转速与转向杆转速。

固定车架台架验证：两端IK手部位置联合误差约0.36 mm；实际转向约-0.3527/+0.3521 rad，最后0.5秒RMS分别0.00268/0.00216 rad。物理限位是软约束，因此有小幅越界。无求解器警告；大动作过渡期间原25 N·m手臂力矩上限有约11–12%的关节采样饱和，末段最深接触约3.7 mm。保留原力矩上限，不凭空增加电机能力。该台架结果不代表扩大范围后GPU自由骑行一定更好。

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.audit_controller_limits
uv run --locked --extra training python -I -m bicycle_sim.cli.probe_rider_gpu \
  --device cuda:0 --num-envs 32 --steps 1100 --record-only \
  --output output/phase2/b_gpu_limits_v4
uv run --locked --extra training python -I -m bicycle_sim.cli.analyze_rider_probe \
  output/phase2/b_gpu_limits_v4
```

新探针额外保存未裁剪转向需求、IK目标、执行目标、手臂实际角度与实际力矩，便于区分转向参考饱和、目标速度裁剪与电机跟不上。`--record-only`不施加20秒成功率门槛，仍统计真实失败与生存率。

接下来可以直接开展SAC短预算训练，不必等CPU/GPU结果完全一致。SAC沿用skrl实现，仍直接学习14维双臂动作，不把上述人工PD/IK叠加到策略上。先检查GPU更新可用，再训练约105万交互；评估看近期回合长度、20秒生存率、动作饱和与侧倾。如果延长预算，再运行新目录，当前入口不支持续训，不把载入`--evaluate`误作恢复训练。

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.train_rider \
  --algorithm sac --device cuda:0 --smoke \
  --output output/phase2/b_sac_smoke_v4
uv run --locked --extra training python -I -m bicycle_sim.cli.train_rider \
  --algorithm sac --device cuda:0 --num-envs 128 --transitions 1048576 \
  --output runs/b_sac_v4_pilot_seed42
```

若显存不足，用新目录并加`--num-envs 32`；交互预算保持一致。需要对比PPO时，用相同配置、种子、环境数和交互预算，替换算法为`ppo`并使用新目录。新PPO包含先前采样概率修复，不能把差异全部归因于SAC。当前会话无法访问CUDA，新的GPU探针和SAC结果需以本机运行产物为准。
