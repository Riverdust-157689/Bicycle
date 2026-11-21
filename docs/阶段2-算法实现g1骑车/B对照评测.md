# B：零动作与扩大转向范围后的反馈控制对照

日期：2026-11-19。

SAC v4 独立GPU均值策略评测已完成：3,390个完成回合，平均1.517秒，20秒生存率0%，全部因倾角超限终止。训练累计平均1.462秒，因此探索噪声不足以解释失败。目前先补齐同配置对照，不追加长预算训练。

## 统一条件

从`output/phase2/b_sac_v4_eval_seed20000/run.json`加载环境配置，默认128个环境、种子20000、初始前进速度2 m/s，随机航向及小侧倾，20秒回合上限。双臂偏移上限±0.50 rad、目标变化速度4 rad/s；反馈控制转角参考上限±0.35 rad。曲柄参考仍为3.704 rad/s。

两个对照保留相同的局部座位焊接、理想手掌连接、相位腿部PD和曲柄外部电机，不添加转向电机：

- `zero`：14维手臂动作全零，即保持初始手臂目标；不意味着电机或连接没有施力。
- `feedback`：侧倾反馈输出转向参考，通过双臂IK和原有电机间接转动车把。

每个环境只统计首个完整回合，共128个，避免控制器不同导致重置次数不同、完成回合混入不同初始样本。已完成环境在等待其他环境期间仍正常重置和仿真，但后续回合不计入对照。停止时要求全部首回合完成。两次运行保存全部环境的初始qpos/qvel，并以绝对误差1e-7自动核对。

这个首批配对统计与既有SAC的3,390个重置回合统计口径不同：可作整体表现参考，不能声称对已有SAC日志做了逐回合配对。新对照间的逐环境差值才是配对结果。

## 运行

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.compare_rider_controls \
  --source-run output/phase2/b_sac_v4_eval_seed20000 \
  --device cuda:0 --num-envs 128 --seed 20000 \
  --output output/phase2/b_controls_v4
```

命令依次启动两个独立GPU进程。默认最多1100个控制步（约22秒），首批全部完成时提前结束；若修改回合时长，请同步增加`--steps`。对照结果不要求90%生存率才能保存或继续运行，未完成首批回合、非有限数、容量溢出等仍会失败并报告。输出目录必须是新目录。

结果：

- `comparison.json`：配置、初始状态核对、各模式时长/成功率/失败原因、逐环境反馈减零动作的时长差。
- `zero/`、`feedback/`：各自配置、编译场景、首批回合明细、全部环境初始状态、环境0的首个回合姿态与控制/力矩轨迹。

可进一步分析或渲染某个对照，例如：

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.analyze_rider_probe \
  output/phase2/b_controls_v4/feedback
MUJOCO_GL=egl uv run --locked --extra training python -I -m bicycle_sim.cli.render_rider_probe \
  output/phase2/b_controls_v4/feedback --episode 0 \
  --output output/phase2/b_controls_v4/feedback_render
```

本次会话CUDA不可见，已完成入口与CPU检查，尚无本轮两个GPU对照的运行结果。

## 椭球轮胎长回合对照

解析椭球的短步GPU诊断已显著缩小轮地接触差异，下一步用完整模型进行相同的128组首回合闭环对照：

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.compare_rider_controls \
  --source-run output/phase2/b_sac_v4_eval_seed20000 \
  --tire-collision ellipsoid \
  --device cuda:0 --num-envs 128 --seed 20000 \
  --output output/phase2/b_controls_ellipsoid_v1
```

采用20秒回合上限、相同反馈增益和控制范围。`--tire-collision ellipsoid`只对本次构建的模型生效，另存为`environment.tire_collision_model`；不传时沿用来源配置，旧配置缺少该字段时仍为原网格。默认训练配置和原始资产未切换。

椭球转换与短步诊断共用`models/tire.py`中的实现，通过XML重新编译更新碰撞包围盒，并保留半径、半宽、质量惯量和接触参数。IK的目标是掌部连接点，轮胎转换不改变手臂几何和参考表。

`comparison.json`包含两种控制的生存率、时长和平均回合结束车速；每个子目录的`results.json`记录逐回合结束车速及运行中的`trace`。首回合模式下，`trace`只统计尚未结束首回合的环境；已结束并重置的环境不计入速度曲线，因此要同时查看`active_initial_worlds`，不同时间的均值可能来自不同的存活子集。

对照完成后，核对新旧`initial_states.npz`，再与原网格的零动作1.613秒、反馈6.348秒及持续减速结果比较。不能仅凭短步对齐就认定长期平衡成功。检查：61项测试通过，1项CUDA测试跳过；这里尚未运行本轮GPU长回合。
