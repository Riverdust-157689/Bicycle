# 自然抓握骑行：20维SAC短训

本文为实验B的20维自然抓握版本：策略控制双臂与手指，双腿仍由参考控制器驱动。 实验编号及历史文件名对应见[实验划分与结果索引](实验划分与结果索引.md)。

在已通过CPU/GPU固定车架抓握与转向台架的基础上，尝试自由自行车上的SAC。保留座位局部连接、曲柄外部驱动与相位腿部参考，取消两条掌把理想连接。轮胎使用当前已验证数值可控性的解析椭球。

## 本次任务

- 配置：`configs/rl/bicycle_b_contact_sac.json`。
- 动作20维：双臂14维关节目标偏移；六根手指各一个闭合调节量。
- 手指动作映射：`u = clip(0.9 + 0.1*a_hand, 0, 1)`，第一轮实际范围0.8–1.0；变化率0.5/s。该任务先学已握住后的握持调节，不学习寻找车把和从张开状态重新抓取。
- 指关节仍使用软件协同及有限力矩PD，尚非机械欠驱动。手指kp=5、kv=0.15，原模型力矩上限保留。
- 观测148维：原有状态和动作历史扩展至20维，额外包含6个当前闭合量与6个逐指汇总法向力。原有全机器人关节状态已经包含手指角度/角速度，没有重复添加。
- 接触力直接从MuJoCo-Warp求解器提取，只聚合本侧手指与本侧车把接触；归一化比例50 N。双手有指端接触不等同于严格证明力闭合或无滑移。
- 保持原行进平衡奖励，额外增加最大0.1/s的逐指受力接触比例奖励，不按力大小奖励，避免鼓励越夹越紧。仍以倾角超限/非轮部件触地终止；暂不因瞬时失去手指接触立即终止，接触情况单独记录。

## 初始握持状态

`assets/references/dex3_handlebar_grasp.json`保存已验收GPU台架约3秒时的静止握持姿态、对应位置目标及来源哈希。它是可复现的项目参考资产，运行时不依赖旧output目录。

重置时恢复整个已握持状态，再共同旋转车架和机器人根部以生成随机航向与小侧倾。保留参考姿态中机器人相对车架的微小俯仰，避免重置强行对齐两个根部导致手掌错位。双臂零动作对应原稳定握持的控制目标，手指零动作对应闭合量0.9。

腿部相位参考仍用原有文件，握持参考没有改变模型关节拓扑。运行记录保存握持参考哈希、20维动作/148维观测及实际辅助条件；旧14维任务继续按各自保存的配置加载。

## 运行

先验证GPU重置、初始双手接触和SAC短更新：

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.train_rider \
  --config configs/rl/bicycle_b_contact_sac.json \
  --algorithm sac --device cuda:0 --smoke \
  --output output/phase2/b_contact_sac_smoke_v1
```

通过后从头训练同样约105万次交互：

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.train_rider \
  --config configs/rl/bicycle_b_contact_sac.json \
  --algorithm sac --device cuda:0 --num-envs 128 --transitions 1048576 \
  --output runs/b_contact_sac_pilot_seed42
```

显存不足时用新目录及`--num-envs 32`，保留交互预算。此次与旧实验同时改变了抓握约束、动作空间和初始状态，因此用于新任务可行性判断，不将成绩差异全部归因到某个单一改动。

## 结果判断

除近期平均回合时长、20秒生存率、倾角/触地失败外，新增：

- `finger_contact_fraction`：六个指端组中法向合力大于0.5 N的比例。
- `both_hands_contact_fraction`：左右手至少各有一个手指组受力接触的环境比例；这是接触诊断，不是完整抓握质量判断。
- `finger_normal_force_mean_N`：逐指法向合力均值。
- `hand_closure_mean`：实际执行的闭合量均值。

如果接触比例迅速下降，先看手臂把手拉开还是手指松开；如果接触保持但仍约1秒倾倒，再检查转向协调与学习配置。不要仅以SAC损失下降判断成功。

评测最终确定性策略：

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.train_rider \
  --algorithm sac --device cuda:0 \
  --evaluate runs/b_contact_sac_pilot_seed42/final.pt \
  --seed 20000 --num-envs 128 --transitions 262144 \
  --output output/phase2/b_contact_sac_eval_seed20000
```

20维任务的逐帧GPU录制可用`probe_rider_gpu --controller policy --source-run runs/b_contact_sac_pilot_seed42`，再用现有渲染器播放。旧CPU策略诊断入口专用于14维任务，会明确拒绝20维模型，避免错误解释检查点。

## 当前检查边界

71项CPU测试通过，2项CUDA测试跳过，包含20维SAC/PPO网络更新、协同目标一致性、自然抓握模型无掌把连接、初始化接触、重置姿态旋转几何。原生CPU自由车0.5秒初始化检查中六指均受力，速度保持约2 m/s、无求解器警告。

当前会话CUDA不可见，新的20维GPU环境、冒烟和训练尚未在这里运行。固定车架抓握通过不代表自由骑行策略已学会平衡；第一轮仍是短预算可行性试验。
