# G1 + Dex3 Bicycle

G1 + Dex3机器人骑自行车的仿真与强化学习项目，研究目标是稳定主动骑行及后续的可微分世界模型学习。目前推进实验C：MuJoCo-Warp上的全身控制基线，可微分世界模型尚未接入。

## 阶段2实验划分

| 实验 | 策略控制权限 | 任务 |
| --- | --- | --- |
| A | 纯自行车2维：曲柄转速＋转向角 | 平衡与目标速度/世界方向跟踪 |
| B | 机器人上半身；已有双臂14维、双臂＋手指20维版本 | 人车转向耦合、抓握与行进平衡；双腿由参考控制器驱动 |
| C | 机器人全身35维：手臂14＋手指6＋腿12＋腰3 | 学习双腿与上半身协同，实现主动踩踏与稳定骑行 |

B和C按双腿是否由策略控制划分。全部35维全身版本，包括相位残差、直接腿控、有/无辅助及滚阻速度奖励，都属于C。完整配置与结果索引见[阶段2实验划分](docs/阶段2-算法实现g1骑车/实验划分与结果索引.md)。

C的历史配置名`bicycle_b_whole_body_*`、运行目录`b_whole_body_*`及兼容字段`"experiment": "B"`保留不变；它们不代表研究编号仍是B。

## 当前状态：实验C（2026-11-21）

最新实验C采用35维全身动作、腿部全范围关节位置目标、零曲柄辅助、轮地滚阻和固定2 m/s速度奖励。GPU滑行台架与无连接接触初始化已通过检查；SAC从头训练16,777,216次交互完成。

冻结最终策略评测128个首回合，20秒存活率和速度合格成功率均为0，平均持续11.654秒，最长18.72秒。轨迹显示策略仍抬起双脚，尚未形成持续主动踩踏。没有检测到本轮评测的后端物理异常，不能据此宣布已经学会骑车。

- [最新实验结果与失败分析](docs/阶段2-算法实现g1骑车/滚阻与速度奖励SAC16M结果.md)：训练与评测数据、脚踏证据、失败视频及下一步建议。
- [35维全身控制](docs/阶段2-算法实现g1骑车/无骨盆连接的35维全身控制.md)：动作权限、初始化、奖励、指标与各版命令。
- [轮地滚阻检查](docs/阶段2-算法实现g1骑车/轮地滚阻与无动力滑行检查.md)：CPU/GPU滑行对照与滚阻参数选择。
- [阶段2进度记录](docs/阶段2-算法实现g1骑车/进度记录.md)：按推进顺序保留的实验记录。

SAC折扣调整、骑姿附近的探索初始化及固定车架主动踩踏实验目前是待验证建议，尚未启用。当前SAC折扣因子仍为0.99；不要把候选0.999写成已经完成的改动。

## 环境与复现

在项目根目录使用`uv`管理Python、依赖和`.venv`，依赖以`pyproject.toml`和`uv.lock`为准，不复制其他项目环境，不修改`site-packages`。

```bash
uv python install 3.12
uv sync --locked --extra training
uv run --locked --extra training python -I -m pytest -q
```

训练额外依赖为锁定的skrl 1.4.2。`python -I`用于隔离宿主ROS等Python路径。完整测试最近记录为112项通过、5项CUDA测试因执行环境不可见GPU而跳过；GPU训练、滑行对照和接触初始化的实际运行结果另见对应实验记录。

依赖变更使用`uv add/remove`，常规复现不重建锁文件。GPU训练明确要求可见CUDA，不自动退回CPU。

## 实验C当前设置

最新配置：[bicycle_b_whole_body_direct_noassist_rolling_speed_sac.json](configs/rl/bicycle_b_whole_body_direct_noassist_rolling_speed_sac.json)。

| 项目 | 设置 |
| --- | --- |
| 动作／观测 | 35维／207维 |
| 手臂／手指／腿部／腰部 | 14／6／12／3维，均由策略控制 |
| 腿部动作 | 全范围绝对位置目标，动作0对应关节区间中点；无相位参考跟踪 |
| 手指 | 6个软件协同闭合量控制14个实际手指电机 |
| 支撑与连接 | 实际座位、手把、脚踏接触；无骨盆或掌把equality |
| 外部曲柄辅助 | 0；没有转向执行器或后轮电机 |
| 轮地接触 | 解析椭球轮胎；两条轮地6维接触pair；滚阻b=0.00405 m |
| 初始状态 | 车速2 m/s；使用已验收的GPU无连接接触初始化 |
| 速度奖励 | 目标2 m/s，宽度0.75 m/s；低于1.5 m/s连续扣分 |
| 20秒成功 | 未倒，全程及最后5秒分别至少90%的控制步处于1.5～2.5 m/s |

速度要求用于奖励和成功统计，低速不提前终止。`survival_20s_rate`只表示原始不倒比例，本轮任务成功主要看`speed_qualified_20s_rate`；训练近期指标为`recent_speed_qualified_20s_rate`。即使速度合格，也需要检查脚踏接触和做功，才能判断是否主动蹬踏。

## 评测与回放

最新已完成训练：`runs/b_whole_body_rolling_speed_sac_seed42_16m`。评测从该运行的`run.json`加载配置和参考哈希，默认读取`final.pt`。模型、参考、运行目录及`output/`产物需在本地存在；仅有代码不能还原训练权重。

用新目录复测128个首回合，保存全部GPU轨迹：

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.probe_rider_gpu \
  --controller policy --device cuda:0 --num-envs 128 --steps 1100 \
  --seed 20000 --crank-assistance 0 \
  --source-run runs/b_whole_body_rolling_speed_sac_seed42_16m \
  --first-episodes-only --record-only --record-all-envs \
  --output output/phase2/b_whole_body_rolling_speed_16m_eval_repeat
```

已有评测为`output/phase2/b_whole_body_rolling_speed_16m_eval`。重渲染其中最长的111号失败回合，2倍慢放：

```bash
MUJOCO_GL=egl uv run --locked --extra training python -I \
  -m bicycle_sim.cli.render_rider_probe \
  output/phase2/b_whole_body_rolling_speed_16m_eval \
  --env-id 111 --slow-motion 2 \
  --output output/phase2/b_whole_body_rolling_speed_16m_env111_render_repeat
```

视频写入输出目录的`replay.mp4`，标注脚踏受力与鞋底—踏板间距。渲染只使用保存的GPU姿态，不重新运行CPU动力学。本轮没有成功回合，不能用`--successful`选出样本；后续有成功样本时，该选项会同时检查速度合格条件。

本轮训练复现命令见[全身控制文档](docs/阶段2-算法实现g1骑车/无骨盆连接的35维全身控制.md#本轮训练命令已完成)。原运行目录不会覆盖，重跑必须使用新目录。实验B/C共享的`train_rider`入口目前不支持恢复优化器状态继续训练。

## 车辆设计与物理检查

基础几何以[configs/bicycle_v0.json](configs/bicycle_v0.json)为准。训练适配层根据`configs/rl/`调整座高、轮胎碰撞、接触及执行器设置；各次运行保存自己的配置与场景，不能将基础资产等同于所有训练工况。

| 基础几何 | 数值 |
| --- | --- |
| 轴距／车轮半径 | 0.90 m／0.27 m |
| 基础座面中心（向前、高度） | 0.24 m、0.77 m；当前全身工况座面升高约17.13 mm |
| 握位（向前、左右、高度） | 0.69 m、±0.22 m、1.02 m |
| 曲柄长度／脚踏中心横向位置 | 0.11 m／±0.14 m |
| 转向轴偏竖直／机械拖距 | 约11.36°／0.04 m |
| 车把连接 | 横把直接安装在转向立管上 |
| 传动 | 2:1单向阻尼自由轮；倒踩不驱动，前进滑行不反拖曲柄 |

阶段1基础资产使用192×24采样的单凸圆冠轮胎；当前GPU训练使用解析椭球碰撞胎。非轮部件惯量采用等效近似，轮组暂沿用实心圆盘惯量；质量、摩擦及滚阻尚未经实车辨识。参数与边界见[基础物理检查](docs/阶段1-仿真平台搭建/自行车物理与接触检查.md)、[轮胎基准测试](docs/阶段1-仿真平台搭建/轮胎基准测试.md)和[GPU动力学差异排查](docs/阶段2-算法实现g1骑车/GPU动力学差异排查.md)。

基础资产生成与运动学检查用于阶段1复现，不代表稳定骑行：

```bash
uv run --locked python -I -m bicycle_sim.cli.generate_bicycle
uv run --locked python -I -m bicycle_sim.control.rider_fit
uv run --locked python -I -m bicycle_sim.cli.check_front_clearance
uv run --locked python -I -m bicycle_sim.cli.inspect_platform
MUJOCO_GL=egl uv run --locked python -I -m bicycle_sim.cli.render_three_views
```

## 项目结构与历史实验

- `assets/g1_dex3/`：机器人资产快照、网格、许可证和哈希清单。
- `assets/bicycle/`、`assets/scenes/`：基础自行车及人车组合诊断场景。
- `assets/references/`：抓握和无连接接触初始化参考。
- `configs/bicycle_v0.json`、`configs/rl/`：基础几何及各版独立RL工况。
- `src/bicycle_sim/`：模型生成、控制、强化学习环境和CLI。
- `tests/`、`tools/`：回归检查与离线分析、绘图工具。
- `runs/`、`output/`：本地训练和诊断产物；不把产物是否在Git中当成运行是否完成的依据。
- `docs/`：[阶段1计划](docs/阶段1-仿真平台搭建/阶段1实施计划.md)、[阶段1进度](docs/阶段1-仿真平台搭建/阶段1进度记录.md)、[阶段2进度](docs/阶段2-算法实现g1骑车/进度记录.md)。

实验A是纯自行车速度/世界方向跟踪，采用RSL-RL PPO；实验B先后采用双臂14维控制和自然抓握20维控制；实验C包括后来所有35维全身版本。三者的任务、辅助条件和动作空间不同，结果与检查点不混用。早期方案与结果见[阶段2实施方案](docs/阶段2-算法实现g1骑车/实施方案.md)、[早期B训练说明](docs/阶段2-算法实现g1骑车/B-GPU训练说明.md)、[自然抓握长训练](docs/阶段2-算法实现g1骑车/自然抓握SAC长训练结果.md)。
