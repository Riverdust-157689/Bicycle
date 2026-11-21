# B：先训练人车耦合的行进平衡

本文属于实验B，只由策略控制上半身；后续35维全身控制属于实验C。 实验编号及历史文件名对应见[实验划分与结果索引](实验划分与结果索引.md)。

> 历史版本说明（2026-11-21更新）：本文记录早期14维双臂控制及显式辅助，不代表当前35维全身、真实抓握、无连接、零曲柄辅助的训练工况。最新GPU训练结果与速度指标见[滚阻与速度奖励SAC16M结果](滚阻与速度奖励SAC16M结果.md)，当前实现见[35维全身控制](无骨盆连接的35维全身控制.md)。下文“当前”“尚待验证”等描述以各历史实验当时状态为准。

> 首轮PPO已经完成但未收敛，动作采样模式已修正。开始新训练前请先按[PPO首轮失败诊断](PPO首轮失败诊断.md)运行GPU控制器探针与短预算复验，不直接重跑完整预算。

> 2026-11-19：已修正B足底中央缺少碰撞面及纯位置随动的阻尼拖拽，加入参考速度PD。腰部仍由固定目标PD控制，手指仍维持初始张开手型，不由策略控制。台架数据和控制权限见[手腰与脚踏控制检查](手腰与脚踏控制检查.md)。

当前先回答“机器人能否通过手臂转向让整车持续不倒”。不要求跟踪指定世界方向或精确速度，不训练零速起步和静止定车。A 的目标跟踪配置和已有检查点保留，B 是独立任务，不能将 A 的检查点直接加载到 B 的不同维度网络上。

## 静止检查与 A 学得快的原因

运行 `uv run --locked python -I -m bicycle_sim.cli.audit_balance` 可复现裸车检查，结果为 `output/phase2/balance_audit.json`。使用当前主模型，自由车架，没有固定支架或平衡控制，倾倒阈值为侧倾 0.65 rad：

| 初始侧倾 | 静止、无动力时到倾倒阈值 |
| --- | --- |
| 0° | 3.926 s |
| 0.01° | 2.130 s |
| 0.1° | 1.582 s |
| 1° | 1.034 s |

保持曲柄零速、车把零角度的伺服也会倒：完全直立约 4.296 s，0.1° 约 1.582 s。初始完全直立只是在接近不稳定对称平衡点附近多停留一会儿，不是稳定平衡。以上结果没有求解器警告。

无动力、初速 5 m/s 时可滑行约 11–12 s 才倒；但把车把目标固定在零角度，即使初速为 5 m/s 也会约 1.2–1.7 s 倒下。这说明当前车模有显著的速度和被动转向效应，也说明“高速滑行暂时不倒”不能直接作为策略学会平衡的证据。应同时比较无动力、固定动作和策略闭环，不能只看视频。

A 使用裸车、平地、两维动作、有限力矩伺服，从 2 m/s 行进状态开始；初始侧倾仅 ±0.02 rad，没有推扰、抓握或全身探索。1000 次更新是 `1000×16×64=1,024,000` 次控制步交互，约 1024 万个物理子步，不是 1000 条数据。这样的低维任务较快出现持续骑行是合理的，但不证明零速平衡、鲁棒性、模型完全真实或 G1 已经学会骑车。此前的高速策略也确实暴露了奖励偏差。

B 的原生 CPU 诊断中，启用局部座位／手把连接、双臂固定名义目标、腿部跟随相位，初速 0/2 m/s 分别约 2.522/2.012 s 后超过整体倾角 0.65 rad，无求解器警告。局部辅助连接没有把整车固定到世界。

## B 首版具体做什么

- **动力学**：同一 G1＋Dex3 和自行车，约 43.394 kg；车架与机器人均保留自由根关节，使用 mjlab 的 `Simulation` 调用 MuJoCo Warp 批量物理。
- **策略动作**：双臂 14 个关节的位置目标偏移，范围 ±0.25 rad，目标变化限速 2 rad/s；实际由原有有限关节力矩 PD 执行。没有直接转向执行器，没有覆盖转向 qpos/qvel，没有世界平衡力。
- **隔离问题的辅助**：骨盆与车架局部座位焊接；两掌与握把局部理想点连接；曲柄速度伺服（可驱动或制动）；腿部按真实曲柄相位插值 IK 参考并由带参考速度的 PD 跟随；B适配层增加连续足底碰撞面。辅助连接允许传递理想反力，当前没有真实抓握的力上限，不能称为 Dex3 自主抓握。
- **初始条件**：前进 2 m/s，曲柄参考约 3.704 rad/s，随机世界航向与小初始侧倾，无运行中推扰。
- **观测**：130 维，含车体系重力、速度、角速度，43 个机器人关节的位置偏移和速度，转向角／角速度，曲柄相位，轮速，上一步动作和转向执行链的手臂目标偏移。目标限速状态可见。
- **奖励**：行进中的存活与较小整体倾角，轻量角速度、动作幅度／变化惩罚；静止无行进存活奖励。无世界航向目标、无精确速度跟踪。当前允许转圈，不能把生存率当作直线骑行成功率。
- **失败**：整体倾角超过 0.65 rad 或非轮胎部件触地；接触按物理子步累计。20 s 为时间截断。非有限状态或 Warp 容量溢出报错，不用复位掩盖。

B 从头训练上肢策略；不直接迁移裸车 A 的权重，不添加教师／学生、动捕或世界模型。初期使用理想局部连接先验证耦合，之后再逐项撤除辅助。

## 环境与运行命令

本轮没有替你下载库。PPO/SAC 共同使用成熟的 **skrl 1.4.2** 实现；SAC 策略使用重参数化 tanh Gaussian 及其 log-prob Jacobian，算法更新由 skrl 完成。新增依赖通过 uv 声明为可选 extra；`uv add` 同步更新项目和锁文件，后续命令始终携带 `--extra training`，避免 uv 同步时移除可选训练依赖。

在项目根目录执行一次：

```bash
uv add --optional training 'skrl==1.4.2'
uv sync --locked --extra training
uv run --locked --extra training python -I -m bicycle_sim.control.rider_fit
uv run --locked --extra training python -I -m pytest -q
```

最后一条会额外执行可选的 PPO、SAC **CPU 网络更新**测试。没有安装 skrl 时该项跳过，当前已安装后的结果为 **49 passed**，含两个算法的CPU更新测试；未安装时该项仍会跳过。`rider_fit` 会生成相位参考，训练从该文件加载并记录哈希；它只准备参考，不代表动态成功。

如宿主缓存路径不可写，可在运行前设置项目内缓存（这些目录已被 Git 忽略）：

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export WARP_CACHE_PATH="$PWD/.cache/warp"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
```

先确认驱动与 PyTorch 都看得见 GPU：

```bash
nvidia-smi
uv run --locked --extra training python -I -c \
  "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

再分别做 GPU 冒烟检查。每条会检查批量物理、非有限状态／容量、局部复位不影响其他环境，并要求策略参数确实更新。**第一次会编译 Warp 内核，可能明显较慢**。输出目录必须是新目录。

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.train_rider \
  --algorithm ppo --device cuda:0 --smoke --output output/phase2/b_ppo_smoke

uv run --locked --extra training python -I -m bicycle_sim.cli.train_rider \
  --algorithm sac --device cuda:0 --smoke --output output/phase2/b_sac_smoke
```

两项都通过后，按相同 128 环境、种子和约 839 万控制步交互训练，先串行运行以免两个任务争用显存：

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.train_rider \
  --algorithm ppo --device cuda:0 --num-envs 128 --seed 42 \
  --transitions 8388608 --output runs/b_ppo_seed42

uv run --locked --extra training python -I -m bicycle_sim.cli.train_rider \
  --algorithm sac --device cuda:0 --num-envs 128 --seed 42 \
  --transitions 8388608 --output runs/b_sac_seed42
```

显存不足时两种算法都改为 `--num-envs 32` 并使用新目录，保留同一交互预算。SAC 的 replay 容量按总 transition 数限制，默认约 26 万条，不随环境数意外变成几千万条。PPO 预算向整 rollout 取整，并在 run.json 记录实际交互数。相同交互预算不代表相同优化步数或墙钟时间。

使用最终检查点做无更新的独立评估（新种子和新目录；对应算法须一致）：

```bash
uv run --locked --extra training python -I -m bicycle_sim.cli.train_rider \
  --algorithm ppo --device cuda:0 --num-envs 32 --seed 10000 \
  --transitions 65536 --evaluate runs/b_ppo_seed42/final.pt \
  --output runs/b_ppo_eval

uv run --locked --extra training python -I -m bicycle_sim.cli.train_rider \
  --algorithm sac --device cuda:0 --num-envs 32 --seed 10000 \
  --transitions 65536 --evaluate runs/b_sac_seed42/final.pt \
  --output runs/b_sac_eval
```

评估使用策略均值，未完成回合数量单独列出，不加入已完成回合分母。这是固定预算的初步评价；正式比较应扩大评估预算并运行至少 3 个训练种子。保留 `metrics.jsonl` 中的持续时间、生存率、回报及墙钟时间，不能只比较训练总回报。

每个目录保存 `run.json`、已编译 `scene.xml`、`metrics.jsonl`、skrl 本地日志和 `final.pt`。运行配置明确记录辅助模式、原始参考／代码／依赖锁哈希。训练前检查 CUDA，不会静默退回 CPU，也不会用 CPU 物理加 GPU 网络冒充 GPU 仿真。

## 当前验证边界

已通过模型编译、CPU 原生短动力学、相位插值和奖励数学测试，以及 Warp 主机侧模型／数据转换；旧阶段测试仍通过。Warp 转换提示部分 capsule/mesh、cylinder 等碰撞对最多生成一个接触点，CPU/GPU 接触一致性仍须单独验收。当前会话 `nvidia-smi` 失败、PyTorch 无 CUDA，skrl的PPO/SAC CPU更新测试已通过，但 **B 的 GPU 执行和GPU更新尚未在这里验证，更没有训练成功结果**。请先运行上面的 CPU 测试和 GPU 冒烟命令，不能跳过冒烟检查直接把入口视为已验收。

采用的接口依据：[mjlab Simulation](https://github.com/mujocolab/mjlab)、[MuJoCo Warp](https://mujoco.readthedocs.io/en/stable/mjwarp/)、[skrl PPO 1.4.2](https://skrl.readthedocs.io/en/1.4.2/api/agents/ppo.html)、[skrl SAC 1.4.2](https://skrl.readthedocs.io/en/1.4.2/api/agents/sac.html)。具体物理接口同时核对了项目当前锁定安装版本；没有修改 site-packages。


## CUDA stream 初始化修正（2026-11-19）

第一次GPU自检报告`Warp error: unknown stream`，位置在`Simulation.create_graph()`的`wp.ScopedCapture`。旧入口两处把由PyTorch当前stream转换出的外部stream设为Warp当前stream。现改为`shared_cuda_stream`：由Warp创建并持有原生stream，通过官方`wp.stream_to_torch`让PyTorch使用同一stream；仿真构建、张量读写及优化器更新全部处于该作用域，退出前同步完成。不关闭CUDA图捕获，不修改site-packages或依赖版本。

新增`tests/test_gpu_stream.py`检查原生stream上的mjlab构建／图捕获与Torch写入后的物理响应。本会话没有可见GPU，当前49项通过、1项CUDA回归测试跳过；需要在实际GPU上重跑自检，不能将该修正视为已完成GPU验收。日志中的TorchScript弃用提示和部分碰撞对单接触点提示不是这次异常的触发位置。


当前控制范围与SAC试验命令见[GPU执行证据与扩大控制范围](侧倾反馈与双臂IK回放.md#gpu执行证据与扩大控制范围当前配置)。GPU逐帧数据已经确认人工控制器在工作，未达到20秒不作为尝试SAC的阻断条件。默认双臂目标偏移为±0.50 rad、目标变化速度4 rad/s，人工探针转向参考为±0.35 rad；策略仍直接输出14维双臂动作。
