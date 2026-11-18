# G1 + Dex3 Bicycle

机器人骑车仿真与可微分世界模型研究。项目仅保留已确认的车辆设计，参数以[configs/bicycle_v0.json](configs/bicycle_v0.json)为唯一来源。

## 当前设计

| 参数 | 数值 |
| --- | --- |
| 轴距／车轮半径 | 0.90 m／0.27 m |
| 座面中心（向前、高度） | 0.24 m、0.77 m |
| 握位（向前、左右、高度） | 0.69 m、±0.22 m、1.02 m |
| 曲柄长度／脚踏中心横向位置 | 0.11 m／±0.14 m |
| 转向轴偏竖直／机械拖距 | 约11.36°／0.04 m |
| 车把连接 | 横把直接安装在转向立管上 |
| 传动 | 后轮与曲柄2:1理想双向耦合，无自由轮 |
| 初始双手 | 掌心向下、手指展开，与握把近零间隙且无穿透 |

车辆几何方案已确认；当前验证范围为单车结构、初步物理响应与人车运动学。承重抓握、动态踩踏、平衡控制和GPU并行验收尚待完成。

## 环境与复现

只使用项目`pyproject.toml`、`uv.lock`和根目录`.venv`；不复制旧项目环境或补丁。首次在另一台机器复现：

```bash
uv python install 3.12
uv sync --locked
```

依赖变更使用`uv add/remove`，常规复现不重建锁文件、不手动pip覆盖。命令使用`python -I`隔离宿主ROS等Python路径。已验证Python 3.12.12、mjlab 1.6.1、MuJoCo及MuJoCo Warp 3.11.0；GPU可用性另行检查。

## 生成与检查

在项目根目录运行：

```bash
uv run --locked python -I -m bicycle_sim.cli.generate_bicycle
uv run --locked python -I -m bicycle_sim.control.rider_fit
uv run --locked python -I -m bicycle_sim.cli.check_front_clearance
uv run --locked python -I -m bicycle_sim.cli.inspect_platform
uv run --locked python -I -m pytest -q
```

查看单车：

```bash
uv run --locked python -I -m mujoco.viewer --mjcf=assets/bicycle/scene_free.xml
```

三视图（需要可用EGL）：

```bash
MUJOCO_GL=egl uv run --locked python -I -m bicycle_sim.cli.render_three_views
```

输出写入被Git忽略的`output/phase1/`。人车场景包含`mounted_kinematic`关键帧，查看骑乘姿态时需加载该帧；它不是动态稳定控制器。

## 目录与边界

- `assets/g1_dex3/`：机器人原始快照、网格、许可证和哈希清单。
- `assets/bicycle/`：同一设计的自由场景及固定架诊断场景，固定架不用于骑行成功评价。
- `assets/scenes/rider_fit_diagnostic.xml`：当前人车组合诊断场景。
- `configs/bicycle_v0.json`：确认方案的唯一参数配置。
- `src/bicycle_sim/`：模型生成、IK、接触检查与CLI。
- `tests/`：结构、物理和初始手部几何回归检查。
- `tools/collect_robot_assets.py`：资产导入工具，后续仿真不依赖旧项目目录。
- `docs/`：研究依据、[实施计划](docs/阶段1实施计划.md)、[当前验证状态](docs/阶段1进度记录.md)。

模型使用192×24采样的单凸圆冠轮胎、非轮部件等效箱体惯量和无自由轮传动；轮组暂沿用实心圆盘惯量，质量与摩擦尚非实车辨识参数。26项测试已通过，但不等于完成动态骑行验收。前部净空通过静态相位/转向网格采样检查，不作为连续扫掠或受载变形的保证。

## 物理与接触检查

详见[自行车物理与接触检查](docs/自行车物理与接触检查.md)。座管端帽已退到座面底部，接触参数已显式分类。已将存在小倾角自稳的解析圆柱替换为圆冠凸包；0.1°微扰下无控制会倾倒。承重骑行、侧倾/侧滑收敛与GPU后端仍待验证。诊断命令：`uv run --locked python -I -m bicycle_sim.cli.audit_physics`。

轮胎比较测试：`uv run --locked python -I -m bicycle_sim.cli.benchmark_tires`。已完成84个CPU滚动工况，结果见[轮胎基准测试](docs/轮胎基准测试.md)。主模型已采用192×24单凸包，其余候选仅在内存生成；GPU吞吐仍待验证。示意与参数见[轮胎方案建议](docs/轮胎方案建议.md)。
