# G1 + Dex3 Bicycle：机器人骑行强化学习与运动数据研究调研

**调研截至：2026 年 11 月 18 日**

## 一、核心结论

查阅现有机器人骑行、两轮车强化学习、人形机器人驾驶以及骑行运动捕捉相关研究后，我认为你的项目有几条值得重点关注的技术路线。

**首先，机器人骑自行车已经不是一个完全没有研究基础的任务。** 特别是 2026 年，RAI Institute、Georgia Tech 等机构已经实现了通过强化学习控制专用自行车机器人完成骑行、跳跃、前后轮平衡甚至翻转等复杂动作。

但是，需要区分两类问题：

1. **专用自行车机器人控制**：机器人的执行机构与自行车高度集成，直接控制转向、驱动或身体配重。
2. **通用人形机器人骑自行车**：人形机器人通过双手握把、双脚踩踏、身体重心调整来控制一辆具有独立动力学的自行车。

后者更接近你的 G1 + Dex3 Bicycle 项目。它需要同时解决自行车平衡、人形机器人全身协调、多点接触和动力传递等问题，训练难度明显更高。

### 我认为最值得阅读的 7 项研究

| 研究 | 年份 | 主要方法 | 对你的价值 |
|---|---|---|---|
| **Scooter-Driving Humanoid Robot** | 2026 | PPO + 全身 IK + Sim2Real | ⭐⭐⭐⭐⭐ 人形机器人通过双手操纵两轮车 |
| **Bicycle Acrobatics with RL** | 2026 | 多种目标条件 RL + 技能组合 | ⭐⭐⭐⭐⭐ 骑行技能的奖励设计与分阶段训练 |
| **Path Following and Stabilization of a Bicycle Model** | 2026 | SAC + Curriculum Learning | ⭐⭐⭐⭐⭐ 纯转向控制如何学会平衡 |
| **LineRides** | 2026 | 空间轨迹引导 RL，无需动捕 | ⭐⭐⭐⭐⭐ 如何摆脱完整参考动作依赖 |
| **Flip Stunts using Iterative Motion Imitation** | 2026 | 迭代模仿 + RL | ⭐⭐⭐⭐ 如何从不准确的运动参考学习 |
| **CycleRL** | 2026 | RL + 域随机化 + Sim2Real | ⭐⭐⭐⭐ 平衡、速度和转向联合奖励 |
| **Balancing Control of a Bicycle-Riding Humanoid Robot** | 2018 | 重心估计 + 模型控制 | ⭐⭐⭐⭐ 人形机器人骑车的经典动力学研究 |

这些工作分别覆盖了人形机器人车辆操作、两轮车动态平衡、奖励设计、参考动作学习和真实环境部署。[MDPI](https://www.mdpi.com/1424-8220/26/19/6324?utm_source=chatgpt.com)

### 对你的项目，初步建议是：

**不要从完整的人形机器人骑行动作模仿开始，而是采用“解析骑行约束 + 分层控制 + RL + 可选的运动先验”的路线。**

理由是：

- 骑行具有明确的运动学约束，不必让 RL 从零发现所有踩踏规律。
- 自行车的平衡与转向高度耦合，适合先学习低维控制。
- 人形机器人具有大量与骑行无关的自由度，直接训练全身动作容易导致探索效率低。
- 完整骑行动捕数据并不包含所需的关节力矩与接触力，直接模仿未必能学会真正骑车。
- 你的研究目标还包括世界模型，所以应该保留能对比 model-free RL、model-based control 和可微分模型学习的实验结构。

下面重点分析已经发表的具体方法，以及如何迁移到你的项目。

---

## 二、现有机器人骑行研究：哪些真正与你的任务接近？

### 2.1 Scooter-Driving Humanoid Robot（2026）

**这是我认为最值得优先研究的一篇。**

论文：[Scooter-Driving Humanoid Robot: Sim-to-Real Transfer Through Deep Reinforcement Learning](https://www.mdpi.com/1424-8220/26/19/6324?utm_source=chatgpt.com)

这篇工作的核心是让一个完整的人形机器人操纵双轮踏板车，而不是直接给车辆安装专门的自动转向控制器。

研究已经在真实双轮车辆上实现了动态平衡与航向控制，甚至展示了载人情况下的控制能力。

它的系统设计非常值得参考：

**RL 不直接输出所有人形机器人关节动作，而是输出转向控制指令，随后通过逆运动学实现双手操纵。**

也就是：

\[
\boxed{
\text{车辆状态}
\xrightarrow{\pi_{\mathrm{RL}}}
\delta_{\mathrm{target}}
\xrightarrow{\mathrm{IK}}
q_{\mathrm{robot,target}}
}
\]

这一架构将问题拆成两个部分：

- RL 负责学习复杂的两轮车动态平衡。
- IK 负责将转向动作转化成人形机器人的关节运动。

这与你当前已经建立的人车 IK 诊断场景非常契合。

但它与自行车骑行还有一个关键区别：**该研究没有让 RL 同时学习双脚踩踏，速度由外部指令给定。** [MDPI](https://www.mdpi.com/1424-8220/26/19/6324?utm_source=chatgpt.com)

因此，你可以借用它的架构，却不能直接将其当作完整骑行问题的解决方案。

### 2.2 Bicycle Acrobatics with Reinforcement Learning（2026）

论文：[Bicycle Acrobatics with Reinforcement Learning](https://borowiecki.dev/abs/2608.00880?utm_source=chatgpt.com)

项目：[Bicycle Acrobatics 项目主页](https://shamelfahmi.com/bicycle_acrobatics/?utm_source=chatgpt.com)

这是自行车 RL 领域相当重要的一项近期工作。

它使用专用的 Ultra Mobility Vehicle（UMV），利用不同形式的 RL 学习：

- 正常骑行与路径跟踪；
- 前轮抬起、后轮平衡；
- 单轮跳跃与双轮跳跃；
- 前空翻等复杂动作；
- 不同技能之间的自动切换。

尤其值得注意的是，作者没有尝试使用一个统一的动作模仿目标解决全部任务，而是采用了多种目标表达：

| 方法 | 学习目标 |
|---|---|
| Waypoint following | 跟踪空间路径 |
| Pose reaching | 到达特定位置和姿态 |
| Twist tracking | 跟踪线速度与角速度 |
| Guided tracking | 沿引导轨迹运动 |
| Motion imitation | 模仿给定参考运动 |

作者还设计了根据状态触发的技能切换机制，使多个控制策略能够组合成长时间的动作序列。[Borowiecki.dev](https://borowiecki.dev/abs/2608.00880?utm_source=chatgpt.com)

**对你最重要的启发是：不应该假定每一种骑行能力都需要完整的动捕动作作为监督。**

例如：

- 保持平衡可以用侧倾与角速度描述。
- 正常行驶可以用速度命令描述。
- 转弯可以用曲率或航向命令描述。
- 踩踏可以用曲柄相位和脚踏接触约束描述。

而完整人体动作参考可以作为辅助先验，而不是所有任务的唯一目标。

### 2.3 Path Following and Stabilization of a Bicycle Model（2026）

论文：[论文全文](https://link.springer.com/article/10.1007/s11044-026-10144-x?utm_source=chatgpt.com)

这篇论文特别适合作为你第一个 RL baseline 的理论参考。

它使用 Whipple 自行车动力学模型，不引入额外的反作用轮或配重装置。

RL 策略只输出转向角指令：

\[
a_t=\delta_t^{\mathrm{target}}
\]

训练任务是同时完成：

1. 自行车保持侧向稳定。
2. 自行车沿目标路径行驶。

作者使用 SAC，并通过课程学习逐步提高任务难度。论文中仿真步长为 0.005 s，RL 控制周期为 0.05 s，即 20 Hz。

一个重要实验结果是：在作者的初步比较中，SAC 明显优于 PPO 和 A2C，因此主要实验采用 SAC。

这不意味着 SAC 在你的 GPU 并行人形机器人环境中一定优于 PPO，但说明**低维连续转向控制值得单独建立 SAC baseline**。[Springer](https://link.springer.com/article/10.1007/s11044-026-10144-x?utm_source=chatgpt.com)

---

## 三、RL 奖励函数：现有研究具体怎么设计？

这是本次调研中最有直接应用价值的部分。

我发现一个很有意思的现象：**成功的两轮车 RL 工作，往往没有使用非常复杂的奖励，而是围绕少数物理上有意义的状态量设计目标。**

### 3.1 人形机器人驾驶踏板车：6 项奖励

2026 年的 Scooter-Driving Humanoid Robot 论文公开了比较完整的奖励函数。

定义：

- \(\varphi\)：车辆侧倾角；
- \(\omega_\varphi\)：侧倾角速度；
- \(\Delta\psi\)：航向跟踪误差；
- \(u\)：策略输出的转向动作。

论文包含六类奖励：

| 奖励 | 数学形式 | 目的 |
|---|---|---|
| 航向跟踪 | \(1-\Delta\psi^2\) | 保证正确行驶方向 |
| 侧倾控制 | \(1-\varphi^2\) | 避免过度倾斜 |
| 侧倾速度 | \(1-\omega_\varphi^2\) | 抑制快速倾倒 |
| 动作幅度 | 与 \(u^2\) 有关的惩罚 | 减少不必要的转向 |
| 动作平滑 | \(1-\dot u^2\) | 避免转向振荡 |
| 对称性 | 约束左右镜像状态的动作 | 利用车辆左右对称结构 |

论文采用的奖励组合为：

\[
\begin{aligned}
R={}&1.0r_\psi
+0.1r_\varphi
+0.35r_\omega\\
&+2.0r_{\mathrm{action}}
+0.2r_{\mathrm{smooth}}\\
&+3.0r_{\mathrm{sym}}
\end{aligned}
\]

其中动作惩罚不是普通的固定二次项，而是：

\[
r_{\mathrm{action}}
=
1-u^2w_\varphi w_\omega
\]

\[
w_\varphi=1-\tanh(50\varphi^2)
\]

\[
w_\omega=1-\tanh(50\omega_\varphi^2)
\]

这意味着：

**当车辆即将倾倒时，动作惩罚自动减弱，允许策略采取较大的紧急转向动作。**

这是一个非常值得借鉴的设计。

因为如果始终惩罚大幅转向，RL 可能学会保持动作平稳，却无法在侧倾扰动出现时及时恢复。

论文还使用了左右镜像状态的策略对称性约束：

\[
u(M(o))\approx-u(o)
\]

这里 \(M\) 表示状态的左右镜像变换。作者认为这可以改善学习效率和策略的一致性。

这些奖励并非经过系统性全局超参数搜索；作者明确说明权重主要通过经验调节获得。[MDPI](https://www.mdpi.com/1424-8220/26/19/6324?utm_source=chatgpt.com)

**对于 G1 Bicycle，我建议优先借鉴三点：**

第一，侧倾角和侧倾角速度同时进入奖励。

第二，紧急恢复时减弱动作正则化。

第三，可以引入左右对称性，但必须同步镜像整个机器人姿态、关节、接触状态和踏板相位，不能只对车辆状态取负。

特别要注意：G1 全身骑行不一定具有与简化两轮车完全相同的镜像对称性，例如左右腿的踩踏相位和接触情况需要一起变换。

### 3.2 纯自行车 RL：甚至不需要直接奖励直立

2026 年的 *Path Following and Stabilization of a Bicycle Model* 使用了一个相当简洁的奖励设计：

\[
\rho=\chi_1\rho_y+\chi_2\rho_\varphi
\]

其中路径跟踪奖励为：

\[
\rho_y=\exp\left(-\frac{1}{2}e_y^2\right)
\]

这里 \(e_y\) 是车轮接地点到参考路径的距离，单位为米，论文中的数值形式相当于使用固定的 1 m 距离尺度。

直立奖励为：

\[
\rho_\varphi=
\frac{\eta_\varphi-|\varphi|}
{\eta_\varphi}
\]

其中：

\[
\eta_\varphi=45^\circ
\]

超过 \(45^\circ\) 即终止。

论文研究了不同的奖励组合，包括：

\[
\chi_1=1,\qquad \chi_2=0
\]

也就是：

\[
\boxed{\rho=\rho_y}
\]

**这意味着在这一实验设置中，可以完全不使用显式的直立奖励，而通过路径跟踪目标和倾倒终止条件引导策略学习平衡。**

当然，这不是说机器人在任何场景下都能只凭路径奖励学会平衡。它依赖路径任务、终止条件以及训练课程的共同作用。

论文还报告，将路径跟踪奖励与侧倾奖励等权组合并不必然改善训练，反而可能增加学习难度。[Springer](https://link.springer.com/article/10.1007/s11044-026-10144-x?utm_source=chatgpt.com)

这对你的研究有一个重要启发：

### 直立是否应该是奖励目标？

在实际骑行中，转弯需要侧倾。

如果你直接奖励：

\[
r_{\mathrm{upright}}
=
\exp(-k\varphi^2)
\]

那么奖励事实上在鼓励：

\[
\varphi\rightarrow0
\]

但在稳态协调转弯的近似条件下：

\[
\tan\varphi_{\mathrm{desired}}
\approx
\frac{v^2\kappa}{g}
\]

其中：

- \(v\)：行驶速度；
- \(\kappa\)：路径曲率；
- \(g\)：重力加速度。

因此，更合理的设计之一是：

\[
r_{\mathrm{lean}}
=
\exp\left[
-k_\varphi
(\varphi-\varphi_{\mathrm{desired}})^2
\right]
\]

但我不建议一开始就强制使用这个参考侧倾角。

原因是这个表达式来自简化的协调转弯模型，忽略了转向瞬态、人形机器人身体运动和复杂接触。

更适合的安排是：

- 初期直线平衡：奖励较小的侧倾角。
- 后期转弯：主要奖励路径跟踪与不倾倒。
- 将理论侧倾角作为可选辅助目标，比较是否真正改善性能。

---

### 3.3 CycleRL：平衡、速度和转向共同学习

论文：[CycleRL: Sim-to-Real Deep Reinforcement Learning for Robust Autonomous Bicycle Control](https://arxiv.org/abs/2603.15013?utm_source=chatgpt.com)

这项工作专注于自行车的鲁棒自主控制，采用复合奖励和域随机化来学习：

\[
\text{Balance}
+
\text{Velocity Tracking}
+
\text{Steering Control}
\]

它比只考虑转向平衡的工作更接近实际骑行。

根据论文摘要，重点包括：

1. 维持两轮车辆动态平衡。
2. 跟踪目标速度。
3. 学习转向控制。
4. 利用域随机化实现仿真到现实的迁移。

不过，我目前没有核实到足以可靠复述的完整奖励权重表，因此不把自行设计的公式冒充论文原始定义。[arXiv](https://arxiv.org/abs/2603.15013?utm_source=chatgpt.com)

对你而言，这篇论文可以用于比较两种策略：

\[
\pi_\delta(s,v_{\mathrm{target}})
\]

以及

\[
\pi_{\delta,\tau}(s,v_{\mathrm{target}})
\]

其中第一种只控制转向，第二种同时控制转向与驱动力矩。

在 G1 Bicycle 上，第二种策略可以通过机器人双腿产生踏板力矩，而不是在车辆上直接添加额外驱动电机。

---

## 四、有没有不依赖骑行动捕的更好方法？

有，而且这可能是整个项目最值得研究的部分。

### 4.1 LineRides：不模仿关节轨迹，只规定空间行为

论文：[LineRides: Line-Guided Reinforcement Learning for Bicycle Robot Stunts](https://seungeunrho.github.io/LineRides/?utm_source=chatgpt.com)

这是 2026 年发表在 IEEE RA-L 的工作。

其出发点与你目前的问题高度相关：

> 对复杂机器人任务，很难获得真实、可行且适配机器人结构的参考运动。

所以作者不要求提供完整动作轨迹，而是提供：

- 一条空间引导线；
- 少量关键姿态；
- 对轨迹偏差的容许范围。

策略在这些宽松约束下自主探索物理上可行的动作。

最有价值的是它的两项设计。

**第一，允许参考路径不完全符合实际动力学。**

例如你画出一条期望路径，但是自行车由于非完整约束，无法严格沿路径执行。

此时不应该强迫：

\[
x_t=x_t^{\mathrm{ref}}
\]

而可以采用容许偏差的目标：

\[
r_{\mathrm{guide}}
=
\exp\left[
-k\max(0,d(x_t,\mathcal L)-\epsilon)^2
\right]
\]

这是我根据论文思想写出的示意公式，不是论文原式。

其中 \(\mathcal L\) 是空间引导轨迹，\(\epsilon\) 是允许误差。

**第二，使用空间进度而不是固定时间进度。**

如果参考动作采用：

\[
q^{\mathrm{ref}}(t)
\]

策略必须在规定时间到达指定状态。

但骑车是一个速度可能变化的过程。

LineRides 使用沿空间路径的运动进度处理时间上的模糊性，使策略不必严格复现参考动作的时间安排。[seungeunrho.github.io](https://seungeunrho.github.io/LineRides/?utm_source=chatgpt.com)

这一思想可以直接迁移到踩踏学习。

### 我的延伸：踩踏可以采用相位约束，而不是完整动捕

你的自行车已经定义：

\[
l_{\mathrm{crank}}=0.11\ \mathrm m
\]

并且具有确定的曲柄传动关系。

因此左右踏板位置可以由曲柄相位 \(\theta_c\) 确定：

\[
p_L^{\mathrm{pedal}}
=
f_L(\theta_c)
\]

\[
p_R^{\mathrm{pedal}}
=
f_R(\theta_c)
\]

机器人需要学习的不是重新发现踏板的几何运动，而是：

\[
\boxed{
\text{如何在满足脚踏运动学约束的同时产生有效驱动力矩}
}
\]

具体可以设计：

\[
r_{\mathrm{foot}}
=
\exp\left(
-\frac{
\|p_L-p_L^{\mathrm{pedal}}\|^2
+
\|p_R-p_R^{\mathrm{pedal}}\|^2
}{\sigma_p^2}
\right)
\]

再加上：

\[
r_{\mathrm{speed}}
=
\exp\left(
-\frac{(v-v_{\mathrm{cmd}})^2}{\sigma_v^2}
\right)
\]

也就是说：

- 踏板相位提供运动学参考。
- 速度目标提供任务监督。
- RL 学习符合动力学的双腿协作。
- 无需先找到一条完全匹配 G1 的人类踩踏轨迹。

这一方案比直接对全身关节轨迹施加 imitation loss 更适合建立第一个可运行的训练系统。

不过，脚踏位置跟踪只能约束运动学，**不能保证脚掌产生正确的切向推力**。后续仍然需要接触力、功率输出或真实前进速度作为任务相关监督。

---

### 4.2 Iterative Motion Imitation：参考动作不准确也可以利用

论文：[Flip Stunts on Bicycle Robots using Iterative Motion Imitation](https://imi-umv.github.io/?utm_source=chatgpt.com)

这项 ICRA 2026 工作解决了一个很实际的问题：

初始参考动作可能不符合真实机器人动力学，甚至包含自碰撞。

作者采用迭代运动模仿：

1. 从初始参考轨迹开始训练策略。
2. 使用策略在物理仿真中产生新的运动。
3. 根据新 rollout 更新模仿参考。
4. 继续训练，使运动逐步变得可执行。

最终，作者在自行车机器人上实现了原始参考动作无法直接完成的翻转任务。[imi-umv.github.io](https://imi-umv.github.io/?utm_source=chatgpt.com)

对你的项目而言，可以先用几何方法生成：

\[
q_{\mathrm{ref}}(\theta_c,\delta)
\]

其中：

- \(\theta_c\)：曲柄相位；
- \(\delta\)：转向角；
- \(q_{\mathrm{ref}}\)：满足手部握持和脚踏位置约束的人形机器人姿态。

这个参考姿态不需要具有正确的动力学。

随后让 RL 在模仿姿态的基础上探索真实动作，再通过物理成功的 rollout 改进参考运动。

**这里最需要注意的是：策略 rollout 不一定比参考动作好。**

因此，不能简单地把所有 rollout 重新加入训练。

应该筛选：

- 任务成功率；
- 前进距离；
- 平衡持续时间；
- 接触约束；
- 机械功率；
- 是否依赖非物理的接触穿透。

---

## 五、骑行动捕数据：有哪些能直接使用？

这一部分的结果比我预期更明确：**存在公开的人类骑行动捕数据，而且包含 BVH 骨架动画，可以用于人形机器人姿态重定向。**

但需要区分“骑行姿态参考”和“动态平衡示范”。

### 5.1 Dataset of Motion Capture of Cyclists（2024）

数据地址：[Zenodo：Dataset of Motion Capture of Cyclists](https://zenodo.org/records/10668611?utm_source=chatgpt.com)

这是目前我最建议下载检查的数据集。

研究使用 Phasespace Impulse X2E 动捕系统，包含 24 台摄像机。

参与者在自行车训练台上进行骑行，并改变踏频和骑行姿势。

**踏频包括：**

\[
60,\ 75,\ 90,\ 100,\ 110\ \mathrm{RPM}
\]

姿态包括：

- Straight Arms
- Comfortable
- Aggressive
- Aero Position
- Standing

数据公开了三种格式：

| 格式 | 内容 | 对 G1 的用途 |
|---|---|---|
| C3D | 原始运动捕捉标记数据 | 重新构建人体运动学 |
| BVH | 人体骨架动画 | 全身关节重定向 |
| GPX | 相关骑行训练记录 | 辅助对齐骑行工况 |

这些信息可以直接从数据集说明中确认。[Zenodo](https://zenodo.org/records/10668611?utm_source=chatgpt.com)

### 能不能用于 G1？

**可以用于姿态先验和踩踏运动参考，但不能直接作为动态平衡控制专家数据。**

主要有三个问题。

**第一，训练台限制了车辆的平衡自由度。**

骑行者不需要像真实道路骑行那样持续通过转向修正车辆侧倾。

因此，这份数据主要适合学习：

\[
q_{\mathrm{body}}(\theta_c,\dot\theta_c)
\]

而不适合直接学习：

\[
a_{\mathrm{balance}}
=
\pi(\varphi,\dot\varphi)
\]

**第二，人类与 G1 的身体结构不一致。**

例如：

- 大腿、小腿长度不同；
- 髋关节自由度不同；
- 手腕活动范围不同；
- 身体重心分布不同。

因此，不能直接复制 BVH 关节角度，必须进行约束重定向。

**第三，人体动捕没有提供真实的机器人关节力矩。**

即使重定向后姿态完全正确，也不意味着机器人能够在踏板上产生有效的驱动力。

我的建议是把它当作一个**自然运动先验**，而不是控制策略的最终监督目标。

### 5.2 其他数据资源

| 数据集 | 内容 | 适用性 |
|---|---|---|
| Dataset of Motion Capture of Cyclists | 骑行人体 BVH/C3D | ⭐⭐⭐⭐⭐ 踩踏与姿态 |
| SFU Motion Capture Database | 各种人体动捕 | ⭐⭐⭐ 需筛选骑行片段 |
| RobotCycle Dataset | 城市真实骑行传感器数据 | ⭐⭐ 车辆状态与环境建模 |
| Cycling World Model Dataset | 自行车第一视角视频 | ⭐⭐ 视觉世界模型 |
| BikeZ-ETH | 多人骑行轨迹 | ⭐⭐ 路径与群体运动 |

其中 RobotCycle 主要用于城市骑行环境感知、轨迹与定位等研究，并不直接提供你需要的机器人全身关节控制数据。[ori-mrg.github.io](https://ori-mrg.github.io/robotcycle-dataset/?utm_source=chatgpt.com)

因此，数据选择上，我会明确优先使用第一项。

### 5.3 如何将人类骑行动捕重定向到 G1？

你已经有 `rider_fit` 和人车组合 IK，这非常适合作为重定向的基础。

我建议不要直接做传统的全身关节角度重定向，而是进行**以车辆为参考坐标系的接触约束重定向**。

优化问题可以写为：

\[
\begin{aligned}
q_t^*=\arg\min_q\quad&
w_p\|p_{\mathrm{pelvis}}(q)-p_{\mathrm{pelvis}}^{ref}\|^2\\
&+w_b\|R_{\mathrm{torso}}(q)-R_{\mathrm{torso}}^{ref}\|^2\\
&+w_q\|q-q_{\mathrm{nominal}}\|^2
\end{aligned}
\]

满足：

\[
p_{\mathrm{hand},L/R}(q)
=
p_{\mathrm{grip},L/R}(\delta)
\]

\[
p_{\mathrm{foot},L/R}(q)
=
p_{\mathrm{pedal},L/R}(\theta_c)
\]

以及：

\[
q_{\min}\le q\le q_{\max}
\]

这里的位置约束应使用车辆坐标系下的相对位置，并允许根据手掌、脚掌的真实接触几何设置合适的容差。

这就把问题从“模仿人的关节运动”，转化成：

**在保持人类骑行姿态特点的同时，满足当前机器人与自行车的几何约束。**

你甚至可以首先不使用动捕，只根据当前车辆模型生成解析的踏板相位轨迹，随后再加入动捕姿态作为优化正则项。

这样可以通过对照实验明确检验：真实人体动捕究竟提供了多少额外价值。

---

## 六、结合当前 G1 + Dex3 项目，推荐怎样训练？

这里我倾向于将系统分成三个控制层次，而不是直接训练一个全身关节策略。

### 6.1 第一层：车辆运动控制

输入：

\[
o_{\mathrm{bike}}
=
[
\varphi,\dot\varphi,
\delta,\dot\delta,
v,\dot\psi,
v_{\mathrm{cmd}},\kappa_{\mathrm{cmd}}
]
\]

输出：

\[
a_{\mathrm{bike}}
=
[
\delta_{\mathrm{cmd}},
\tau_{c,\mathrm{cmd}}
]
\]

分别表示：

- 期望转向角；
- 期望曲柄驱动力矩。

但这里有一点非常重要：

**\(\tau_c\) 只能是传递给全身控制器的期望力矩，不能作为真实骑行评价时绕过机器人、直接施加到曲柄上的外部驱动力。**

否则就不再是在研究 G1 如何踩踏自行车。

建议先将其作为训练阶段的特权接口，研究车辆可控性与低维 RL 收敛。

### 6.2 第二层：人形机器人全身控制

这一层负责将车辆动作转换成机器人动作。

例如：

\[
q_{\mathrm{target}}
=
\mathrm{IK}
(
\delta_{\mathrm{cmd}},
\theta_c,
q_{\mathrm{posture}}
)
\]

但只使用 IK 不足以实现真实骑行。

原因是 IK 只决定姿态，不决定人车之间的作用力。

真正需要控制的是：

\[
\begin{bmatrix}
F_{\mathrm{hand}}\\
F_{\mathrm{foot}}\\
F_{\mathrm{seat}}
\end{bmatrix}
\]

因此，最终的全身控制器还需要具备以下能力：

- 维持双手与车把的稳定接触；
- 在踏板上产生切向推力；
- 利用身体重心移动辅助平衡；
- 保持座面支撑；
- 避免自碰撞和接触穿透。

这里可以选择 WBC/QP，也可以采用低层 RL 策略。

对于你当前的研究，我更倾向于：

\[
\boxed{
\text{高层 RL}
\rightarrow
\text{几何参考 + 任务空间目标}
\rightarrow
\text{低层 RL/WBC}
}
\]

原因是接触动力学较复杂，而解析运动学约束已经明确。

### 6.3 第三层：残差动作学习

为了避免 IK 过分限制机器人动作，可以采用 residual control：

\[
a_t
=
a_{\mathrm{nominal}}(s_t,c_t)
+
\Delta a_{\mathrm{RL}}(s_t,c_t)
\]

其中：

- \(a_{\mathrm{nominal}}\)：IK 或 WBC 产生的参考动作；
- \(\Delta a_{\mathrm{RL}}\)：策略学习的动作修正。

例如机器人骑车时，理论踩踏动作会提供：

\[
q_{\mathrm{leg}}^{ref}(\theta_c)
\]

而 RL 学习：

\[
\Delta q_{\mathrm{leg}}
=
\pi_{\mathrm{RL}}(s)
\]

最终执行：

\[
q_{\mathrm{leg}}^{target}
=
q_{\mathrm{leg}}^{ref}(\theta_c)
+
\Delta q_{\mathrm{leg}}
\]

这里需要注意，位置残差不等于有效踏板力矩；如果采用位置控制，还要检查关节 PD 闭环能否产生正确的驱动力，并加入力矩限制。

这一方案的一个好处是，**从一开始就赋予机器人基本的人车运动学结构，RL 只需学习实际动力学和接触引起的修正。**

---

## 七、我建议的 RL 奖励：先简单，后完整

不建议第一版就把十几种奖励全部加进去。

建议分三个实验。

### 实验 A：只学习转向平衡

目标：验证当前自行车模型可以通过转向实现动态平衡。

首先不涉及 G1，使用车辆特权转向控制。

\[
a_t=\delta_{\mathrm{cmd}}
\]

假设使用外部设定的前进速度 \(v_0\)，奖励可以是：

\[
r_t
=
w_1e^{-k_\varphi\varphi^2}
+
w_2e^{-k_\omega\dot\varphi^2}
-
w_3(\Delta\delta)^2
\]

终止条件：

\[
|\varphi|>\varphi_{\max}
\]

建议在多个速度下训练，而不是只考虑单一速度。

这个实验的主要价值不是获得最终骑行策略，而是验证：

**当前轮胎、转向轴、拖距和接触模型所表现出来的侧向动力学，是否适合后续 RL 学习。**

尤其要检查随机侧倾扰动下的恢复能力。

### 实验 B：双腿踩踏

这里可以使用固定架诊断场景，但不能将固定架结果计为完整骑行成功。

输入包括：

\[
o_t=
[
q,\dot q,
\theta_c,\dot\theta_c,
v_{\mathrm{cmd}}
]
\]

奖励：

\[
\begin{aligned}
r_t={}&
w_v\exp\left[-k_v(v-v_{\mathrm{cmd}})^2\right]\\
&+w_p r_{\mathrm{foot}}\\
&-w_\tau\|\tau\|^2\\
&-w_s\|\Delta a\|^2
\end{aligned}
\]

如果固定架阻止车辆平移，则不能直接使用 \(v\) 作为观测速度，应改用后轮角速度对应的等效滚动速度作为诊断指标。

建议让踩踏任务先学会：

1. 正向持续踩踏；
2. 改变目标速度；
3. 在不同曲柄初始相位下启动；
4. 在一定负载阻力下持续输出有效功率。

重点是**真实由脚部接触驱动曲柄**，而不是依赖附加驱动器。

### 实验 C：完整的人形机器人骑行

将前两个实验结合，策略同时处理转向、踩踏和身体协调。

我建议的初始奖励为：

\[
\boxed{
\begin{aligned}
r_t={}&
w_v r_{\mathrm{velocity}}
+w_b r_{\mathrm{balance}}\\
&+w_p r_{\mathrm{pedal}}
+w_h r_{\mathrm{hand}}\\
&-w_a c_{\mathrm{action}}
-w_e c_{\mathrm{effort}}
\end{aligned}
}
\]

对应：

| 项目 | 作用 |
|---|---|
| Velocity | 推动机器人持续骑行 |
| Balance | 避免快速倾倒 |
| Pedal | 维持正确踩踏接触与有效驱动 |
| Hand | 保持握把接触 |
| Action | 避免高频振荡 |
| Effort | 降低不必要的能量消耗 |

我的建议是**不要显式奖励机器人保持与初始坐姿完全一致**。

因为真实骑行中，机器人可能需要通过髋关节、肩关节和躯干侧倾来维持平衡。

如果强制姿态跟踪太严格，可能抑制这种运动能力。

同样，不应简单地在所有阶段奖励“接触力越大越好”，因为机器人可能通过过大的手部拉力、脚踏法向压力或座面挤压来钻奖励漏洞。

---

## 八、你的第二个研究目标：可微分世界模型应该怎么做？

我认为这里比“让 G1 学会骑车”本身更有研究价值。

因为你实际上可以研究：

> **机器人是否能够通过学习自行车动力学，将一个任务中获得的平衡能力迁移到另一个任务，并减少从零学习新技能所需的交互数据？**

这与你之前关注的“机器人能否像人一样快速学会骑车”直接相关。

但我建议先区分两种不同的世界模型。

### 8.1 解析动力学模型与学习动力学模型

第一种是基于物理方程的模型。

自行车在直立状态附近，可以采用 Whipple 模型描述：

\[
M\ddot q+vC_1\dot q+(gK_0+v^2K_2)q=f
\]

其中：

\[
q=
\begin{bmatrix}
\varphi\\
\delta
\end{bmatrix}
\]

\(f\) 包括侧倾和转向方向的广义力矩。

该模型的意义是能够解释为什么：

- 自行车平衡与速度有关；
- 转向可以改变侧倾动力学；
- 部分自行车在一定速度区间内具有自稳定性；
- 低速骑行和高速骑行需要不同的控制行为。

它可以作为一个低维、可微分的动力学基线。

第二种是通过交互学习状态转移：

\[
s_{t+1}=f_\theta(s_t,a_t)
\]

其价值在于能够表示实际复杂系统中难以精确建模的因素，例如：

- 轮胎侧滑；
- 机器人身体产生的附加惯性；
- 手部与车把的柔顺接触；
- 脚踏和脚掌之间的力传递；
- 电机关节控制延迟。

对于 G1，我建议进一步采用：

\[
\boxed{
s_{t+1}
=
f_{\mathrm{physics}}(s_t,a_t)
+
f_\theta(s_t,a_t)
}
\]

即**物理先验 + 学习残差动力学**。

不过，状态中包含位置和姿态时，应当在合适的局部坐标系或状态流形中表示残差，不能不加区分地对所有状态直接相加。

### 8.2 不要一开始就预测整个 G1 + Bicycle 的状态

如果完整状态包含：

\[
s=
[
q_{\mathrm{G1}},
\dot q_{\mathrm{G1}},
q_{\mathrm{bike}},
\dot q_{\mathrm{bike}},
\text{contacts}
]
\]

那么学习模型必须处理：

- 高维关节动力学；
- 碰撞和接触切换；
- 摩擦；
- 人车闭链运动学；
- 滚动与侧滑；
- 不同接触模式之间的不连续性。

这会显著增加建模难度。

对于第一版，我更推荐**控制相关的低维世界模型**：

\[
z_t=
[
\varphi,\dot\varphi,
\delta,\dot\delta,
v,\dot\psi,
\theta_c,\dot\theta_c
]
\]

并学习：

\[
z_{t+1}=f_\theta(z_t,u_t)
\]

其中 \(u_t\) 可以是低维的转向、踩踏和身体重心控制目标。

这里存在一个前提：低维 \(z_t\) 可能并不是马尔可夫状态。双手、双脚接触与身体运动会影响未来车辆动态。

因此可以比较：

\[
f_\theta(z_t,u_t)
\]

和

\[
f_\theta(z_{t-k:t},u_{t-k:t})
\]

观察加入历史以后，多步预测与下游控制是否改善。

这个实验也能回答一个很有研究价值的问题：

**骑行动力学中，学习到的隐状态能否捕获车辆运动状态之外的有效信息，例如接触状态和未知惯性？**

---

### 8.3 MuJoCo Warp 与可微分性需要特别区分

你当前的技术栈使用 MuJoCo、MuJoCo Warp 和 mjlab。

这里需要注意：

**GPU 并行物理仿真，不等于端到端可微分物理仿真。**

MuJoCo 官方文档明确区分了：

- **MuJoCo Warp**：主要面向 GPU 并行物理计算和高吞吐 RL。
- **MuJoCo MJX**：基于 JAX，动力学计算 API 支持自动微分。

因此，你目前使用的 MuJoCo Warp 可以作为：

\[
\text{GPU Simulation}
\rightarrow
\text{Rollout Data}
\rightarrow
\text{Learned World Model}
\]

的数据来源。

但不能仅因为它运行在 GPU 上，就假设可以直接通过完整物理仿真对策略求梯度。[MuJoCo Documentation](https://mujoco.readthedocs.io/en/latest/mjwarp/?utm_source=chatgpt.com)

如果要研究可微分世界模型，我建议比较三条路线：

| 方法 | 是否需要学习动力学 | 可微分部分 | 建议 |
|---|---|---|---|
| PPO + MuJoCo Warp | 否 | 策略网络 | 第一阶段基线 |
| Learned World Model + MPC | 是 | 学习的动力学模型 | 优先研究 |
| MJX + Differentiable Control | 不一定 | 支持的仿真计算图 | 后续对照 |

尤其是第三条路线，需要先验证轮胎接触、摩擦和人车闭链接触在当前模型上的兼容性和梯度质量。

即使 MJX 提供自动微分，碰撞接触引起的非光滑性也可能导致梯度不稳定。

---

## 九、2026 年最新的一项研究也值得关注

**Autonomous Trail Racing with a Bicycle Robot**

发布日期：2026 年 11 月 14 日。

论文：[Autonomous Trail Racing with a Bicycle Robot](https://arxiv.org/abs/2611.15280?utm_source=chatgpt.com)

这是本次调研中比较新的工作。

RAI Institute 等机构使用 UMV 自行车机器人，实现了真实山地道路上的高速自主行驶。

公开结果包括：

- 在约 1 km 的非结构化山地道路上自主骑行；
- 最高速度超过 5 m/s；
- 平均速度约 3.26 m/s；
- 融合视觉、LiDAR 和地形语义信息；
- 公开仿真和感知工具。

它的一个重要研究观点是：

**不一定需要将运动控制和路径导航完全分开。**

作者提出把导航目标编码为地形语义层，在统一的 RL 框架中优化局部导航和运动控制。[RAI Institute](https://rai-inst.com/resources/papers/autonomous-trail-racing-with-a-bicycle-robot/?utm_source=chatgpt.com)

不过，对你的当前项目而言，我并不建议直接采用这种端到端方案。

原因是你的主要难点目前不是路径规划，而是人车接触与动力学协调。

等到 G1 能够稳定完成直线骑行后，再研究端到端转弯和路径跟踪会更合适。

---

## 十、推荐的实验路线

结合当前阶段已经完成的车辆几何、初步物理检查和骑乘 IK，我建议按如下顺序推进。

| 阶段 | 目标 | 方法 | 验收依据 |
|---|---|---|---|
| 1 | 自行车动力学基准 | 扰动响应、转向与速度扫参 | 与简化动力学的定性趋势一致 |
| 2 | 自行车 RL 平衡 | SAC/PPO，仅控制转向 | 多速度和扰动下保持平衡 |
| 3 | G1 踩踏 | 曲柄相位 IK + RL/WBC | 由真实脚踏接触持续驱动 |
| 4 | G1 双手握把 | 接触控制 + IK/WBC | 能承重和施加有效转向力矩 |
| 5 | G1 完整骑行 | 分层 RL + 残差控制 | 无固定架、无外部驱动稳定骑行 |
| 6 | World Model | 低维动力学预测 + MPC | 减少真实仿真交互样本 |
| 7 | 迁移学习 | 不同速度、轮胎、质量等 | 量化新任务适应速度 |

这里有两个需要提前处理的问题。

### 10.1 首先验证车轮与地面的侧向响应

你已经完成了 84 个 CPU 滚动工况测试，并把圆柱轮胎替换为单凸圆冠轮胎。

这是合理的结构验证进展。

但之后最需要检查的不是更多静态几何，而是：

\[
\frac{\partial\dot\varphi}{\partial\delta},
\qquad
\frac{\partial\dot\psi}{\partial\delta},
\qquad
\frac{\partial F_y}{\partial v_y}
\]

即：

- 转向是否能产生符合预期的侧倾响应；
- 转向是否能改变车辆航向；
- 轮胎是否提供合理的横向摩擦约束。

这里的偏导数是动力学分析目标，不要求直接通过接触仿真自动微分得到。

你可以利用有限差分和轨迹响应来估计。

更重要的是，你当前的 192×24 凸包轮胎只能表示某种接触几何，并不自动等价于具有真实侧偏刚度、接触斑和轮胎变形特性的自行车轮胎。

因此，**骑行成功率必须在多个接触与摩擦参数下进行鲁棒性评估**，否则 RL 可能利用数值接触模型中特有的行为。

### 10.2 你当前的 2:1 固定传动对训练有影响

项目中的后轮与曲柄采用理想双向耦合，无自由轮。

假定传动关系为：

\[
\omega_{\mathrm{rear}}=2\omega_c
\]

在纯滚动条件下：

\[
v=r_w\omega_{\mathrm{rear}}
\]

因此：

\[
v=2r_w\omega_c
\]

代入：

\[
r_w=0.27\ \mathrm m
\]

得到：

\[
v=0.54\omega_c
\]

这意味着当自行车以 2 m/s 行驶时：

\[
\omega_c\approx3.70\ \mathrm{rad/s}
\]

对应踏频约为：

\[
35.4\ \mathrm{RPM}
\]

这对你的训练有一个直接影响：

**当前固定传动使车速与踏频高度耦合。**

所以不能像具有自由轮的真实自行车一样，让机器人停止踩踏但车辆继续自由滑行。

停止踩踏时，运动中的后轮仍会通过传动机构驱动曲柄旋转，机器人必须能够被动跟随，或者主动控制接触力。

这不妨碍研究骑行，但需要在设计训练任务时明确这一点。

否则 RL 可能学出不符合预期的行为，例如在低速情况下通过强力阻挡曲柄来制动，或者利用不真实的踏板接触方式调节速度。

---

## 十一、真正值得研究的科学问题

最后我想强调：如果项目的目标不仅是完成骑行演示，还包括具身学习与可微分世界模型，那么可以利用这个平台开展几个比较有价值的研究。

### 方向 A：骑车能力是否可以通过其他技能迁移？

可以比较三种初始化：

\[
\pi_{\mathrm{scratch}}
\]

从零训练骑行。

\[
\pi_{\mathrm{locomotion}}
\]

由 G1 已学会的行走、平衡或扰动恢复策略初始化。

\[
\pi_{\mathrm{bicycle}}
\]

由简化车辆控制任务迁移。

重点不只比较最终成功率，还要比较：

\[
N_{\mathrm{samples}}
\]

即达到同样骑行成功率所需要的交互数据量。

你之前关心的“走路和平衡能力能否帮助机器人快速学会骑车”，在这里可以形成一个非常明确的受控实验。

但需要注意，**步行中的质心控制经验不一定能直接迁移到骑行**。

因为步行依赖脚掌与地面接触，而骑行主要依赖车轮非完整约束、转向和人车之间的力传递。

因此，应进一步研究到底是哪些状态表征或动力学知识能够迁移。

### 方向 B：世界模型能否帮助学习新的骑行任务？

这是我认为更适合持续深入的主线。

可以比较：

| 方法 | 训练特点 | 研究目的 |
|---|---|---|
| PPO / SAC | 无显式世界模型 | Model-free baseline |
| RL + Dynamics Auxiliary Loss | 学习世界模型但不用于规划 | 判断预测监督是否改善策略学习 |
| World Model + MPC | 利用预测进行动作选择 | 检验规划收益 |
| World Model + Dyna | 使用模型产生额外训练数据 | 检验样本效率 |

这个实验能够避免一个常见问题：

> 引入世界模型以后，策略表现更好了，但无法确定是世界模型本身提供了有用的动力学知识，还是因为增加了网络容量、训练数据或辅助损失。

因此应该统一网络容量、真实交互预算和控制频率，至少对比：

\[
\text{RL}
\]

\[
\text{RL + WM Prediction}
\]

\[
\text{RL + WM Planning}
\]

并分别评价：

- 世界模型一步与多步预测精度；
- 骑行成功率；
- 扰动恢复能力；
- 新速度与新质量分布的泛化；
- 达到指定成功率所需的交互样本数。

这样才能直接回答：**学习世界模型究竟有没有帮助机器人更快学会骑车？**

### 方向 C：骑行技能能否从低维任务学习到全身控制？

这是更偏控制与机器人学习的方向。

例如先学习：

\[
\pi_{\mathrm{bike}}
:
s_{\mathrm{bike}}
\rightarrow
(\delta,\tau_c)
\]

再通过 G1 的身体控制能力实现：

\[
(\delta,\tau_c)
\rightarrow
a_{\mathrm{G1}}
\]

比较：

- 从零训练全身策略；
- 分层控制；
- 预训练车辆控制后再训练全身策略；
- 基于世界模型进行分层控制。

这有助于研究：**任务动力学知识与具体机器人身体结构之间，哪些部分可以分离，哪些部分必须联合学习？**

---

## 十二、我的最终建议

如果现在只选择三件事推进，我建议：

**第一，优先建立纯自行车 RL 平衡 baseline。**

不要让高维 G1 控制问题干扰对车辆动力学的判断。先验证你的自行车模型能够通过转向实现合理的动态平衡。

**第二，下载 2024 年的 Cyclists Motion Capture 数据集，但先不把它作为 RL 训练必需品。**

优先通过解析曲柄相位和 IK 生成 G1 的踩踏姿态。之后再研究动捕是否能够改善动作自然性、能量效率和训练速度。

**第三，明确将世界模型研究定位为“提升新骑行技能的样本效率”，而不是单纯追求预测准确率。**

例如先让 G1 学会固定速度直线骑行，再要求它适应不同速度、不同转向曲率、不同扰动强度。

比较是否可以通过已经学到的动力学模型，更快完成新任务。

这条研究路线既可以产生可运行的骑行系统，也有希望形成独立的学习算法研究，而不只是一次针对特定机器人的运动控制工程实现。

### 建议优先阅读顺序

1. [Scooter-Driving Humanoid Robot — 人形机器人控制架构与奖励](https://www.mdpi.com/1424-8220/26/19/6324?utm_source=chatgpt.com)
2. [Path Following and Stabilization — 纯转向 RL、SAC 与课程学习](https://link.springer.com/article/10.1007/s11044-026-10144-x?utm_source=chatgpt.com)
3. [Bicycle Acrobatics — 不同骑行技能的 RL 设计](https://arxiv.org/abs/2608.00880?utm_source=chatgpt.com)
4. [LineRides — 不依赖动捕的空间引导学习](https://seungeunrho.github.io/LineRides/?utm_source=chatgpt.com)
5. [Cyclists Motion Capture — 可用于重定向的人类骑行数据](https://zenodo.org/records/10668611?utm_source=chatgpt.com)
6. [Open-DiffLoco — 基于 MJX 和 SHAC 的可微分物理策略学习](https://arxiv.org/abs/2608.02069?utm_source=chatgpt.com)

**其中，我最建议你深入复现的是前两项的控制思想，再把它们整合到 G1 + Dex3 的人车系统中。** 这样既能获得明确的奖励设计依据，也能在加入复杂全身接触之前，知道真正的学习瓶颈来自哪里。