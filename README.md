# 面向可解释自动驾驶规划的神经回路策略拓扑优化

东北大学机器学习课程大作业

## 研究背景

自动驾驶规划系统正从模块化流水线向端到端学习演进。UniAD、DriveTransformer 等端到端 Transformer 模型在 nuScenes 等基准上取得了领先性能，但面临三大挑战：

1. **计算效率**：端到端大模型参数量巨大，难以在车端边缘设备上实时部署（L2 级系统要求延迟 20-100ms）
2. **可解释性与合规**：欧盟 AI Act（2026 年 8 月生效）要求高风险 AI 系统具备可追溯性和可解释性，当前黑箱模型无法满足
3. **长尾泛化**：模型在分布外罕见场景下表现不稳定

与此同时，生物启发的液态神经网络（Liquid Neural Networks）展示了独特潜力：Neural Circuit Policies（NCP）仅用 **19 个神经元、253 个突触** 即可完成自动驾驶车道保持，参数量比 LSTM 少数个数量级，且注意力自动聚焦道路关键区域，具备内在可解释性。然而，NCP 的四层布线拓扑（sensory→inter→command→motor）目前完全依赖手工设计，缺乏系统化的优化方法。

## 核心挑战

**Challenge 1: ODE 动力学约束与 Q-value 估计的矛盾**。LTC 的 ODE 求解器要求权重非负、连接高度稀疏（邻接矩阵 88.6% 为零）、反转电位固定为 ±1，导致 motor 神经元的输出范围和表达能力严重受限。我们在实验中发现直接用 motor 输出做 Q-values 时性能接近随机策略。如何在保持 NCP 生物合理性的同时获得足够的函数逼近能力，是一个核心技术挑战。

**Challenge 2: 参数效率高但安全性不足**。NCP 以最少参数取得最高 reward，但碰撞率（52.5%）明显高于 GRU（34.2%）。稀疏连接限制了安全关键信息从 sensory 到 motor 的传递路径。如何在不破坏稀疏结构的前提下改善安全性，是拓扑搜索要解决的核心问题。

**Challenge 3: 拓扑搜索效率**。每评估一个候选拓扑需训练数万步，而 NCP 的 6 步 ODE 展开使其训练速度仅为 MLP 的一半。在有限算力下高效搜索大规模拓扑空间，需要设计低成本的适应度近似策略。

**Challenge 4: 可解释性的量化**。NCP 的"可解释性"目前主要通过定性分析（激活热图、拓扑可视化）展示，缺乏定量指标。如何量化衡量模型的可解释程度，是严谨论证 NCP 优势的关键。

## 研究问题

1. NCP 作为轻量可解释的 Planning Head，在多场景自动驾驶决策任务中，能否以极少参数量达到甚至超越传统 MLP/LSTM/GRU 的性能？
2. 能否通过进化搜索算法自动发现比手工设计更优的 NCP 布线拓扑，同时兼顾性能与安全性？
3. NCP 的结构化稀疏连接（对比全连接 LTC）是否带来可量化的性能和安全性优势？
4. 如何解决 NCP 的 ODE 约束与 RL Q-value 估计之间的表达能力矛盾？

## 研究内容

### Part A: NCP Planning Head 性能验证

将 NCP 集成为 DQN 的 Q-network，针对 Challenge 1 提出 q_head 方案（用全部神经元状态经线性层映射 Q-values，而非仅用 motor 输出）。在 Highway-env 的 4 个场景（高速巡航、匝道合流、交叉路口、环岛）上与 Random、MLP、LSTM、GRU、FullyConnected-LTC 进行系统对比。评估维度包括累计奖励、碰撞率、参数量、推理延迟和参数效率。3 个随机种子确保统计显著性。

### Part B: NCP 布线拓扑进化搜索

针对 Challenge 2 和 3，设计结构化搜索空间（神经元数量、连接密度、兴奋/抑制比例、循环连接数），使用进化策略（锦标赛选择 + 均匀交叉 + 高斯变异）搜索最优拓扑。采用**安全感知的多目标适应度**（reward - α × collision_rate），在多场景联合评估下自动发现兼顾性能与安全性的布线方案。

### Part C: 消融实验

- Q-head 消融：验证独立 Q-value 头对 NCP 性能的必要性（回应 Challenge 1）
- ODE 展开数消融：确定精度-效率最优权衡点（回应 Challenge 3）
- 网络规模消融：探究 NCP 性能与规模的关系

### Part D: 可解释性分析

针对 Challenge 4，设计多层次可解释性分析框架：
- 拓扑结构可视化与统计量对比（手工 vs 搜索：连接密度、兴奋/抑制比、层间分布）
- Command 神经元激活与驾驶决策的对应分析
- 碰撞前关键时刻的异常激活模式识别
- 神经元功能归因：逐个 mask 掉 command 神经元，量化对各动作 Q-values 的影响，构建归因矩阵
- 跨场景激活模式降维对比（PCA / t-SNE）

## 创新点

**创新点 1：发现并解决 NCP 在 RL 中的表达能力瓶颈。** 现有 NCP 工作直接用 motor 神经元输出做控制信号，适用于监督学习的车道保持。我们发现 NCP 的 ODE 约束（权重非负、稀疏连接、固定极性）导致 motor 输出无法有效估计 Q-values，性能接近随机策略（reward 7.48）。我们提出用全部神经元隐状态经线性 Q-head 映射的方案，在保持 ODE 动力学可解释性的同时将性能提升至 28.09（超越所有 baseline）。这是首次揭示 NCP 与 RL value estimation 的兼容性问题并给出解决方案。

**创新点 2：面向 NCP 布线拓扑的安全感知进化搜索。** 现有 NCP 的四层布线完全手工设计（借鉴 C. elegans），没有任何自动优化方法。我们提出结构化搜索空间（6 个离散基因 + 约束修复）、安全感知多目标适应度（`fitness = reward - α × collision_rate`）和多场景联合评估，首次实现 NCP 布线参数的自动化搜索优化。

**创新点 3：结构化稀疏 vs 全连接的定量验证。** 设计严格对照实验——NCP 与 FC_LTC 参数量完全相同（5,265），唯一区别是连接拓扑。NCP reward 28.09 vs FC_LTC 26.43（+6%），方差 4.52 vs 7.13（更稳定）。首次定量证明 NCP 的生物启发稀疏拓扑优于同参数量的全连接 LTC。

**创新点 4：面向自动驾驶的多层次 NCP 可解释性分析框架。** 超越单一激活可视化，提出 5 个层次的分析体系：结构层（拓扑统计量对比）、动态层（command 神经元激活与决策对齐）、安全层（碰撞前关键时刻异常模式识别）、因果层（逐神经元 mask 的功能归因矩阵）、泛化层（跨场景激活降维对比）。

## 初步结果（Highway-v0, seed=42）

| 模型 | Eval Reward | 参数量 | Reward/1K Params |
|------|-------------|--------|-----------------|
| Random | 7.48 ± 6.28 | 1 | - |
| LSTM | 19.65 ± 11.85 | 23,621 | 0.83 |
| MLP | 24.62 ± 8.74 | 11,909 | 2.07 |
| FC_LTC | 26.43 ± 7.13 | 5,265 | 5.02 |
| GRU | 27.35 ± 5.08 | 17,797 | 1.54 |
| **NCP (Ours)** | **28.09 ± 4.52** | **5,265** | **5.33** |

NCP 以最少参数量取得最高 reward 和最低方差，参数效率是 GRU 的 3.5 倍。

## 项目结构

```
NEU-MLhomework/
├── main.tex                    # 论文主文件（东北大学本科毕业设计 LaTeX 模板）
├── chapter/                    # 论文各章节
├── reference.bib               # 参考文献
├── experiment_design.md        # 实验设计文档
├── research_background.md      # 研究背景与相关工作（含 77 篇参考文献）
└── code/                       # 实验代码
    ├── configs/default.yaml    # 超参数配置
    ├── envs/                   # Highway-env 环境封装
    ├── models/                 # NCP/MLP/LSTM/GRU/FC_LTC 模型实现
    ├── search/                 # 进化拓扑搜索
    ├── utils/                  # 可视化、日志、回放缓冲
    ├── scripts/                # 训练/评估/搜索脚本
    └── results/                # 实验结果
```

## 快速开始

```bash
conda activate ncp-highway
cd code/

# 训练单个模型
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --arch ncp --env highway-v0 --seed 42

# 批量训练（tmux 并行）
bash scripts/run_part_a.sh

# 进化拓扑搜索
CUDA_VISIBLE_DEVICES=1 python scripts/run_search.py

# 完整实验流水线
CUDA_VISIBLE_DEVICES=1 python scripts/run_all.py
```

## 技术栈

- Python 3.10 / PyTorch / Highway-env v1.10.2
- NCP/LTC 实现基于 [keras-ncp](https://github.com/mlech26l/keras-ncp)（Apache 2.0）
- 论文模板基于 [NEU-undergraduate-thesis-LaTeX-template](https://github.com/neuljh/NEU-undergraduate-thesis-LaTeX-template)（MIT）
