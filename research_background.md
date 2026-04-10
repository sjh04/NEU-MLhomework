# 研究背景与相关工作

## 一、研究背景

### 1.1 自动驾驶规划的发展与挑战

自动驾驶规划系统负责根据感知信息生成安全、高效的驾驶轨迹，是自动驾驶技术栈的核心环节。近年来，该领域经历了从传统模块化流水线（感知→预测→规划）向端到端学习的范式转变，并在 2025-2026 年进入多范式并存、评估体系重构的新阶段。

**传统模块化方案**将感知、预测、规划解耦为独立子系统，每个模块可独立优化和调试，但模块间的信息损失和误差累积导致整体性能受限。**端到端方案**则将传感器输入直接映射到驾驶动作，通过联合优化消除模块间壁垒。UniAD [1]（CVPR 2023 Best Paper）是这一范式转变的标志性工作，在 nuScenes 规划基准上取得 L2 误差 (3s) 0.71m。此后 VAD [2]（ICCV 2023）、SparseDrive [3]（ECCV 2024）、DriveTransformer [4]（ICLR 2025）持续推进，到 2025 年端到端稀疏 Transformer 已成为主流范式。

与此同时，2024-2025 年间 **VLM/VLA 驱动的规划**（如 EMMA [5]、ORION [6]）、**扩散模型规划**（如 DiffusionDrive [7]）、**大规模 RL 训练**（如 CaRL [8]）以及**世界模型**（如 GAIA-1 [9]、DrivingWorld [10]）等新范式快速涌现，形成了五条技术路线并行发展的格局。工业界方面，Tesla FSD V12/V13 全面采用端到端神经网络取代手写规则 [11]，Waymo 基于 Gemini 的 EMMA 模型展示了 VLM 规划的潜力，Wayve 在伦敦进行 L4 级测试并持续发表世界模型研究。

然而，当前自动驾驶规划仍面临三大核心挑战：

1. **计算效率**：端到端 Transformer 模型参数量巨大，训练需高端 GPU 集群（如 UniAD 需 8×A800），难以在车端边缘 SoC（通常 ≤320 TOPS）上实时部署。L2 级系统要求端到端延迟 20-100ms，VLM 方案延迟通常 >100ms，远不能满足要求。

2. **可解释性与安全合规**：欧盟《人工智能法案》(EU AI Act) 已于 2025 年开始分阶段实施，将于 **2026 年 8 月全面执行** [12]，自动驾驶被归类为"高风险 AI 系统"，要求具备可追溯性、可解释性和人类监督能力。中国工信部于 2024 年发布《智能网联汽车准入管理》规定，要求 L3 及以上系统具备决策可追溯能力 [13]。ISO/PAS 8800（AI 安全在道路车辆中的应用）和 UL 4600 标准在 2024-2025 年持续更新，逐步纳入对神经网络规划器的验证要求。当前端到端黑箱模型无法解释"为什么做出这个决策"，难以通过安全认证。

3. **长尾场景泛化**：自动驾驶面临大量罕见但高危的长尾场景（如施工区域、极端天气、异常交通参与者），当前模型在分布外场景下表现不稳定。

### 1.2 生物启发的轻量可解释架构

面对上述挑战，生物启发的神经网络架构提供了一条差异化路径。秀丽隐杆线虫（*Caenorhabditis elegans*）仅有 302 个神经元和约 7,000 个突触，却能完成趋化性、温度趋向性、回避反射等复杂行为 [14]。这启发研究者思考：**是否可以用极少量但结构化的神经元实现高效、可解释的自动驾驶决策？**

Lechner、Hasani 等人 [15] 在 2020 年提出的神经电路策略（Neural Circuit Policies, NCP）正是这一思路的实现——仅用 19 个神经元和 253 个突触即可完成自动驾驶车道保持任务，参数量比传统 LSTM 网络少数个数量级，且注意力自动聚焦在道路边界和地平线上，与人类驾驶员的注意力模式高度一致。

Liquid AI 公司（由 Hasani、Lechner 等从 MIT CSAIL 创立）在 2024-2026 年间取得了显著进展：完成 **2.5 亿美元 A 轮融资**（AMD Ventures 领投，估值 23.5 亿美元）[16]，发布了 LFM2 系列模型和 LEAP 边缘部署平台，并与 AMD、G42、Insilico Medicine 等建立战略合作。这表明液态神经网络已从学术研究走向商业化落地。

### 1.3 架构搜索的必要性

NCP 的四层布线架构（sensory→inter→command→motor）直接借鉴了线虫神经系统的结构，但具体的连接参数（如每层神经元数量、连接密度、兴奋/抑制比例）仍依赖人工设计。这引出一个自然的问题：**能否通过自动化搜索发现更优的布线拓扑？**

神经架构搜索（NAS）和神经进化方法为此提供了成熟的工具。特别是 NEAT [17]（NeuroEvolution of Augmenting Topologies）专门针对网络拓扑的进化搜索，与 NCP 的布线搜索问题天然契合。2024-2025 年间，LLM 辅助架构设计成为新热点——EvoPrompting [18]（NeurIPS 2023）展示了用 LLM 生成和变异网络架构代码的可行性，多篇后续工作探索用 GPT-4/Claude 作为 NAS 的"变异算子"。

### 1.4 本研究的定位

基于上述背景，本研究提出将 NCP 作为自动驾驶规划的轻量可解释 Planning Head，并设计进化搜索算法自动优化其布线拓扑。在 Highway-env 仿真平台上，通过与 MLP、LSTM、GRU 等基线模型的系统对比，验证 NCP 在性能、效率和可解释性三个维度上的综合优势。

---

## 二、相关工作

### 2.1 液态神经网络与神经电路策略

#### 2.1.1 液态时间常数网络 (LTC)

Hasani、Lechner、Amini、Rus 等人 [19] 于 2021 年在 AAAI 上提出液态时间常数网络（Liquid Time-Constant Networks, LTC），这是一种基于常微分方程（ODE）的新型 RNN 单元。其核心动力学方程为：

$$c_m \frac{dx}{dt} = -g_{leak}(x - v_{leak}) + \sum_j w_j \sigma(x_j)(e_{rev,j} - x)$$

等价于 $dx/dt = -[1/\tau(x,I)] \odot x + f(x,I)$，其中时间常数 $\tau$ 是输入和状态的函数，而非传统 RNN 中的固定常数。这赋予了 LTC 对输入的动态自适应能力——在快速变化的场景中 $\tau$ 自动减小（快速响应），在平稳场景中 $\tau$ 增大（平滑滤波）。

LTC 通过多步欧拉法离散化求解 ODE（默认 6 步展开）。关键的可学习参数包括：泄漏电导 $g_{leak}$、膜电容 $c_m$、突触权重 $w$（均含非负约束），以及突触反转电位 $e_{rev}$（由布线的邻接矩阵决定极性：兴奋性 +1 / 抑制性 -1）。

#### 2.1.2 神经电路策略 (NCP)

Lechner、Hasani 等人 [15] 于 2020 年在 *Nature Machine Intelligence* 上发表了 NCP，这是液态神经网络在自动驾驶领域的开创性工作。

NCP 受秀丽隐杆线虫神经连接组启发，设计了结构化的四层稀疏布线架构：
- **感觉层 (Sensory)**：接收外部输入特征
- **中间层 (Inter)**：进行特征整合
- **命令层 (Command)**：做出高级决策，具有层内循环连接
- **运动层 (Motor)**：输出控制信号

连接通过 `sensory_fanout`、`inter_fanout`、`motor_fanin` 和 `recurrent_command_synapses` 精确控制。每条突触具有兴奋性（+1）或抑制性（-1）的极性。

**关键成果**：
- 仅 **19 个神经元、253 个突触** 即可完成端到端车道保持，而 LSTM 基线需约 10 万参数
- 注意力分析显示 NCP 自动聚焦道路边界和地平线，与人类驾驶员注意力模式一致
- 三层可解释性框架：注意力归因→全局动力学（PCA）→细胞级可审计

#### 2.1.3 闭式连续时间网络 (CfC)

Hasani、Lechner 等人 [20] 于 2022 年在 *Nature Machine Intelligence* 上发表了 CfC，用闭式解析解替代 LTC 中的 ODE 数值求解器。将 6 步欧拉展开压缩为单次前向传播，训练速度提升约 **5-8 倍**，同时保持液态网络的因果性和表达能力。在驾驶实验中，CfC 以约 4,000 个参数完成端到端车道保持。

2025-2026 年间，CfC 在多个新领域取得应用进展：卫星遥感时间序列重建（CfC-mmRNN，比传统方法提升 33%-42%）、车联网异常行为检测、工业机器人滑模控制（SMC-CfC-G）、以及脑电情感识别（AC-CfC）[21]。

#### 2.1.4 Liquid-S4

Hasani、Lechner 等人 [22] 于 2023 年在 ICLR 上发表了 Liquid-S4，将液态动力学与结构化状态空间模型（S4）结合。引入输入依赖的状态转移矩阵，在保持 S4 并行训练能力的同时获得液态网络的自适应特性。在 Long Range Arena 基准上超越原始 S4。Liquid-S4 是从 LTC/CfC（串行 RNN）通向 LFM（大规模并行模型）的关键跳板。

#### 2.1.5 液态基础模型 (LFM) 与 Liquid AI 最新进展

Liquid AI 在 2025-2026 年间发布了一系列重要模型和平台 [16][23]：

**LFM2 系列**（2025 年 7 月）：包含 350M/700M/1.2B 三个规模，采用**门控短卷积 + 分组查询注意力（GQA）**的混合架构，在 10T tokens 上训练，CPU 推理速度是 Qwen3/Gemma3 的 2 倍。LFM2 技术报告 [23] 的重要结论："在边缘设备预算下，额外添加 SSM/linear attention 并不能提升质量——大部分 hybrid SSM 的收益可以被短卷积 + 少量全局注意力捕获。"

**LFM2-VL**（2025 年 8 月）：视觉-语言多模态模型（450M/1.6B），GPU 推理比同类快 2 倍。**LFM2-VL-3B**（2025 年 10 月）进一步推出 3B 参数版本，与 AMD、Robotec.ai 联合演示在 AMD Ryzen AI 处理器上的嵌入式机器人实时多模态感知 [24]。

**LFM2-2.6B 与 Exp 版**（2025 年 11-12 月）：IFEval 79.56%，GSM8K 82.41%。实验版（Exp）通过纯 RL 训练，IFBench 超越 **263 倍大的 DeepSeek R1-0528** [25]。

**LFM2.5**（2026 年 1 月）：最新版本，1.2B Base/Instruct，28T tokens 预训练 + 规模化 RL，IFEval 达 86.23。同步更新 LFM2.5-VL-1.6B 视觉语言版及日语/音频语言变体 [26]。

**Nanos 概念**（2025 年 9 月）：350M-2.6B 参数模型在专业 Agent 任务上达到 GPT-4o 级别，运行成本降低 50 倍 [27]。

**LEAP 边缘部署平台**（2025 年 7 月）：开发者仅需 10 行代码即可在手机/笔记本部署模型，原生支持 AMD Ryzen AI 处理器 [28]。

**产业合作**：与 G42 合作开发中东/北非/全球南方的私有化 AI 方案（2025 年 6 月）[29]；与 Insilico Medicine 合作推出药物发现专用模型 LFM2-2.6B-MMAI（2026 年 3 月）[30]。

**架构演化的关键认知**：从 LTC 的 ODE 动力学到 LFM2 的门控卷积+GQA，液态网络经历了从纯 ODE 范式到务实混合架构的重大转变。但其核心设计哲学始终一致：**用输入自适应的动态系统替代静态参数化计算**。

#### 2.1.6 鲁棒飞行导航

Chahine、Hasani 等人 [31] 于 2023 年在 *Science Robotics* 上发表了液态网络在无人机视觉导航中的应用。仅 19 个神经元的 NCP 在分布外场景下（未见过的森林、不同季节/光照）表现出远超大型网络的鲁棒性，有力证明了稀疏生物启发架构在 OOD 泛化方面的潜力。

#### 2.1.7 其他 LNN 变体与新应用

- **Attn-LTC**：将空间注意力与 LTC 结合用于轨迹预测，参数量远少于 Transformer 方案
- **LGTC**（Liquid Graph Time-Constant Networks）[32]：将液态动力学与图神经网络结合用于多智能体分布式控制，CDC 2024
- **Liquid DINO**：LNN + DINOv2 视觉 Transformer，驾驶员行为识别平均准确率 83.79%
- **NCP 用于绿色 AI 网络** [33]（2025）：利用 LTC 稀疏性实现极小模型的通信网络优化
- **StayLTC**（2025）：LTC 用于医院住院时长预测

#### 2.1.8 竞争架构对比（2025-2026）

| 架构 | 特点 | 现状（2026 年初）|
|------|------|-----------------|
| **Liquid/LFM2** | 混合卷积+GQA，边缘部署优先 | 小模型效率标杆，商业产品落地 |
| **Mamba2** | SSM+线性注意力，并行训练 | 长序列退化问题待解，多用于混合架构 |
| **RWKV-7** | RNN+Transformer 混合，线性复杂度 | 社区活跃，开源生态完善 |
| **xLSTM-7B** | 改良 LSTM，训练速度是 Transformer 的 3.5 倍 | 多项基准上 Pareto 占优，势头强劲 |

总体趋势：**混合架构**（注意力+状态空间/卷积）成为主流，Liquid AI 凭借边缘部署和 AMD 硬件优化建立了差异化优势。

### 2.2 自动驾驶规划方法

#### 2.2.1 端到端规划

**UniAD** [1]（Yihan Hu et al., CVPR 2023 Best Paper）将检测、跟踪、建图、运动预测和规划统一在 query-based Transformer 框架中，在 nuScenes 上取得 L2 误差 (3s) 0.71m、碰撞率 0.31%。

**VAD** [2]（Bo Jiang et al., ICCV 2023）用向量化场景表示替代稠密栅格，L2 误差 (3s) 约 0.65m，推理更快。

**SparseDrive** [3]（Wenchao Sun et al., ECCV 2024）采用全稀疏架构，在感知和规划间共享稀疏实例特征，约 9.0 FPS。

**DriveTransformer** [4]（ICLR 2025）提出任务间双向交互 Transformer + 稀疏流式设计，确立了 Transformer 在端到端规划中的主导地位。

**BridgeAD** [59]（CVPR 2025）通过历史预测桥接过去与未来，增强端到端规划的时序一致性。

**2025-2026 年趋势**：CVPR 2025/2026、ICLR 2025/2026 上端到端驾驶论文数量激增，稀疏表示+端到端成为默认范式。**LEAD** [60]（CVPR 2026）基于 CARLA 构建新一代端到端驾驶研究框架。Waymo 已确认其量产车辆采用端到端基础模型控制 [61]，三大公司（Waymo、Tesla、Wayve）在技术路线上正在趋同。

#### 2.2.2 VLM/LLM 驱动的规划

**EMMA** [5]（Waymo, 2024）基于 Gemini 多模态模型，将端到端驾驶统一为视觉-语言生成任务。**DriveVLM** [34]（Tsinghua MARS Lab, 2024）实现链式推理（场景描述→场景分析→层级规划）与传统规划器的双系统架构。**GPT-Driver** [35]（2023）将轨迹生成重构为语言建模任务。**LightEMMA** [36]（2025）探索轻量化 VLM 驾驶模型。

2025 年，多个团队探索 **VLA（Vision-Language-Action）** 统一模型：**ORION** [6]（ICCV 2025）在统一框架内完成视觉理解、语言推理和轨迹规划；**RT-2** [37]（Google DeepMind, 2023）率先验证 VLM 直接输出动作 token 的可行性；**OpenVLA** [38]（2024）开源了 VLA 基础设施。

**OpenDriveVLA** [62]（AAAI 2026）是首个大规模开源 VLA 驾驶模型。**VLA-MP** 融合 BEV 多模态感知与 GRU-自行车动力学级联适配器，将语义理解转化为物理一致轨迹。ICCV 2025 Workshop 发布了首个 VLA4AD 综述 [63]，系统梳理了超过 20 个代表性模型。理想汽车已在量产车上部署 VLA 架构。

**核心局限**：推理延迟高（>100ms）、幻觉问题、缺乏空间精确推理。2025-2026 年研究方向集中在模型蒸馏、量化和专用小模型设计以降低延迟。

#### 2.2.3 扩散模型规划

**DiffusionDrive** [7]（CVPR 2025 Highlight）通过截断扩散实现高效轨迹生成，NAVSIM PDMS 达 88.1。**CTG++** [39] 利用扩散模型进行可控交通场景生成。

**Diffusion Planner** [64]（ICLR 2025 Oral）首次充分发挥扩散模型在运动规划中的能力，支持预测与规划联合建模，无需规则后处理。**GoalFlow** [65]（CVPR 2025）采用目标点引导的 Flow Matching，仅需一步去噪即可生成轨迹。**Flow-Planner** [66]（NeurIPS 2025）通过精细轨迹 tokenization 和时空融合架构增强交互建模。**FlowDrive** 引入 moderated guidance 机制提升轨迹多样性。

**2025-2026 年趋势**：Flow Matching 正在取代扩散模型成为轨迹生成的主流方法，一步去噪大幅降低推理延迟。

#### 2.2.4 强化学习规划

**Think2Drive** [40]（2024）在 CARLA 中通过 RL 训练端到端驾驶策略，在 Bench2Drive 闭环基准上领先。**CaRL** [8]（2024/2025）采用简单奖励 + PPO 大规模训练，超越所有先前方法，开启了"大规模 RL 训练"新范式。

受 DeepSeek-R1 启发，2025 年出现了**纯 RL 冷启动**技术路线，以少量数据通过多阶段 RL 训练实现驾驶策略学习。**V-Max** 基于 Waymax 硬件加速仿真器构建大规模 RL 研究框架。小鹏图灵芯片已在量产车中运行 VLA + 自主强化学习模型。Waymo 内部训练循环大量使用 RL 进行大规模策略优化。

**2025-2026 年趋势**：离线 RL 持续升温；世界模型 + RL 的组合训练范式受到关注；约束 RL（CPO、WCSAC）与形式化验证结合确保策略安全性。**DRLSL** [67]（2026）将深度 RL 与符号一阶逻辑结合，实现安全的高速公路自动驾驶。

#### 2.2.5 世界模型

**GAIA-1** [9]（Wayve, 2023）是 9B 参数生成式世界模型。**DrivingWorld** [10]（NeurIPS 2025）构建长时域时空一致的驾驶世界模型。**GenAD** [41]（CVPR 2024）将生成式建模与规划结合。

Wayve 发布 **GAIA-2** [68]（2025 年 3 月，150 亿参数），支持多视角可控视频生成。**GAIA-3** [69]（2025 年 12 月）进一步提升仿真真实感与安全场景覆盖。**Waymo World Model** [70]（2026 年 2 月）提出超大规模高保真仿真生成模型，刷新驾驶仿真质量上限。

**2025-2026 年趋势**：世界模型正从"视频生成工具"转变为自动驾驶训练与评估的核心基础设施，被用于大规模 RL 策略训练和安全场景验证。

### 2.3 基准与评估体系

- **nuScenes** [42]（Motional, CVPR 2020）：最广泛使用的开环规划基准（L2 位移误差、碰撞率），但开环指标与实际驾驶性能相关性弱——简单恒速直行基线即可获得不错 L2 分数
- **NAVSIM** [43]（CoRL 2025）：非反应式闭环评估框架，引入 PDM Score（综合舒适度、进度、安全等多维度）
- **Bench2Drive** [44]（NeurIPS 2024）：基于 CARLA 的大规模闭环基准，44 种交互场景
- **nuPlan**（Motional）：2024-2025 年持续更新，成为闭环规划评估的重要平台
- **Highway-env** [45]（Leurent, 2018, Farama Foundation）：轻量级 RL 驾驶决策环境，本研究选用此平台

**NAVSIM v2**（CoRL 2025）采用伪仿真范式，实现大规模真实数据驱动的非反应式评测。**Bench2Drive-Speed** 支持期望速度条件控制，**Bench2Drive-VL** 首次将 VLM 驾驶智能体纳入闭环评测。**LEAD** [60]（CVPR 2026）基于 CARLA 构建新一代端到端驾驶研究框架。

**开环 vs 闭环之争**是 2024-2025 年社区核心讨论，推动了评估体系全面向闭环转变。2025-2026 年社区在讨论更好的评估指标，特别是安全性、舒适性和长尾场景处理能力。

### 2.4 自动驾驶中的可解释性

Atakishiyev 等人 [46]（2024）在综述 "Explainable AI for Safe and Trustworthy Autonomous Driving" 中总结了五类 XAI 贡献：可解释设计、代理模型、可解释监控、辅助解释和可解释验证。Zablocki 等人 [47] 在 IJCV 2022 综述中全面梳理了可解释性挑战。

当前主流方法包括：
1. **事后解释**（Post-hoc）：Grad-CAM、注意力可视化、显著性图——应用广泛但难以反映真实决策机制
2. **概念级解释**：BDD-X 数据集 [48]（UC Berkeley, ECCV 2018）提供驾驶行为自然语言解释；DriveVLM [34] 利用链式推理实现分层解释
3. **内在可解释架构**：NCP [15] 的四层结构使每个神经元可被逐一追踪——**内在可解释性**远优于事后方法

**2025 年新进展**：
- **Concept Bottleneck Models** 被应用于端到端驾驶，将中间表征对齐到人类可理解的语义概念（如"前方有行人"、"车道线偏移"）
- **机制可解释性**（Mechanistic Interpretability）在 2025 年迎来爆发：Anthropic 的归因图（attribution graphs）成功应用于生产模型 Claude 3.5 Haiku [71]；DeepMind 的 Gemma Scope 2 将 SAE 分析扩展到 270 亿参数。该方法从 LLM 领域向控制模型扩展
- Waymo 发布了规划模块可解释性技术报告（2024），展示注意力可视化和关键场景归因方法
- **EU AI Act 最终期限**：2026 年 8 月 2 日是几乎所有要求的生效截止日 [72]，制造商需开发实时 AI 决策解释仪表板

NCP 的可解释性优势在于其结构化设计天然映射到"感知→理解→决策→执行"的认知流程，兴奋/抑制极性提供因果推理线索，且极少的参数量使全面审计成为可能。

### 2.5 神经架构搜索 (NAS)

#### 2.5.1 NAS 基础方法

NAS 旨在自动化设计神经网络架构，主要范式包括：
- **基于 RL 的 NAS**：Zoph & Le [49]（2017）最早提出，需约 2000 GPU 天
- **进化方法**：通过变异和交叉演化网络群体，搜索灵活性高
- **可微分 NAS (DARTS)**：Liu 等人 [50]（ICLR 2019）将搜索时间降至 2-3 天，在 PTB 语言建模上达到 55.7 测试困惑度

#### 2.5.2 面向 RL 策略的 NAS

ES-ENAS [51]（Song, Choromanski 等, 2021）以零额外成本搜索 RL 策略架构。Miao 等人 [52]（AutoML 2022）证明 DARTS 兼容 PPO、Rainbow-DQN 和 SAC。EMNAS-RL [53]（2025）提出鲁棒的多目标进化网络架构搜索。

#### 2.5.3 神经进化与拓扑搜索

**NEAT** [17]（Stanley & Miikkulainen, 2002）通过历史标记实现拓扑交叉，利用物种化保护创新。**HyperNEAT** [54] 通过 CPPN 间接编码连接模式，可表征生物网络的对称性和重复性。EvoTorch [55]（2023-2024）等框架将进化搜索与 PyTorch 深度集成，使 NEAT 类方法更易在 GPU 上运行。

#### 2.5.4 2025 年新趋势：LLM 辅助架构设计

**EvoPrompting** [18]（NeurIPS 2023）展示了用 LLM 生成和变异网络架构代码的可行性。2024-2025 年，多篇工作探索用 GPT-4/Claude 作为 NAS 的"变异算子"，替代传统遗传算法的随机变异。硬件感知 NAS（如 Once-for-All、TinyNAS）在边缘部署场景持续演进，NVIDIA 针对 Jetson 系列优化了 NAS 流程。

2025 年最重要的进展是 **TensorNEAT** [73]——基于 JAX 的 GPU 加速 NEAT 库，支持原始 NEAT、CPPN 和 HyperNEAT 算法，相比 CPU 库实现约 **500 倍加速**。此外，**RZ-NAS** [74] 引入反思式零成本策略增强 LLM 引导 NAS；**LoRA-NAS** [75] 将 NAS 与低秩自适应结合。

这些方法与 NCP 的布线搜索问题天然契合。本研究采用进化策略进行 NCP 布线拓扑搜索。

### 2.6 深度强化学习与高速公路驾驶

#### 2.6.1 DQN 及其变体

DQN [56]（Mnih et al., Nature 2015）通过经验回放和目标网络稳定训练，是处理离散动作空间的经典方法。后续改进包括 Double DQN、Dueling DQN、Prioritized Experience Replay 等。在自动驾驶场景中，离散元动作（保持、变道、加速、减速）天然适配 DQN。

#### 2.6.2 Highway-env 平台

Highway-env [45]（Leurent, 2018）由 Farama Foundation 维护，提供高速公路巡航、匝道合流、交叉路口、环岛等多种场景。其特点包括：基于运动学的轻量车辆模型、离散/连续动作空间、可配置观测类型（运动学向量、灰度图像、TTC 网格）、IDM 驱动的交通流。

多项研究在 Highway-env 上对比了不同 RL 算法 [57]：PPO 在平均回报和方差方面通常优于 DQN、SAC 和 DDPG，但 DQN 在离散动作空间下训练更稳定且更易实现。2025 年的详细对比 [76] 进一步确认了这一结论，同时发现 Double DQN 在高速公路稳定场景中表现出色，而 PPO 在动态复杂场景中更鲁棒。**总体共识**：PPO 在连续动作空间和复杂任务中更优，DQN 在离散空间中收敛更快、训练成本更低。

2025 年基于 Highway-env 的新研究包括：混合 RL 决策框架（结合规则与学习策略评估置信度）、多智能体 RL 双车协同驾驶 [77] 等。MetaDrive 和 SMARTS 模拟器也获得更多关注，提供更丰富的多智能体场景。

#### 2.6.3 安全感知强化学习

安全是 RL 驾驶的核心关切。相关方法包括：虚拟安全笼 [58]、基于 TTC 的安全过滤器、约束 RL（CPO、WCSAC）以及形式化验证与 RL 训练的结合。2025 年，Waabi 等公司展示了从高保真仿真直接部署到实车的成果，sim-to-real 迁移取得重要进展。

NCP 的可审计性为安全验证提供了独特优势——可以追踪特定危险场景下每个神经元的响应，分析安全违规的根本原因。

---

## 三、本研究的创新点

基于上述文献调研，本研究的创新点在于：

1. **首次系统评估 NCP 作为 DQN Q-network 在多场景自动驾驶决策中的性能**：现有 NCP 研究集中在简单车道保持任务，尚未在多场景（高速、合流、交叉口、环岛）决策任务上与主流 RNN 基线进行系统对比。

2. **首次提出面向 NCP 布线拓扑的进化搜索算法**：现有 NCP 布线完全手工设计，本研究设计了结构化搜索空间和多场景联合适应度评估，实现布线参数的自动优化。

3. **多维度可解释性分析**：不仅评估 NCP 的性能指标，还通过 command 神经元激活可视化、跨场景激活模式对比、搜索拓扑与手工拓扑的结构差异分析，深入挖掘 NCP 的可解释性价值。

---

## 参考文献

[1] Yihan Hu et al., "Planning-oriented Autonomous Driving," CVPR 2023 (Best Paper).

[2] Bo Jiang et al., "VAD: Vectorized Scene Representation for Efficient Autonomous Driving," ICCV 2023.

[3] Wenchao Sun et al., "SparseDrive: End-to-End Autonomous Driving with Sparse Representations," ECCV 2024.

[4] "DriveTransformer," ICLR 2025.

[5] Waymo, "EMMA: End-to-End Multimodal Model for Autonomous Driving," 2024.

[6] "ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language-Action Model," ICCV 2025.

[7] "DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving," CVPR 2025.

[8] "CaRL: Simple Reward + PPO Large-Scale Training for Autonomous Driving," 2024/2025.

[9] Hu et al., "GAIA-1: A Generative World Model for Autonomous Driving," Wayve, 2023.

[10] "DrivingWorld," NeurIPS 2025.

[11] Tesla AI Day, "FSD V12/V13: End-to-End Neural Network Architecture," 2024-2025.

[12] Schmoeller et al., "EU AI Act Explained: How Europe's New AI Regulations Will Affect Autonomous Transport," 2025.

[13] 中国工信部, "智能网联汽车准入管理规定," 2024.

[14] White et al., "The Structure of the Nervous System of the Nematode C. elegans," Phil. Trans. R. Soc. B, 1986.

[15] Lechner, Hasani, Amini, Henzinger, Rus, Grosu, "Neural Circuit Policies Enabling Auditable Autonomy," Nature Machine Intelligence, 2020.

[16] Liquid AI, "We Raised $250M to Scale Capable and Efficient General-Purpose AI," 2024. https://www.liquid.ai/blog/we-raised-250m-to-scale-capable-and-efficient-general-purpose-ai

[17] Stanley & Miikkulainen, "Evolving Neural Networks through Augmenting Topologies," Evolutionary Computation, 2002.

[18] "EvoPrompting: Language Models for Code-Level Neural Architecture Search," NeurIPS 2023.

[19] Hasani, Lechner, Amini, Rus et al., "Liquid Time-constant Networks," AAAI 2021.

[20] Hasani, Lechner, Amini et al., "Closed-form Continuous-time Neural Networks," Nature Machine Intelligence, 2022.

[21] Multiple authors, CfC applications in remote sensing (CfC-mmRNN), IoV security, robot control (SMC-CfC-G), EEG emotion recognition (AC-CfC), 2025.

[22] Hasani, Lechner et al., "Liquid Structural State-Space Models," ICLR 2023.

[23] Liquid AI, "LFM2 Technical Report," arXiv:2511.23404, 2025.

[24] Liquid AI, "Agentic Robotics at the Edge: LFM2-VL-3B with AMD and Robotec.ai," 2025. https://www.liquid.ai/blog/agentic-robotics-at-the-edge

[25] Liquid AI, "LFM2-2.6B-Exp," Hugging Face, 2025. https://huggingface.co/LiquidAI/LFM2-2.6B-Exp

[26] "Liquid AI Releases LFM2.5," MarktechPost, 2026. https://www.marktechpost.com/2026/01/06/liquid-ai-releases-lfm2-5/

[27] Liquid AI, "Nanos: Extremely Small Foundation Models," 2025. https://www.liquid.ai/press/liquid-unveils-nanos

[28] Liquid AI, "LEAP: Liquid Edge-AI Platform," 2025. https://www.liquid.ai/press/liquid-ai-launches-leap-and-apollo

[29] Liquid AI & G42, "Partnership for Private AI Solutions," 2025. https://www.liquid.ai/press/g42-and-liquid-ai-partner

[30] Liquid AI & Insilico Medicine, "LFM2-2.6B-MMAI for Drug Discovery," 2026. https://www.liquid.ai/press/liquid-ai-insilico-medicine-partnership

[31] Chahine, Hasani et al., "Robust Flight Navigation Out of Distribution with Liquid Neural Networks," Science Robotics, 2023.

[32] "Liquid Graph Time-Constant Networks," arXiv:2404.13982, CDC 2024.

[33] "NCP for Green AI-Native Networks," arXiv:2504.02781, 2025.

[34] Tsinghua MARS Lab, "DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models," 2024.

[35] Jiageng Mao et al., "GPT-Driver: Learning to Drive with GPT," 2023.

[36] "LightEMMA: Lightweight End-to-End Multimodal Model for Autonomous Driving," 2025.

[37] Brohan et al., "RT-2: Vision-Language-Action Models," Google DeepMind, 2023.

[38] Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," 2024.

[39] "CTG++: Controllable Traffic Generation," 2024.

[40] "Think2Drive: Efficient RL by Thinking in Latent World Model," 2024.

[41] "GenAD: Generative Autonomous Driving," CVPR 2024.

[42] Caesar et al., "nuScenes: A Multimodal Dataset for Autonomous Driving," CVPR 2020.

[43] "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation," CoRL 2025.

[44] "Bench2Drive: Multi-Ability Benchmarking of Closed-Loop E2E AD," NeurIPS 2024.

[45] Leurent, "An Environment for Autonomous Driving Decision-Making," highway-env, 2018.

[46] Atakishiyev et al., "Explainable AI for Safe and Trustworthy Autonomous Driving," arXiv:2402.10086, 2024.

[47] Zablocki et al., "Explainability of Deep Vision-Based AD Systems," IJCV, 2022.

[48] Kim et al., "Textual Explanations for Self-Driving Vehicles (BDD-X)," ECCV 2018.

[49] Zoph & Le, "Neural Architecture Search with Reinforcement Learning," ICLR 2017.

[50] Liu et al., "DARTS: Differentiable Architecture Search," ICLR 2019.

[51] Song, Choromanski et al., "ES-ENAS: Efficient Evolutionary Architecture Search," 2021.

[52] Miao et al., "Differentiable Architecture Search for Reinforcement Learning," AutoML 2022.

[53] "EMNAS-RL: Robust Multi-Objective Evolutionary Network Architecture Search," arXiv:2506.08533, 2025.

[54] Stanley et al., "A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks (HyperNEAT)," Artificial Life, 2009.

[55] EvoTorch: Scalable Evolutionary Computation in PyTorch, 2023-2024.

[56] Mnih et al., "Human-level Control through Deep Reinforcement Learning," Nature, 2015.

[57] "Comparison of DQN, PPO, SAC for Highway Driving," IEEE, 2024.

[58] "Virtual Safety Cages for Autonomous Vehicle Control," PMC, 2021.

[59] Zhang et al., "BridgeAD: Bridging Past and Future End-to-End Autonomous Driving with Historical Prediction," CVPR 2025.

[60] "LEAD: A New Framework for End-to-End Autonomous Driving Research," CVPR 2026. https://github.com/autonomousvision/lead

[61] "The Era of End-to-End Autonomy," arXiv:2603.16050, 2026.

[62] "OpenDriveVLA: An Open-Source Large-Scale VLA Model for Autonomous Driving," AAAI 2026. https://github.com/DriveVLA/OpenDriveVLA

[63] "VLA for Autonomous Driving Survey," ICCV 2025 Workshop. arXiv:2506.24044.

[64] Zheng et al., "Diffusion Planner: Joint Prediction and Planning for Autonomous Driving," ICLR 2025 (Oral). https://github.com/ZhengYinan-AIR/Diffusion-Planner

[65] Yin et al., "GoalFlow: Goal-Conditioned Flow Matching for Multimodal Trajectories Generation," CVPR 2025. https://github.com/YvanYin/GoalFlow

[66] "Flow-Planner: Flow Matching for Autonomous Driving Planning," NeurIPS 2025.

[67] "DRLSL: Deep RL with Symbolic First-Order Logic for Safe Highway Driving," SAGE Journals, 2026.

[68] Wayve, "GAIA-2: 15B Parameter Multiview Controllable World Model," arXiv:2503.20523, 2025.

[69] Wayve, "GAIA-3: Enhanced Simulation Fidelity and Safety Scenario Coverage," 2025. https://wayve.ai/press/wayve-launches-gaia3/

[70] Waymo, "The Waymo World Model: A New Frontier for AD Simulation," 2026. https://waymo.com/blog/2026/02/the-waymo-world-model

[71] Anthropic, "Attribution Graphs for Mechanistic Interpretability," 2025.

[72] EU AI Act Implementation Timeline, 2026. https://axis-intelligence.com/eu-ai-act-news-2026/

[73] "TensorNEAT: GPU-Accelerated NEAT Library in JAX," arXiv:2404.01817, 2025. https://github.com/EMI-Group/tensorneat

[74] "RZ-NAS: Reflective Zero-Cost NAS with LLM Guidance," OpenReview, 2025.

[75] "LoRA-NAS: Combining NAS with Low-Rank Adaptation," Springer, 2025.

[76] "Comparative Analysis of DQN, PPO, SAC for Autonomous Driving," IEEE, 2025.

[77] "Multi-Agent RL for Cooperative Highway Driving," IEEE TITS, 2025.
