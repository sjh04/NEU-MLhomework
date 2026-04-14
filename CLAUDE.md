# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

东北大学机器学习课程大作业：**面向可解释自动驾驶规划的神经回路策略拓扑优化**。在 Highway-env 上用 DQN 训练 NCP（Neural Circuit Policy）作为可解释 Planning Head，并用进化算法搜索最优 wiring 拓扑。

## Environment Setup

```bash
conda activate ncp-highway          # Python 3.10, PyTorch, highway-env, matplotlib, pyyaml
cd /mnt/aisdata/sjh04/LFM/NEU-MLhomework/code
```

GPU: 两张 A10 (23GB each)。用 `CUDA_VISIBLE_DEVICES=1` 避开 GPU 0 上的其他任务。

## Common Commands

```bash
# Train single model
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --arch ncp --env highway-v0 --seed 42 --steps 50000

# Train all 4 archs in parallel via tmux
bash scripts/run_part_a.sh

# Run evolutionary topology search
CUDA_VISIBLE_DEVICES=1 python scripts/run_search.py

# Evaluate a trained model
python scripts/evaluate.py --arch ncp --env highway-v0 --checkpoint results/ncp_highway-v0_s42/model.pt

# Full pipeline (Part A + B + D)
CUDA_VISIBLE_DEVICES=1 python scripts/run_all.py

# Monitor running experiments
for w in ncp_hw mlp_hw lstm_hw gru_hw; do echo "=== $w ===" && tmux capture-pane -t ncp_exp:$w -p | grep -E "^\s+\[" | tail -1; done
```

No test suite; validation is through training scripts and evaluation metrics.

## Architecture

### Data Flow
```
Highway-env obs (5,7) → flatten (35,) → Q-Network → Q-values (5) → argmax → discrete action
```

All 4 environments (highway-v0, merge-v0, intersection-v0, roundabout-v0) are normalized to the same observation shape via `envs/env_factory.py` (Kinematics, 5 vehicles × 7 features [presence,x,y,vx,vy,cos_h,sin_h], flattened to 35). Collision reward set to -5.0.

### Q-Network Hierarchy
`models/q_networks.py` defines 6 architectures sharing `QNetworkBase`:
- **RandomQNetwork** (1 param) — uniform random, no learning (lower bound)
- **MLPQNetwork** (13,189 params) — feedforward, not recurrent
- **LSTMQNetwork** (26,181 params) — LSTM(64) + Linear head
- **GRUQNetwork** (19,717 params) — GRU(64) + Linear head
- **NCPQNetwork** (7,152 params) — LTCCell(NCP wiring) + 2-layer Q-head (our method)
- **FCLTCQNetwork** (7,152 params) — FullyConnected LTC + 2-layer Q-head (ablation baseline)

All expose `forward(obs, hidden) → (q_values, new_hidden)` and `init_hidden(batch_size)`. Factory: `build_q_network(arch, obs_dim, act_dim, config, wiring)`.

### NCP-specific Design (Critical)
- `models/ltc_cell.py`: ODE-based RNN cell with 6 Euler unfolds per step. Sparsity masks and reversal potentials come from the wiring's adjacency matrix.
- `models/wiring.py`: `NCP` class builds 4-layer sparse connectivity (sensory→inter→command→motor). Neuron ID layout: `[0..motor | command | inter]`.
- **Q-head**: NCPQNetwork uses a **2-layer MLP** (`Linear(state_size,32)→ReLU→Linear(32,act_dim)`) on the **full hidden state** (all neurons), NOT just motor neuron outputs. Using raw motor outputs as Q-values performs very poorly (reward ~7 vs baselines ~28).
- After every `optimizer.step()`, call `ltc.apply_weight_constraints()` to enforce `w, sensory_w, cm, gleak ≥ 0`.

### DQN Agent
`models/dqn_agent.py`: **Double DQN** with target network and epsilon-greedy. Uses online network to select best action, target network to evaluate Q-value (reduces overestimation). For recurrent models (LSTM/GRU/NCP), samples length-8 sequences from `utils/replay_buffer.py` that don't cross episode boundaries.

### Evolutionary Search
`search/genome.py`: `WiringGenome` dataclass with 6 integer genes (inter_neurons, command_neurons, sensory_fanout, inter_fanout, recurrent_command_synapses, motor_fanin) and constraint repair.
`search/evolution.py`: Tournament selection + crossover + Gaussian mutation. Fitness = average reward across highway-v0 and roundabout-v0 after 15K-step quick training.

## Key Config: `configs/default.yaml`
- `device: "cuda"` — set via `CUDA_VISIBLE_DEVICES` externally
- `dqn.train_steps: 50000`, `dqn.sequence_length: 8`
- `ncp.*`: default wiring params (inter=12, cmd=8, fanout=4, etc.)
- `search.*`: population=20, generations=30

## Paper (LaTeX)
东北大学本科毕业设计模板。学号 20235937，计算机科学与工程学院，计算机科学与技术，孙家恒，指导教师卢炳先副教授。

`main.tex` is the entry point; chapters in `chapter/`, abstracts in `intro/`, bibliography in `reference.bib`. Compile with `xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex`.

论文撰写需遵守东北大学本科毕业设计（论文）书写印制规范：字体字号（论文题目黑体二号、章标题黑体小二号、节标题黑体四号、正文宋体小四号）、页边距（上下2.5cm、左3.0cm、右2.5cm）、图表公式按章编号（如 图2.1、表3.4、式（4.2））、引文采用方括号上标如 [3]。

## Conventions
- 实验结果保存到 `code/results/`，图表复制到 `img/` 用于论文
- 中文 commit message
- 长时间实验用 tmux，脚本见 `scripts/run_part_a.sh`
- NCP 原始实现来自 keras-ncp（Apache 2.0），位于 `/mnt/aisdata/sjh04/LFM/NCP/code/keras-ncp/`
