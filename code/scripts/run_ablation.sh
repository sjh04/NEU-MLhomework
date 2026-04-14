#!/bin/bash
# Part C: Ablation experiments on highway-v0, seed=42
# C1: Q-head ablation (raw motor output vs 2-layer q_head)
# C2: ODE unfolds ablation (1/3/6/12)
# All on GPU 1

set -e
cd /mnt/aisdata/sjh04/LFM/NEU-MLhomework/code
GPU=1
STEPS=50000
ENV="highway-v0"
SEED=42

echo "=== Part C: Ablation Experiments ==="
echo "Start: $(date)"

# C2: ODE unfolds ablation
# Need a modified train script that accepts --ode_unfolds
for unfolds in 1 3 6 12; do
    dir="results/ablation_ode${unfolds}_${ENV}_s${SEED}"
    mkdir -p "$dir"
    [ -f "$dir/model.pt" ] && echo "[SKIP] ode_unfolds=$unfolds" && continue
    echo "[START] ode_unfolds=$unfolds"
    CUDA_VISIBLE_DEVICES=$GPU python -c "
import sys, os
sys.path.insert(0, '.')
from models.wiring import NCP
from models.ltc_cell import LTCCell
from models.q_networks import NCPQNetwork
from models.dqn_agent import DQNAgent
from envs.env_factory import get_obs_dim, get_action_dim
from scripts.train import train
from utils.common import load_config, set_seed

config = load_config('configs/default.yaml')
config['device'] = 'cuda'
config['save_dir'] = 'results'
config['log_interval'] = 1000
set_seed($SEED)

obs_dim, act_dim = get_obs_dim(), get_action_dim('$ENV')
wiring = NCP(inter_neurons=12, command_neurons=8, motor_neurons=act_dim,
             sensory_fanout=4, inter_fanout=4, recurrent_command_synapses=4, motor_fanin=4)
# Create LTCCell with custom ode_unfolds
import torch.nn as nn
from models.q_networks import QNetworkBase
class AblationNCP(QNetworkBase):
    def __init__(self):
        super().__init__(obs_dim, act_dim)
        self.wiring = wiring
        self.ltc = LTCCell(wiring, in_features=obs_dim, ode_unfolds=$unfolds)
        state_size = wiring.units
        q_hidden = max(32, state_size)
        self.q_head = nn.Sequential(nn.Linear(state_size, q_hidden), nn.ReLU(), nn.Linear(q_hidden, act_dim))
    @property
    def is_recurrent(self): return True
    def init_hidden(self, batch_size):
        import torch
        return torch.zeros(batch_size, self.ltc.state_size, device=next(self.parameters()).device)
    def forward(self, obs, hidden=None):
        import torch
        if hidden is None: hidden = self.init_hidden(obs.size(0))
        if obs.dim() == 3:
            for t in range(obs.size(1)): _, hidden = self.ltc(obs[:, t, :], hidden)
        else: _, hidden = self.ltc(obs, hidden)
        return self.q_head(hidden), hidden

# Monkey-patch build_q_network
import models.q_networks as qn
orig = qn.build_q_network
qn.build_q_network = lambda *a, **kw: AblationNCP()
agent = DQNAgent(obs_dim, act_dim, 'ncp', config, device='cuda')
qn.build_q_network = orig

# Rename output dir
import shutil
result = train('ncp', '$ENV', config, seed=$SEED, train_steps=$STEPS, agent=agent)
src = 'results/ncp_${ENV}_s${SEED}'
dst = '$dir'
if os.path.exists(src):
    for f in os.listdir(src):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    shutil.rmtree(src)
print('Done: ode_unfolds=$unfolds')
" > "$dir/train.log" 2>&1 &
done

echo "=== Waiting for ablation experiments ==="
wait

echo ""
echo "=== All ablations done: $(date) ==="
for unfolds in 1 3 6 12; do
    dir="results/ablation_ode${unfolds}_${ENV}_s${SEED}"
    if [ -f "$dir/model.pt" ]; then echo "  ode=$unfolds: DONE"; else echo "  ode=$unfolds: FAILED"; fi
done
