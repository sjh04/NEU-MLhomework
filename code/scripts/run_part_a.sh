#!/bin/bash
# Part A: Train all 4 architectures on all 4 environments
# Uses tmux to run experiments in parallel on GPU 1

SESSION="ncp_exp"
CONDA_ENV="ncp-highway"
CODE_DIR="/mnt/aisdata/sjh04/LFM/NEU-MLhomework/code"
GPU="1"
STEPS=50000

# Kill existing session if any
tmux kill-session -t $SESSION 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION

# 4 environments × 4 architectures = 16 experiments
# Run 4 at a time (one per env), then repeat for next arch
ENVS=("highway-v0" "merge-v0" "intersection-v0" "roundabout-v0")
ARCHS=("ncp" "mlp" "lstm" "gru")

window_idx=0
for arch in "${ARCHS[@]}"; do
    for env in "${ENVS[@]}"; do
        if [ $window_idx -eq 0 ]; then
            tmux rename-window -t $SESSION "${arch}_${env}"
        else
            tmux new-window -t $SESSION -n "${arch}_${env}"
        fi

        tmux send-keys -t $SESSION:"${arch}_${env}" \
            "source activate $CONDA_ENV && cd $CODE_DIR && CUDA_VISIBLE_DEVICES=$GPU python scripts/train.py --arch $arch --env $env --seed 42 --steps $STEPS 2>&1 | tee results/${arch}_${env}_s42/train.log" \
            Enter

        window_idx=$((window_idx + 1))
    done
done

echo "Started $window_idx experiments in tmux session '$SESSION'"
echo "Monitor with: tmux attach -t $SESSION"
echo "Check GPU:    nvidia-smi"
