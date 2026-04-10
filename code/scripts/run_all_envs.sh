#!/bin/bash
# Run all experiments: 6 archs × 4 envs × 3 seeds = 72 experiments
# Strategy: run 6 archs in parallel per (env, seed), then move to next
# Uses GPU 1 only

set -e
cd /mnt/aisdata/sjh04/LFM/NEU-MLhomework/code

ENVS=("highway-v0" "merge-v0" "intersection-v0" "roundabout-v0")
SEEDS=(42 0 123)
ARCHS=("ncp" "mlp" "lstm" "gru" "fc_ltc" "random")
STEPS_LEARN=50000
STEPS_RANDOM=5000
GPU=1

total=$((${#ENVS[@]} * ${#SEEDS[@]}))
current=0

for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        current=$((current + 1))
        echo ""
        echo "================================================================"
        echo "  Batch ${current}/${total}: env=${env}, seed=${seed}"
        echo "  $(date)"
        echo "================================================================"

        pids=()
        for arch in "${ARCHS[@]}"; do
            dir="results/${arch}_${env}_s${seed}"
            mkdir -p "$dir"

            if [ "$arch" == "random" ]; then
                steps=$STEPS_RANDOM
            else
                steps=$STEPS_LEARN
            fi

            # Skip if already completed (model.pt exists)
            if [ -f "$dir/model.pt" ]; then
                echo "  [SKIP] $arch on $env seed=$seed (already done)"
                continue
            fi

            echo "  [START] $arch on $env seed=$seed ($steps steps)"
            CUDA_VISIBLE_DEVICES=$GPU python scripts/train.py \
                --arch $arch --env $env --seed $seed --steps $steps \
                > "$dir/train.log" 2>&1 &
            pids+=($!)
        done

        # Wait for all parallel jobs in this batch
        echo "  Waiting for ${#pids[@]} jobs..."
        for pid in "${pids[@]}"; do
            wait $pid
        done
        echo "  Batch ${current}/${total} done at $(date)"
    done
done

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE at $(date)"
echo "================================================================"
