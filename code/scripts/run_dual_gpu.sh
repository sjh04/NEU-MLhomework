#!/bin/bash
# Dual-GPU parallel experiment runner
# GPU 0: even batches, GPU 1: odd batches
# Each batch = 6 archs in parallel for one (env, seed)

set -e
cd /mnt/aisdata/sjh04/LFM/NEU-MLhomework/code

ARCHS=("ncp" "mlp" "lstm" "gru" "fc_ltc" "random")
STEPS_LEARN=50000
STEPS_RANDOM=5000

run_batch() {
    local gpu=$1
    local env=$2
    local seed=$3
    local batch_id=$4
    local total=$5

    echo "[GPU${gpu}] Batch ${batch_id}/${total}: env=${env}, seed=${seed} @ $(date '+%H:%M:%S')"

    local pids=()
    for arch in "${ARCHS[@]}"; do
        local dir="results/${arch}_${env}_s${seed}"
        mkdir -p "$dir"

        if [ "$arch" == "random" ]; then
            local steps=$STEPS_RANDOM
        else
            local steps=$STEPS_LEARN
        fi

        if [ -f "$dir/model.pt" ]; then
            echo "  [GPU${gpu}] SKIP $arch (done)"
            continue
        fi

        echo "  [GPU${gpu}] START $arch"
        CUDA_VISIBLE_DEVICES=$gpu python scripts/train.py \
            --arch $arch --env $env --seed $seed --steps $steps \
            > "$dir/train.log" 2>&1 &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo "[GPU${gpu}] Batch ${batch_id}/${total} done @ $(date '+%H:%M:%S')"
}

# Build job list: (env, seed) pairs
ENVS=("highway-v0" "merge-v0" "intersection-v0" "roundabout-v0")
SEEDS=(42 0 123)

declare -a JOBS_ENV JOBS_SEED
for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        JOBS_ENV+=("$env")
        JOBS_SEED+=("$seed")
    done
done

total=${#JOBS_ENV[@]}
echo "================================================================"
echo "  Total: $total batches, 2 GPUs, ~${total}/2 rounds"
echo "  Start: $(date)"
echo "================================================================"

# Process 2 batches at a time (one per GPU)
i=0
while [ $i -lt $total ]; do
    gpu0_pid=""
    gpu1_pid=""

    # GPU 0: batch i
    if [ $i -lt $total ]; then
        run_batch 0 "${JOBS_ENV[$i]}" "${JOBS_SEED[$i]}" $((i+1)) $total &
        gpu0_pid=$!
    fi

    # GPU 1: batch i+1
    j=$((i+1))
    if [ $j -lt $total ]; then
        run_batch 1 "${JOBS_ENV[$j]}" "${JOBS_SEED[$j]}" $((j+1)) $total &
        gpu1_pid=$!
    fi

    # Wait for both GPUs
    [ -n "$gpu0_pid" ] && wait $gpu0_pid
    [ -n "$gpu1_pid" ] && wait $gpu1_pid

    i=$((i+2))
done

echo ""
echo "================================================================"
echo "  ALL $total BATCHES COMPLETE @ $(date)"
echo "================================================================"
