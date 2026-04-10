#!/bin/bash
# Fast parallel runner: 2 batches per GPU, 4 batches simultaneous
# Each batch = 6 archs, each GPU handles 12 processes
# ~3GB per GPU, well within 23GB limit

set -e
cd /mnt/aisdata/sjh04/LFM/NEU-MLhomework/code

ARCHS=("ncp" "mlp" "lstm" "gru" "fc_ltc" "random")

start_batch() {
    local gpu=$1 env=$2 seed=$3

    for arch in "${ARCHS[@]}"; do
        local dir="results/${arch}_${env}_s${seed}"
        mkdir -p "$dir"
        [ -f "$dir/model.pt" ] && continue

        local steps=50000
        [ "$arch" == "random" ] && steps=5000

        CUDA_VISIBLE_DEVICES=$gpu python scripts/train.py \
            --arch $arch --env $env --seed $seed --steps $steps \
            > "$dir/train.log" 2>&1 &
    done
}

# Build remaining jobs (skip completed)
declare -a TODO_ENV TODO_SEED
for env in highway-v0 merge-v0 intersection-v0 roundabout-v0; do
    for seed in 42 0 123; do
        done=$(ls results/*_${env}_s${seed}/model.pt 2>/dev/null | wc -l)
        [ "$done" -ge 6 ] && echo "[SKIP] $env seed=$seed (all done)" && continue
        TODO_ENV+=("$env")
        TODO_SEED+=("$seed")
    done
done

total=${#TODO_ENV[@]}
echo "================================================================"
echo "  Remaining: $total batches to run"
echo "  Strategy: 4 batches at a time (2 per GPU)"
echo "  Start: $(date)"
echo "================================================================"

i=0
round=1
while [ $i -lt $total ]; do
    echo ""
    echo "--- Round $round @ $(date '+%H:%M:%S') ---"
    pids=()

    # Launch up to 4 batches: 2 on GPU0, 2 on GPU1
    for slot in 0 1 2 3; do
        idx=$((i + slot))
        [ $idx -ge $total ] && break

        gpu=$((slot / 2))  # slot 0,1 -> GPU0; slot 2,3 -> GPU1
        env="${TODO_ENV[$idx]}"
        seed="${TODO_SEED[$idx]}"

        echo "  [GPU$gpu] Batch $((idx+1))/$total: $env seed=$seed"
        start_batch $gpu "$env" $seed
    done

    # Collect all background PIDs
    pids=($(jobs -p))
    echo "  Waiting for ${#pids[@]} processes..."
    wait
    echo "  Round $round done @ $(date '+%H:%M:%S')"

    i=$((i + 4))
    round=$((round + 1))
done

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE @ $(date)"
echo "================================================================"
