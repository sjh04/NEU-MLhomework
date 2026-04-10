#!/bin/bash
# Launch ALL remaining experiments at once
# GPU assignment: even batch index -> GPU 0, odd -> GPU 1
# ~40-50 processes total, ~3-4 CPU cores each, fits in 60 cores

set -e
cd /mnt/aisdata/sjh04/LFM/NEU-MLhomework/code

ARCHS=("ncp" "mlp" "lstm" "gru" "fc_ltc" "random")
total_launched=0
batch_idx=0

echo "================================================================"
echo "  Launching ALL experiments @ $(date)"
echo "================================================================"

for env in highway-v0 merge-v0 intersection-v0 roundabout-v0; do
    for seed in 42 0 123; do
        # Check if entire batch is done
        done_count=$(ls results/*_${env}_s${seed}/model.pt 2>/dev/null | wc -l)
        if [ "$done_count" -ge 6 ]; then
            echo "[SKIP] $env seed=$seed (6/6 done)"
            continue
        fi

        gpu=$((batch_idx % 2))
        batch_idx=$((batch_idx + 1))

        for arch in "${ARCHS[@]}"; do
            dir="results/${arch}_${env}_s${seed}"
            mkdir -p "$dir"
            [ -f "$dir/model.pt" ] && continue

            steps=50000
            [ "$arch" == "random" ] && steps=5000

            echo "  [GPU$gpu] $arch $env s$seed"
            CUDA_VISIBLE_DEVICES=$gpu python scripts/train.py \
                --arch $arch --env $env --seed $seed --steps $steps \
                > "$dir/train.log" 2>&1 &
            total_launched=$((total_launched + 1))
        done
    done
done

echo ""
echo "================================================================"
echo "  Launched $total_launched processes @ $(date)"
echo "  Waiting for all to complete..."
echo "================================================================"

wait

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE @ $(date)"
echo "================================================================"

# Print summary
echo ""
echo "=== Results Summary ==="
for env in highway-v0 merge-v0 intersection-v0 roundabout-v0; do
    for seed in 42 0 123; do
        done_count=$(ls results/*_${env}_s${seed}/model.pt 2>/dev/null | wc -l)
        echo "  $env seed=$seed: $done_count/6"
    done
done
