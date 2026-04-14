#!/bin/bash
# Train NCP with the best searched genome on all envs × 3 seeds
# Strategy: backup existing ncp_*, train, rename to ncp_searched_*, restore

set -e
cd /mnt/aisdata/sjh04/LFM/NEU-MLhomework/code

ENVS=("highway-v0" "merge-v0" "intersection-v0" "roundabout-v0")
SEEDS=(42 0 123)

# Backup existing hand-designed NCP results
echo "=== Backing up hand-designed NCP results ==="
for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        src="results/ncp_${env}_s${seed}"
        bak="results/_ncp_bak_${env}_s${seed}"
        if [ -d "$src" ]; then
            mv "$src" "$bak"
            echo "  backup: $src -> $bak"
        fi
    done
done

# Launch all training jobs in parallel (4 per GPU)
echo ""
echo "=== Launching searched NCP training ==="
launched=0
for i in "${!ENVS[@]}"; do
    for j in "${!SEEDS[@]}"; do
        env="${ENVS[$i]}"
        seed="${SEEDS[$j]}"
        gpu=1  # use GPU 1 only (GPU 0 too crowded)
        launched=$((launched + 1))

        # Skip if already completed (ncp_searched_*)
        if [ -f "results/ncp_searched_${env}_s${seed}/model.pt" ]; then
            echo "  [SKIP] $env seed=$seed (already done)"
            continue
        fi

        dir="results/_ncp_searched_tmp_${env}_s${seed}"
        mkdir -p "$dir"

        echo "  [GPU$gpu] $env seed=$seed"
        CUDA_VISIBLE_DEVICES=$gpu python scripts/train_searched.py \
            --env $env --seed $seed \
            > "$dir/train.log" 2>&1 &
    done
done

echo ""
echo "=== Waiting for $launched training jobs ==="
wait
echo ""

# Rename ncp_* to ncp_searched_*, restore backups
echo "=== Renaming results ==="
for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        src="results/ncp_${env}_s${seed}"
        dst="results/ncp_searched_${env}_s${seed}"
        bak="results/_ncp_bak_${env}_s${seed}"
        tmp="results/_ncp_searched_tmp_${env}_s${seed}"

        if [ -d "$src" ]; then
            rm -rf "$dst"
            mv "$src" "$dst"
            echo "  ncp -> ncp_searched: $env s$seed"
        fi
        if [ -d "$bak" ]; then
            mv "$bak" "$src"
            echo "  restored backup: $env s$seed"
        fi
        rm -rf "$tmp"
    done
done

echo ""
echo "=== Done ==="
ls -d results/ncp_searched_*/ 2>/dev/null | wc -l
echo "searched NCP experiments"
