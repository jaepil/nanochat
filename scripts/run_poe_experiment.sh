#!/bin/bash
# PoE Local Learning Experiment: baseline vs flat PoE
# Designed for 8xA100/H100 on AWS
set -euo pipefail
export OMP_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"

RESULTS_DIR="$HOME/poe_results"
mkdir -p "$RESULTS_DIR"
cd /home/ubuntu/nanochat

# ---- Setup: download data and train tokenizer ----
echo "========================================"
echo "  Setup: data download + tokenizer"
echo "========================================"
# Download data shards (need ~170 for d20 training)
uv run python3 -m nanochat.dataset -n 8
uv run python3 -m nanochat.dataset -n 170 &
DATASET_PID=$!
# Train tokenizer on first 8 shards
uv run python3 -m scripts.tok_train
echo "Waiting for dataset download..."
wait $DATASET_PID
echo "Setup complete."

# ---- Experiment 1: Baseline (standard backprop) ----
echo "========================================"
echo "  Experiment 1: Baseline (standard BP)"
echo "========================================"
uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --run=dummy \
    --depth=20 \
    --poe-mode=none \
    --device-batch-size=8 \
    --save-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    2>&1 | tee "$RESULTS_DIR/baseline.log"

# Copy checkpoint
cp -r ~/.cache/nanochat/base_checkpoints/d20 "$RESULTS_DIR/baseline_ckpt"
echo "Baseline checkpoint saved."

# ---- Experiment 2: Flat PoE (poe_every=5, 4 stages) ----
echo "========================================"
echo "  Experiment 2: Flat PoE (4 stages)"
echo "========================================"
rm -rf ~/.cache/nanochat/base_checkpoints/d20

uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --run=dummy \
    --depth=20 \
    --poe-mode=flat \
    --poe-every=5 \
    --device-batch-size=8 \
    --save-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    2>&1 | tee "$RESULTS_DIR/flat_poe.log"

# Copy checkpoint
cp -r ~/.cache/nanochat/base_checkpoints/d20 "$RESULTS_DIR/flat_poe_ckpt"
echo "Flat PoE checkpoint saved."

echo "========================================"
echo "  All experiments complete!"
echo "  Results in: $RESULTS_DIR"
echo "========================================"
ls -lh "$RESULTS_DIR"/
