#!/bin/bash
# KD Experiment: Gemma 4 31B-it (teacher) -> nanochat d32 (student)
# Designed for p4d.24xlarge (8x A100-40GB) on AWS
set -euo pipefail
export OMP_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$HOME/.cache/huggingface"

RESULTS_DIR="$HOME/kd_results"
mkdir -p "$RESULTS_DIR"
cd /home/ubuntu/nanochat

# ---- Phase 0: Setup data + tokenizer ----
echo "========================================"
echo "  Phase 0: Data + Tokenizer"
echo "========================================"
uv run python3 -m nanochat.dataset -n 8
uv run python3 -m nanochat.dataset -n 60 &
DATASET_PID=$!
uv run python3 -m scripts.tok_train
wait $DATASET_PID

# ---- Phase 1: Generate teacher logits ----
echo "========================================"
echo "  Phase 1: Teacher logit generation"
echo "========================================"
uv sync --group dev  # install transformers, bitsandbytes
uv run pip install bitsandbytes accelerate
uv run python3 -m scripts.generate_teacher_logits \
    --teacher google/gemma-4-31b-it \
    --top-k 16 \
    --num-tokens 500000000 \
    --batch-size 4 \
    --seq-len 2048 \
    2>&1 | tee "$RESULTS_DIR/teacher_logits.log"

# ---- Phase 2: Student KD training (PoE + KD) ----
echo "========================================"
echo "  Phase 2: Student KD + PoE training"
echo "========================================"
uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --run=dummy \
    --depth=32 \
    --poe-mode=flat \
    --poe-every=8 \
    --device-batch-size=8 \
    --window-pattern=L \
    --kd-logits-dir="$HOME/.cache/nanochat/teacher_logits" \
    --kd-alpha=0.5 \
    --kd-temperature=2.0 \
    --target-param-data-ratio=3 \
    --save-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    2>&1 | tee "$RESULTS_DIR/kd_poe_train.log"

cp -r ~/.cache/nanochat/base_checkpoints/d32 "$RESULTS_DIR/kd_poe_ckpt"

echo "========================================"
echo "  KD EXPERIMENT DONE"
echo "========================================"
