#!/bin/bash
set -ex

LOGFILE="/workspace/training_output.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "  Options LLM Training - Phase 1"
echo "  Lightning.ai L4 GPU"
echo "  Started: $(date)"
echo "=========================================="

# Step 1: Install deps
echo ""
echo "=== Installing dependencies ==="
pip install -q --upgrade pip
pip install -q transformers peft bitsandbytes accelerate datasets huggingface-hub trl scipy sentencepiece protobuf

# Step 2: Download repo
echo ""
echo "=== Downloading repo ==="
curl -sL https://github.com/Rohan5commit/options-trading-bot/archive/refs/heads/main.tar.gz -o /tmp/repo.tar.gz
mkdir -p /workspace
tar xzf /tmp/repo.tar.gz --strip-components=1 -C /workspace
cd /workspace
echo "Repo downloaded. Files:"
ls /workspace/

# Step 3: Check GPU
echo ""
echo "=== GPU Info ==="
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# Step 4: Build dataset
echo ""
echo "=== Building dataset ==="
python finetune/build_dataset.py
echo "Training examples: $(wc -l < ./training_data/train.jsonl)"

# Step 5: Train
echo ""
echo "=== Starting training ==="
echo "HF_TOKEN set: $([ -n "$HF_TOKEN" ] && echo 'yes' || echo 'NO - THIS MAY CAUSE FAILURES')"
export GPU_PRICE_HR=0.48
export TRAINING_BUDGET=15.0
export GPU_NAME="L4"
python finetune/train.py

echo ""
echo "=========================================="
echo "  Training Complete: $(date)"
echo "=========================================="
