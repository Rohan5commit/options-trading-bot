#!/bin/bash
# Lightning.ai Phase 1 training script
# Run inside a Lightning.ai Studio with L4 GPU
#
# Setup:
#   1. Open Lightning.ai Studio with L4 GPU ($0.48/hr)
#   2. Clone repo: git clone https://github.com/Rohan5commit/options-trading-bot.git
#   3. cd options-trading-bot
#   4. bash finetune/run_lightning.sh
#
# Phase 1 budget: $15 = 31.25 hours on L4
# After Phase 1, switch to Modal Phase 2 (see: python finetune/lightning_helper.py phase2)

set -e

echo "=========================================="
echo "  Options LLM Training - Phase 1"
echo "  Lightning.ai L4 GPU"
echo "=========================================="
echo ""
echo "GPU: L4 at \$0.48/hr"
echo "Budget: \$15 (31.25 hours)"
echo "Training: 160K examples × 8 epochs"
echo ""

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU detected. Start a Studio with L4 GPU."
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q torch --index-url https://download.pytorch.org/whl/cu121

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Set it to resume from checkpoints."
    echo "  export HF_TOKEN=your_token_here"
fi

# Build dataset if not exists
if [ ! -f "./training_data/train.jsonl" ]; then
    echo "Building training dataset (160K examples)..."
    python finetune/build_dataset.py
else
    echo "Training dataset already exists."
fi

# Count examples
TRAIN_COUNT=$(wc -l < ./training_data/train.jsonl)
echo "Training examples: $TRAIN_COUNT"
echo ""

# Start training
echo "Starting Phase 1 training..."
echo "Checkpoints save to HF Hub every 1000 steps"
echo "After \$15 budget, switch to Modal Phase 2"
echo ""

export GPU_PRICE_HR=0.48
export TRAINING_BUDGET=15.0
export GPU_NAME="L4"

python finetune/train.py

echo ""
echo "=========================================="
echo "  Phase 1 Complete!"
echo "=========================================="
echo ""
echo "Next: Switch to Modal Phase 2"
echo "  python finetune/lightning_helper.py phase2"
echo ""
