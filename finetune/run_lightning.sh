#!/bin/bash
# Lightning.ai training script
# Run inside a Lightning.ai Studio with L4 GPU
#
# Setup:
#   1. Open Lightning.ai Studio with L4 GPU ($0.60/hr)
#   2. Clone repo: git clone https://github.com/Rohan5commit/options-trading-bot.git
#   3. cd options-trading-bot
#   4. bash finetune/run_lightning.sh
#
# Account switching:
#   1. Checkpoints auto-save to HF Hub every 1000 steps
#   2. When budget runs out, open new Studio with L4 GPU
#   3. Run this script again - it resumes from latest checkpoint

set -e

echo "=== Options LLM Training on Lightning.ai ==="
echo "GPU: L4 at \$0.60/hr"
echo "Budget: \$45 total across accounts"
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
    echo "Building training dataset..."
    python finetune/build_dataset.py
else
    echo "Training dataset already exists."
fi

# Count examples
TRAIN_COUNT=$(wc -l < ./training_data/train.jsonl)
echo "Training examples: $TRAIN_COUNT"

# Start training
echo ""
echo "Starting training..."
echo "Checkpoints save to HF Hub every 1000 steps"
echo "To resume on new account, just run this script again"
echo ""

python finetune/train.py

echo ""
echo "=== Training Complete ==="
