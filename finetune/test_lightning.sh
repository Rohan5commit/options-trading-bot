#!/bin/bash
set -ex

echo "=== Step 1: Download repo ==="
curl -sL https://github.com/Rohan5commit/options-trading-bot/archive/refs/heads/main.tar.gz -o /tmp/repo.tar.gz
mkdir -p /workspace
tar xzf /tmp/repo.tar.gz --strip-components=1 -C /workspace
cd /workspace
echo "Files:"
ls /workspace/

echo "=== Step 2: Check GPU ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== Step 3: Build dataset ==="
python finetune/build_dataset.py

echo "=== Step 4: Check training data ==="
wc -l ./training_data/train.jsonl

echo "=== DONE ==="
