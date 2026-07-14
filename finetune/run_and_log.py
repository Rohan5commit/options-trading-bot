#!/usr/bin/env python3
"""Wrapper that runs training and captures all output for HF Hub upload."""
import os, sys, traceback, json, tempfile
from datetime import datetime

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "Rohan556/options-llm-checkpoints"
output_lines = []

def log(msg):
    line = f"[{datetime.utcnow().isoformat()}] {msg}"
    output_lines.append(line)
    print(line, flush=True)

def upload_log():
    if not HF_TOKEN:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO, exist_ok=True, private=True)
        log_data = {
            "output": "\n".join(output_lines),
            "timestamp": datetime.utcnow().isoformat(),
            "gpu": os.environ.get("GPU_NAME", "unknown"),
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(log_data, f, indent=2)
            f.flush()
            api.upload_file(path_or_fileobj=f.name, repo_id=HF_REPO, path_in_repo="train_log.json")
        print("LOG UPLOADED TO HF HUB")
    except Exception as e:
        print(f"LOG UPLOAD FAILED: {e}")

try:
    log("STEP 1: Installing dependencies")
    rc = os.system("pip install -q transformers peft bitsandbytes accelerate datasets huggingface-hub trl scipy sentencepiece protobuf")
    log(f"pip install exit code: {rc}")

    log("STEP 2: Downloading repo")
    rc = os.system("curl -sL https://github.com/Rohan5commit/options-trading-bot/archive/refs/heads/main.tar.gz -o /tmp/repo.tar.gz")
    log(f"curl exit code: {rc}")
    rc = os.system("mkdir -p /workspace && tar xzf /tmp/repo.tar.gz --strip-components=1 -C /workspace")
    log(f"tar exit code: {rc}")

    log("STEP 3: Checking imports")
    try:
        import torch
        log(f"torch OK: {torch.__version__}, CUDA={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        log(f"torch import FAILED: {e}")

    try:
        import transformers, peft, bitsandbytes, trl
        log(f"transformers={transformers.__version__}, peft={peft.__version__}, bitsandbytes={bitsandbytes.__version__}, trl={trl.__version__}")
    except Exception as e:
        log(f"ML package import FAILED: {e}")

    log("STEP 4: Building dataset")
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")
    try:
        from finetune.build_dataset import build_dataset
        build_dataset()
        log("build_dataset COMPLETED")
    except Exception as e:
        log(f"build_dataset FAILED: {traceback.format_exc()}")

    log("STEP 5: Starting training")
    try:
        from finetune.train import train
        train()
        log("train COMPLETED")
    except Exception as e:
        log(f"train FAILED: {traceback.format_exc()}")

except Exception as e:
    log(f"FATAL: {traceback.format_exc()}")

log("UPLOADING LOG...")
upload_log()
log("DONE")
