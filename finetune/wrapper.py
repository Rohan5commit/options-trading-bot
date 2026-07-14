#!/usr/bin/env python3
"""Wrapper that uploads a startup marker BEFORE train.py imports."""
import os, json, tempfile
from datetime import datetime

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "Rohan556/options-llm-checkpoints"

print(f"[{datetime.utcnow().isoformat()}] Wrapper started", flush=True)
print(f"HF_TOKEN set: {bool(HF_TOKEN)}", flush=True)
print(f"GPU_NAME: {os.environ.get('GPU_NAME', 'unknown')}", flush=True)

try:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, exist_ok=True, private=True)
    marker = {
        "status": "training_started",
        "gpu": os.environ.get("GPU_NAME", "unknown"),
        "budget": float(os.environ.get("TRAINING_BUDGET", "15.0")),
        "timestamp": datetime.utcnow().isoformat(),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(marker, f, indent=2)
        f.flush()
        api.upload_file(
            path_or_fileobj=f.name,
            repo_id=HF_REPO,
            path_in_repo="startup_marker.json",
        )
    print(f"[{datetime.utcnow().isoformat()}] Startup marker uploaded!", flush=True)
except Exception as e:
    print(f"[{datetime.utcnow().isoformat()}] MARKER UPLOAD FAILED: {e}", flush=True)

print(f"[{datetime.utcnow().isoformat()}] Now running train.py...", flush=True)

# Run train.py
import subprocess, sys
result = subprocess.run(
    [sys.executable, "-u", "finetune/train.py"],
    cwd="/workspace",
    env=os.environ.copy(),
)
print(f"[{datetime.utcnow().isoformat()}] train.py exited with code: {result.returncode}", flush=True)
