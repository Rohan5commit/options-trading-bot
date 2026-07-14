#!/usr/bin/env python3
"""Minimal test: import all packages and upload a marker."""
import json, tempfile, os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "Rohan556/options-llm-checkpoints"

print("START", flush=True)

# Test all imports
try:
    import torch
    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"torch FAIL: {e}", flush=True)

try:
    import transformers, peft, bitsandbytes, trl, datasets
    print(f"transformers: {transformers.__version__}", flush=True)
    print(f"peft: {peft.__version__}", flush=True)
    print(f"bitsandbytes: {bitsandbytes.__version__}", flush=True)
    print(f"trl: {trl.__version__}", flush=True)
    print(f"datasets: {datasets.__version__}", flush=True)
except Exception as e:
    print(f"ML import FAIL: {e}", flush=True)

# Test HF upload
try:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    user = api.whoami()
    print(f"HF user: {user.get('name')}", flush=True)
    api.create_repo(repo_id=HF_REPO, exist_ok=True, private=True)
    marker = {"status": "all_imports_ok", "torch": torch.__version__ if 'torch' in dir() else "N/A"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(marker, f, indent=2)
        f.flush()
        api.upload_file(path_or_fileobj=f.name, repo_id=HF_REPO, path_in_repo="import_test.json")
    print("HF UPLOAD OK", flush=True)
except Exception as e:
    print(f"HF UPLOAD FAIL: {e}", flush=True)

print("DONE", flush=True)
