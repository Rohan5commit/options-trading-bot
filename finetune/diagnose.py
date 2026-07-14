#!/usr/bin/env python3
"""Diagnostic script that tests each step and writes results to HF Hub."""
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "Rohan556/options-llm-checkpoints"
results = {}

def log_result(step, status, detail=""):
    results[step] = {"status": status, "detail": detail, "time": datetime.utcnow().isoformat()}
    print(f"[{status}] {step}: {detail}")

def upload_results():
    if not HF_TOKEN:
        print("No HF_TOKEN, skipping upload")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO, exist_ok=True, private=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f, indent=2)
            api.upload_file(path_or_fileobj=f.name, repo_id=HF_REPO, path_in_repo="diagnostic_results.json")
        print("Results uploaded to HF Hub")
    except Exception as e:
        print(f"Upload failed: {e}")

try:
    # Step 1: pip packages
    log_result("pip_install", "running")
    os.system("pip install -q transformers peft bitsandbytes accelerate datasets huggingface-hub trl scipy sentencepiece protobuf")
    log_result("pip_install", "done")

    # Step 2: Download repo
    log_result("download", "running")
    os.system("curl -sL https://github.com/Rohan5commit/options-trading-bot/archive/refs/heads/main.tar.gz -o /tmp/repo.tar.gz")
    os.system("mkdir -p /workspace && tar xzf /tmp/repo.tar.gz --strip-components=1 -C /workspace")
    log_result("download", "done")

    # Step 3: Check imports
    log_result("check_imports", "running")
    try:
        import torch
        log_result("check_imports", "ok", f"CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    except Exception as e:
        log_result("check_imports", "error", str(e))

    # Step 4: HF token check
    log_result("hf_token", "running")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        user = api.whoami()
        log_result("hf_token", "ok", f"User: {user.get('name', 'unknown')}")
    except Exception as e:
        log_result("hf_token", "error", str(e))

    # Step 5: Build dataset
    log_result("build_dataset", "running")
    try:
        sys.path.insert(0, "/workspace")
        os.chdir("/workspace")
        from finetune.build_dataset import build_dataset
        build_dataset()
        log_result("build_dataset", "done")
    except Exception as e:
        log_result("build_dataset", "error", traceback.format_exc())

    # Step 6: Load model
    log_result("load_model", "running")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import torch
        
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)
        log_result("tokenizer", "ok")
        
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                                      quantization_config=bnb_config,
                                                      device_map="auto", token=HF_TOKEN)
        log_result("load_model", "ok", f"Model loaded, device={model.device}")
    except Exception as e:
        log_result("load_model", "error", traceback.format_exc())

    # Step 7: Upload marker
    log_result("upload_marker", "running")
    try:
        api = HfApi(token=HF_TOKEN)
        marker = {"status": "all_steps_complete", "timestamp": datetime.utcnow().isoformat()}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(marker, f)
            api.upload_file(path_or_fileobj=f.name, repo_id=HF_REPO, path_in_repo="diagnostic_results.json")
        log_result("upload_marker", "ok")
    except Exception as e:
        log_result("upload_marker", "error", str(e))

except Exception as e:
    log_result("unexpected_error", "error", traceback.format_exc())

upload_results()
print("\n=== DIAGNOSTIC SUMMARY ===")
for step, info in results.items():
    print(f"  {step}: {info['status']}")
