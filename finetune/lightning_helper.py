"""
Lightning.ai training helper.
Provides account switching, budget tracking, and checkpoint management.

Usage:
    python finetune/lightning_helper.py status      # Check current training status
    python finetune/lightning_helper.py resume       # Resume from latest checkpoint
    python finetune/lightning_helper.py reset        # Reset for new account
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./finetuned_models/options_llm")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_CHECKPOINT_REPO", "Rohan5commit/options-llm-checkpoints")
GPU_PRICE_HR = float(os.environ.get("GPU_PRICE_HR", "0.60"))
MAX_BUDGET = float(os.environ.get("MAX_BUDGET", "45.0"))


def get_hf_metadata() -> dict | None:
    """Fetch training metadata from HF Hub."""
    if not HF_TOKEN:
        print("No HF_TOKEN set. Cannot check Hub status.")
        return None

    try:
        url = f"https://huggingface.co/api/models/{HF_REPO}"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            # Look for training_metadata.json in siblings
            model_info = resp.json()
            for sibling in model_info.get("siblings", []):
                if sibling.get("rfilename") == "training_metadata.json":
                    raw_url = f"https://huggingface.co/{HF_REPO}/raw/main/training_metadata.json"
                    raw_resp = requests.get(raw_url, headers=headers)
                    if raw_resp.status_code == 200:
                        return json.loads(raw_resp.text)
        return None
    except Exception as e:
        print(f"Error fetching from Hub: {e}")
        return None


def get_local_metadata() -> dict | None:
    """Read local training metadata."""
    metadata_path = Path(OUTPUT_DIR) / "training_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def show_status():
    """Display current training status and budget info."""
    print("=== Options LLM Training Status ===\n")

    # Local metadata
    local = get_local_metadata()
    if local:
        print("Local checkpoint:")
        print(f"  Last step: {local.get('last_step', 'unknown')}")
        print(f"  Last epoch: {local.get('last_epoch', 'unknown')}")
        print(f"  Cost so far: ${local.get('total_cost_usd', 0):.2f}")
        print()
    else:
        print("No local checkpoint found.\n")

    # Hub metadata
    hub = get_hf_metadata()
    if hub:
        print("Hub checkpoint:")
        print(f"  Last step: {hub.get('last_step', 'unknown')}")
        print(f"  Last epoch: {hub.get('last_epoch', 'unknown')}")
        print(f"  Cost so far: ${hub.get('total_cost_usd', 0):.2f}")
        print()
    else:
        print("No Hub checkpoint found.\n")

    # Budget summary
    cost = 0.0
    if hub and "total_cost_usd" in hub:
        cost = hub["total_cost_usd"]
    elif local and "total_cost_usd" in local:
        cost = local["total_cost_usd"]

    remaining = MAX_BUDGET - cost
    hours_left = remaining / GPU_PRICE_HR if GPU_PRICE_HR > 0 else 0

    print("Budget:")
    print(f"  Total budget: ${MAX_BUDGET:.2f}")
    print(f"  Spent: ${cost:.2f}")
    print(f"  Remaining: ${remaining:.2f}")
    print(f"  GPU: L4 at ${GPU_PRICE_HR}/hr")
    print(f"  Hours remaining: {hours_left:.1f} hrs")
    print()

    # Estimate for full training
    print("Full training estimate (100K examples, 10 epochs):")
    print(f"  ~50 hours on L4")
    print(f"  ~$30 total")
    print()

    # Check for local checkpoints
    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        checkpoints = sorted(output_path.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
        if checkpoints:
            print(f"Local checkpoints: {len(checkpoints)}")
            for ckpt in checkpoints[-3:]:
                print(f"  {ckpt.name}")
        else:
            print("No local checkpoints found.")
    else:
        print("Output directory does not exist yet.")


def reset_for_new_account():
    """Reset local state for switching to a new Lightning.ai account."""
    print("=== Resetting for New Account ===\n")

    # Remove local checkpoints (they're on Hub)
    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        checkpoints = list(output_path.glob("checkpoint-*"))
        for ckpt in checkpoints:
            print(f"Removing local checkpoint: {ckpt.name}")
            import shutil
            shutil.rmtree(ckpt)
        print(f"Removed {len(checkpoints)} local checkpoints.\n")

    print("Done! Steps:")
    print("1. Open new Lightning.ai Studio with L4 GPU")
    print("2. Clone repo: git clone https://github.com/Rohan5commit/options-trading-bot.git")
    print("3. Set HF_TOKEN: export HF_TOKEN=your_token")
    print("4. Run: bash finetune/run_lightning.sh")
    print("5. Training resumes from latest Hub checkpoint automatically")


def show_checkpoint_chain():
    """Show the chain of checkpoints on HF Hub."""
    if not HF_TOKEN:
        print("No HF_TOKEN set.")
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)

        print("=== Checkpoint Chain on HF Hub ===\n")

        # List all files in the repo
        files = list(api.list_repo_tree(HF_REPO, recursive=False))
        checkpoints = set()
        for f in files:
            if hasattr(f, 'path') and f.path.startswith("checkpoint-"):
                checkpoints.add(f.path)

        if checkpoints:
            for ckpt in sorted(checkpoints, key=lambda x: int(x.split("-")[1])):
                print(f"  {ckpt}")
        else:
            print("No checkpoints found on Hub.")

        # Check for metadata
        try:
            raw_url = f"https://huggingface.co/{HF_REPO}/raw/main/training_metadata.json"
            resp = requests.get(raw_url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
            if resp.status_code == 200:
                meta = json.loads(resp.text)
                print(f"\nLatest metadata:")
                print(f"  Step: {meta.get('last_step')}")
                print(f"  Epoch: {meta.get('last_epoch')}")
                print(f"  Cost: ${meta.get('total_cost_usd', 0):.2f}")
                print(f"  Time: {meta.get('timestamp')}")
        except Exception:
            pass

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightning.ai training helper")
    parser.add_argument("action", choices=["status", "resume", "reset", "checkpoints"],
                       help="Action to perform")
    args = parser.parse_args()

    if args.action == "status":
        show_status()
    elif args.action == "resume":
        print("To resume training:")
        print("1. Open Lightning.ai Studio with L4 GPU")
        print("2. Clone repo and set HF_TOKEN")
        print("3. Run: bash finetune/run_lightning.sh")
        print("Training auto-resumes from latest Hub checkpoint.")
    elif args.action == "reset":
        reset_for_new_account()
    elif args.action == "checkpoints":
        show_checkpoint_chain()
