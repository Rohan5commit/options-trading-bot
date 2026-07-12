"""
Lightning.ai + Modal training helper.
Two-phase training:
- Phase 1: Lightning.ai L4 ($0.48/hr) — $15 budget
- Phase 2: Modal A10G ($1.10/hr) — $30 budget

Usage:
    python finetune/lightning_helper.py status      # Check training status
    python finetune/lightning_helper.py phase2       # Get Modal phase 2 instructions
    python finetune/lightning_helper.py reset        # Reset for new platform
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

# Phase 1: Lightning L4
LIGHTNING_GPU_PRICE = 0.48
LIGHTNING_BUDGET = 15.0

# Phase 2: Modal A10G
MODAL_GPU_PRICE = 1.10
MODAL_BUDGET = 30.0

TOTAL_BUDGET = LIGHTNING_BUDGET + MODAL_BUDGET


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
    print("=" * 60)
    print("  Options LLM Training Status")
    print("=" * 60)
    print()

    # Training plan
    print("Training Plan:")
    print("  Phase 1: Lightning.ai L4 @ $0.48/hr — $15 budget (31.25 hrs)")
    print("  Phase 2: Modal A10G @ $1.10/hr — $30 budget (27.3 hrs)")
    print(f"  Total budget: ${TOTAL_BUDGET:.2f}")
    print()

    # Local metadata
    local = get_local_metadata()
    if local:
        print("Local checkpoint:")
        print(f"  Last step: {local.get('last_step', 'unknown')}")
        print(f"  Last epoch: {local.get('last_epoch', 'unknown')}")
        print(f"  Cost so far: ${local.get('total_cost_usd', 0):.2f}")
        print(f"  Platform: {local.get('platform', 'unknown')}")
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
        print(f"  Platform: {hub.get('platform', 'unknown')}")
        print()
    else:
        print("No Hub checkpoint found.\n")

    # Budget summary
    cost = 0.0
    if hub and "total_cost_usd" in hub:
        cost = hub["total_cost_usd"]
    elif local and "total_cost_usd" in local:
        cost = local["total_cost_usd"]

    remaining = TOTAL_BUDGET - cost
    
    # Determine current phase
    if cost < LIGHTNING_BUDGET:
        current_phase = "Phase 1 (Lightning L4)"
        phase_remaining = LIGHTNING_BUDGET - cost
        phase_hours = phase_remaining / LIGHTNING_GPU_PRICE
    else:
        current_phase = "Phase 2 (Modal A10G)"
        phase_remaining = cost - LIGHTNING_BUDGET
        phase_remaining = MODAL_BUDGET - phase_remaining
        phase_hours = phase_remaining / MODAL_GPU_PRICE

    print("Budget:")
    print(f"  Total budget: ${TOTAL_BUDGET:.2f}")
    print(f"  Spent: ${cost:.2f}")
    print(f"  Remaining: ${remaining:.2f}")
    print(f"  Current phase: {current_phase}")
    print(f"  Phase hours remaining: {phase_hours:.1f} hrs")
    print()

    # Training estimate
    print("Training estimate (160K examples, 8 epochs):")
    print(f"  ~67 hours total on L4")
    print(f"  ~$32 total cost")
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


def show_phase2_instructions():
    """Show instructions for Modal Phase 2."""
    print("=" * 60)
    print("  Modal Phase 2 Training Instructions")
    print("=" * 60)
    print()
    print("After exhausting $15 on Lightning.ai, switch to Modal for Phase 2.")
    print()
    print("SETUP:")
    print("1. Go to https://modal.com")
    print("2. Create account")
    print("3. Install Modal: pip install modal")
    print("4. Authenticate: modal setup")
    print()
    print("RUN PHASE 2:")
    print("1. Clone repo (if not already):")
    print("   git clone https://github.com/Rohan5commit/options-trading-bot.git")
    print("   cd options-trading-bot")
    print()
    print("2. Create Modal secret:")
    print("   modal secret create options-training \\")
    print("     HF_TOKEN=your_hf_token \\")
    print("     HF_CHECKPOINT_REPO=Rohan5commit/options-llm-checkpoints")
    print()
    print("3. Deploy training:")
    print("   modal deploy finetune/modal_train.py")
    print()
    print("4. Or run directly:")
    print("   modal run finetune/modal_train.py")
    print()
    print("Checkpoints resume automatically from HF Hub.")
    print()
    print("BUDGET: $30 on Modal A10G @ $1.10/hr = 27.3 hours")
    print("=" * 60)


def reset_for_platform():
    """Reset local state for switching platforms."""
    print("=" * 60)
    print("  Resetting for Platform Switch")
    print("=" * 60)
    print()

    # Remove local checkpoints (they're on Hub)
    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        checkpoints = list(output_path.glob("checkpoint-*"))
        for ckpt in checkpoints:
            print(f"Removing local checkpoint: {ckpt.name}")
            import shutil
            shutil.rmtree(ckpt)
        print(f"Removed {len(checkpoints)} local checkpoints.\n")

    print("Done! Your checkpoints are safe on HuggingFace Hub.")
    print()
    print("Next steps:")
    print("1. Close your Lightning.ai Studio")
    print("2. Follow Phase 2 instructions: python finetune/lightning_helper.py phase2")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training helper")
    parser.add_argument("action", choices=["status", "phase2", "reset"],
                       help="Action to perform")
    args = parser.parse_args()

    if args.action == "status":
        show_status()
    elif args.action == "phase2":
        show_phase2_instructions()
    elif args.action == "reset":
        reset_for_platform()
