"""
LoRA fine-tuning script for Llama-3-8B-Instruct on options trading data.
Uses QLoRA (4-bit quantization) for memory efficiency.

Two-phase training plan:
- Phase 1: Lightning.ai L4 ($0.48/hr) — $15 budget = 31.25 hours
- Phase 2: Modal A10G ($1.10/hr) — $30 budget = 27.3 hours

Supports checkpointing for resuming across platforms:
- Saves checkpoints to HF Hub every 1000 steps
- Resumes from latest checkpoint on Modal
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import HfApi
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_MODEL = os.environ.get("BASE_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_CHECKPOINT_REPO", "Rohan5commit/options-llm-checkpoints")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./finetuned_models/options_llm")
TRAINING_DATA = os.environ.get("TRAINING_DATA", "./training_data/train.jsonl")
EVAL_DATA = os.environ.get("EVAL_DATA", "./training_data/test.jsonl")

# Cost tracking
GPU_PRICE_PER_HR = float(os.environ.get("GPU_PRICE_HR", "0.48"))  # L4 on Lightning
TRAINING_BUDGET = float(os.environ.get("TRAINING_BUDGET", "15.0"))  # Phase 1 budget

# LoRA configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyperparameters
# 160K examples × 8 epochs on L4 ($0.48/hr) ≈ 67 hours ≈ $32
# Phase 1 (Lightning): 31.25 hrs → ~3.75 epochs
# Phase 2 (Modal): 27.3 hrs → ~4.25 epochs
# Total: 8 epochs
NUM_EPOCHS = 8
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8  # effective batch = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 2048
LOGGING_STEPS = 50
SAVE_STEPS = 1000
EVAL_STEPS = 1000


def load_and_format_dataset(path: str) -> list[dict[str, str]]:
    """Load JSONL dataset and format for instruction tuning."""
    data = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)
    logger.info("Loaded %d examples from %s", len(data), path)
    return data


def format_prompt(example: dict[str, str]) -> str:
    """Format a single example into a chat prompt for Llama-3."""
    system_msg = "You are an expert options trader. Given market context, output a JSON trade decision."
    user_msg = (
        f"{example.get('instruction', '')}\n\n"
        f"Market Context:\n{example.get('input', '')}"
    )
    assistant_msg = example.get("output", "")

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return messages


def find_latest_checkpoint(output_dir: str, hf_repo: str | None = None) -> str | None:
    """Find latest checkpoint from local dir or HF Hub."""
    # Check local first
    local_dir = Path(output_dir)
    if local_dir.exists():
        checkpoints = sorted(local_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
        if checkpoints:
            logger.info("Found local checkpoint: %s", checkpoints[-1])
            return str(checkpoints[-1])

    # Check HF Hub
    if hf_repo and HF_TOKEN:
        try:
            api = HfApi(token=HF_TOKEN)
            models = api.list_models(search=hf_repo, sort="lastModified", direction=-1)
            for model in models:
                if model.id == hf_repo:
                    logger.info("Found HF checkpoint repo: %s", hf_repo)
                    return f"hf://{hf_repo}"
        except Exception as e:
            logger.warning("Could not check HF Hub: %s", e)

    return None


def save_checkpoint_to_hub(output_dir: str, hf_repo: str, step: int, epoch: float, cost: float) -> None:
    """Push checkpoint to HF Hub for persistence across platforms."""
    if not HF_TOKEN:
        logger.warning("No HF_TOKEN, skipping Hub upload")
        return

    try:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=hf_repo, exist_ok=True, private=True)

        # Upload checkpoint files
        checkpoint_path = Path(output_dir) / f"checkpoint-{step}"
        if checkpoint_path.exists():
            api.upload_folder(
                folder_path=str(checkpoint_path),
                repo_id=hf_repo,
                path_in_repo=f"checkpoint-{step}",
            )
            logger.info("Uploaded checkpoint-%d to %s", step, hf_repo)

        # Save metadata
        metadata = {
            "last_step": step,
            "last_epoch": epoch,
            "total_cost_usd": cost,
            "gpu_price_per_hr": GPU_PRICE_PER_HR,
            "timestamp": datetime.utcnow().isoformat(),
        }
        metadata_path = Path(output_dir) / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        api.upload_file(
            path_or_fileobj=str(metadata_path),
            repo_id=hf_repo,
            path_in_repo="training_metadata.json",
        )
        logger.info("Saved training metadata to Hub")
    except Exception as e:
        logger.error("Failed to upload checkpoint to Hub: %s", e)


def train():
    """Main training function."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading tokenizer from %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model with 4-bit quantization")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )

    model = prepare_model_for_kbit_training(model)

    logger.info("Applying LoRA configuration")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load datasets
    logger.info("Loading training data")
    train_data = load_and_format_dataset(TRAINING_DATA)
    eval_data = load_and_format_dataset(EVAL_DATA) if os.path.exists(EVAL_DATA) else None

    # Format data for SFT
    def formatting_func(examples):
        """Format batch of examples into tokenized prompts."""
        texts = []
        for i in range(len(examples["instruction"])):
            example = {
                "instruction": examples["instruction"][i],
                "input": examples["input"][i],
                "output": examples["output"][i],
            }
            messages = format_prompt(example)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return texts

    # Convert to HuggingFace datasets format
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None

    # Training arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=EVAL_STEPS if eval_dataset else None,
        save_total_limit=3,
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field=None,
        packing=False,
        resume_from_checkpoint=True,  # Auto-resume from latest checkpoint
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Check for existing checkpoint
    resume_checkpoint = find_latest_checkpoint(OUTPUT_DIR, HF_REPO)
    resume_from = None

    if resume_checkpoint:
        if resume_checkpoint.startswith("hf://"):
            hf_repo_path = resume_checkpoint.replace("hf://", "")
            logger.info("Resuming from HF Hub checkpoint: %s", hf_repo_path)
        else:
            logger.info("Resuming from local checkpoint: %s", resume_checkpoint)
            resume_from = resume_checkpoint

    logger.info("Starting training on %s GPU at $%s/hr", 
                os.environ.get("GPU_NAME", "unknown"), GPU_PRICE_PER_HR)
    start_time = datetime.utcnow()

    trainer.train(resume_from_checkpoint=resume_from)

    end_time = datetime.utcnow()
    duration_hours = (end_time - start_time).total_seconds() / 3600
    cost = duration_hours * GPU_PRICE_PER_HR

    # Save the LoRA adapter
    logger.info("Saving LoRA adapter to %s", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training config
    config_path = Path(OUTPUT_DIR) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "base_model": BASE_MODEL,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "trained_at": datetime.utcnow().isoformat(),
            "duration_hours": round(duration_hours, 2),
            "estimated_cost_usd": round(cost, 2),
            "gpu": os.environ.get("GPU_NAME", "unknown"),
            "gpu_price_per_hr": GPU_PRICE_PER_HR,
        }, f, indent=2)

    # Push final adapter to HF Hub
    if HF_TOKEN and HF_REPO:
        try:
            api = HfApi(token=HF_TOKEN)
            api.create_repo(repo_id=HF_REPO, exist_ok=True, private=True)
            api.upload_folder(
                folder_path=OUTPUT_DIR,
                repo_id=HF_REPO,
                path_in_repo="final",
            )
            logger.info("Pushed final adapter to %s", HF_REPO)
        except Exception as e:
            logger.error("Failed to push to Hub: %s", e)

    logger.info("Training complete! Duration: %.1f hrs, Cost: $%.2f", duration_hours, cost)
    print(f"\nTraining complete!")
    print(f"Duration: {duration_hours:.1f} hours")
    print(f"Estimated cost: ${cost:.2f}")
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print(f"Hub repo: {HF_REPO}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train()
