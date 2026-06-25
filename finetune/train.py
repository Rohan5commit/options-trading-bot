"""
LoRA fine-tuning script for Llama-3-8B-Instruct on options trading data.
Uses QLoRA (4-bit quantization) for memory efficiency.
Designed to run on OVH Cloud V100S 32GB (t2-le-45) at $0.88/hr.
Target: ~$30-50 training cost with 100K examples over 10 epochs.
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
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
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_MODEL = os.environ.get("BASE_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./finetuned_models/options_llm")
TRAINING_DATA = os.environ.get("TRAINING_DATA", "./training_data/train.jsonl")
EVAL_DATA = os.environ.get("EVAL_DATA", "./training_data/test.jsonl")

# LoRA configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyperparameters
# 100K examples × 10 epochs on V100S ($0.88/hr) ≈ 50 hours ≈ $44
NUM_EPOCHS = 10
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
    from datasets import Dataset

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

    logger.info("Starting training")
    trainer.train()

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
        }, f, indent=2)

    logger.info("Training complete. Adapter saved to %s", OUTPUT_DIR)
    print(f"\nTraining complete! Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train()
