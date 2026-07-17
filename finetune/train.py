"""
LoRA fine-tuning script for Llama-3-8B-Instruct on options trading data.
Uses QLoRA (4-bit quantization) for memory efficiency.

Two-phase training plan:
- Phase 1: Lightning.ai RTXP 6000 ($2.11/hr) — $15 budget
- Phase 2: Modal A10G ($1.10/hr) — $30 budget

Features:
- Auto batch size detection (targets ~90% GPU utilization)
- Flash Attention 2 for faster training
- Sequence packing for higher throughput
- Checkpoint uploads to HF Hub every ~10 min for crash recovery
- Auto-resume from latest HF Hub checkpoint
"""
import gc
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
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from auto_batch_size import find_optimal_batch_size

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_MODEL = os.environ.get("BASE_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_CHECKPOINT_REPO", "Rohan556/options-llm-checkpoints")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./finetuned_models/options_llm")
TRAINING_DATA = os.environ.get("TRAINING_DATA", "./training_data/train.jsonl")
EVAL_DATA = os.environ.get("EVAL_DATA", "./training_data/test.jsonl")

# Cost tracking
GPU_PRICE_PER_HR = float(os.environ.get("GPU_PRICE_HR", "2.11"))  # RTXP 6000 interruptible
TRAINING_BUDGET = float(os.environ.get("TRAINING_BUDGET", "15.0"))  # Phase 1 budget

# LoRA configuration (Llama-3-8B)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0  # 0 is optimized in PEFT
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training hyperparameters
NUM_EPOCHS = 8
# Batch size is auto-detected — these are fallback defaults
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 2048
LOGGING_STEPS = 10
SAVE_STEPS = 200  # ~10 min between checkpoints
EVAL_STEPS = 200
TARGET_EFFECTIVE_BATCH_SIZE = 64


# ── Hub Upload Callback ────────────────────────────────────────────────────────

class HubUploadCallback(TrainerCallback):
    """Upload checkpoints to HF Hub after each save for crash recovery."""

    def __init__(self, output_dir: str, hf_repo: str, gpu_price_per_hr: float):
        self.output_dir = output_dir
        self.hf_repo = hf_repo
        self.gpu_price_per_hr = gpu_price_per_hr
        self.start_time = None
        self.cost = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.utcnow()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track cost from logged metrics."""
        if self.start_time and state.global_step > 0:
            elapsed = (datetime.utcnow() - self.start_time).total_seconds() / 3600
            self.cost = elapsed * self.gpu_price_per_hr

    def on_save(self, args, state, control, **kwargs):
        """Upload checkpoint to HF Hub after each save."""
        step = state.global_step
        epoch = state.epoch if state.epoch else 0.0

        if self.start_time:
            elapsed = (datetime.utcnow() - self.start_time).total_seconds() / 3600
            self.cost = elapsed * self.gpu_price_per_hr

        logger.info("Uploading checkpoint-%d to HF Hub (step %d, epoch %.2f, cost $%.2f)...",
                     step, step, epoch, self.cost)

        try:
            api = HfApi(token=HF_TOKEN)
            api.create_repo(repo_id=self.hf_repo, exist_ok=True, private=True)

            # Upload checkpoint folder
            checkpoint_path = Path(self.output_dir) / f"checkpoint-{step}"
            if checkpoint_path.exists():
                api.upload_folder(
                    folder_path=str(checkpoint_path),
                    repo_id=self.hf_repo,
                    path_in_repo=f"checkpoint-{step}",
                )
                logger.info("Uploaded checkpoint-%d to %s", step, self.hf_repo)

            # Save and upload metadata
            metadata = {
                "last_step": step,
                "last_epoch": epoch,
                "total_cost_usd": round(self.cost, 2),
                "gpu_price_per_hr": self.gpu_price_per_hr,
                "gpu": os.environ.get("GPU_NAME", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
            }
            metadata_path = Path(self.output_dir) / "training_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            api.upload_file(
                path_or_fileobj=str(metadata_path),
                repo_id=self.hf_repo,
                path_in_repo="training_metadata.json",
            )
            logger.info("Saved training metadata to Hub")

        except Exception as e:
            logger.error("Failed to upload checkpoint to Hub: %s", e)


# ── Helper Functions ───────────────────────────────────────────────────────────

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


def upload_startup_marker():
    """Upload a startup marker to HF Hub so we can verify training started."""
    if not HF_TOKEN or not HF_REPO:
        return
    try:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO, exist_ok=True, private=True)
        import tempfile, json as _json
        marker = {"status": "training_started", "gpu": os.environ.get("GPU_NAME", "unknown"),
                  "budget": TRAINING_BUDGET, "timestamp": datetime.utcnow().isoformat()}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            _json.dump(marker, f, indent=2)
            api.upload_file(path_or_fileobj=f.name, repo_id=HF_REPO, path_in_repo="startup_marker.json")
        logger.info("Startup marker uploaded to %s", HF_REPO)
    except Exception as e:
        logger.warning("Could not upload startup marker: %s", e)


# ── Main Training Function ─────────────────────────────────────────────────────

def train():
    """Main training function with GPU-optimized settings."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    upload_startup_marker()

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
        bnb_4bit_compute_dtype=torch.bfloat16,  # Match bf16=True in training args
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN or None,
        attn_implementation="flash_attention_2",  # Enable Flash Attention 2
    )

    model = prepare_model_for_kbit_training(model)

    logger.info("Applying LoRA configuration: r=%d, alpha=%d, modules=%s",
                LORA_R, LORA_ALPHA, TARGET_MODULES)
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

    # Pre-format all data into text strings
    def format_all(data):
        texts = []
        for entry in data:
            messages = format_prompt(entry)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return texts

    train_texts = format_all(train_data)
    eval_texts = format_all(eval_data) if eval_data else None

    # Convert to HuggingFace datasets format with "text" column
    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts}) if eval_texts else None

    # ── Auto-detect optimal batch size ─────────────────────────────────────────
    logger.info("Auto-detecting optimal batch size for ~90%% GPU utilization...")
    batch_rec = find_optimal_batch_size(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        target_effective_batch_size=TARGET_EFFECTIVE_BATCH_SIZE,
        target_utilization=0.9,
        use_gradient_checkpointing=True,
        lora_rank=LORA_R,
        num_target_modules=len(TARGET_MODULES),
    )

    per_device_bs = batch_rec.per_device_batch_size
    grad_accum = batch_rec.gradient_accumulation_steps

    logger.info("=" * 60)
    logger.info("GPU: %s (%.1f GB)", batch_rec.gpu_name, batch_rec.total_vram_gb)
    logger.info("Auto-detected: batch_size=%d, grad_accum=%d, effective_bs=%d",
                per_device_bs, grad_accum, batch_rec.effective_batch_size)
    logger.info("Estimated utilization: %.1f%%", batch_rec.estimated_utilization * 100)
    logger.info("Static memory: %.2f GB", batch_rec.static_memory_gb)
    logger.info("Activation per sample: %.4f GB", batch_rec.activation_per_sample_gb)
    logger.info("=" * 60)

    # ── Training arguments (GPU-optimized) ─────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=True,
        optim="paged_adamw_8bit",  # 8-bit paged optimizer saves ~4GB VRAM
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=EVAL_STEPS if eval_dataset else None,
        save_total_limit=3,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        packing=True,  # Enable sequence packing (requires FA2)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        resume_from_checkpoint=True,
        use_liger_kernel=True,  # Fused Triton kernels for 10-20% speedup
    )

    # ── Hub upload callback ─────────────────────────────────────────────────────
    hub_callback = HubUploadCallback(
        output_dir=OUTPUT_DIR,
        hf_repo=HF_REPO,
        gpu_price_per_hr=GPU_PRICE_PER_HR,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[hub_callback],
    )

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
            "batch_size": per_device_bs,
            "gradient_accumulation": grad_accum,
            "effective_batch_size": batch_rec.effective_batch_size,
            "gpu_utilization": batch_rec.estimated_utilization,
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
    print(f"GPU: {batch_rec.gpu_name} ({batch_rec.total_vram_gb:.1f} GB)")
    print(f"Batch size: {per_device_bs} x {grad_accum} = {batch_rec.effective_batch_size} effective")
    print(f"GPU utilization: {batch_rec.estimated_utilization:.1%}")
    print(f"Duration: {duration_hours:.1f} hours")
    print(f"Estimated cost: ${cost:.2f}")
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print(f"Hub repo: {HF_REPO}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    upload_startup_marker()
    train()
