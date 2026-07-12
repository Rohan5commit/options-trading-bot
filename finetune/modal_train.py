"""
Modal GPU training script for Phase 2.
Resumes from Lightning.ai checkpoint and trains on Modal A10G ($1.10/hr).

Usage:
    modal deploy finetune/modal_train.py
    
Or run directly:
    modal run finetune/modal_train.py
"""
import modal

app = modal.App("options-llm-trainer")

# Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.25.0",
        "datasets>=2.16.0",
        "huggingface-hub>=0.20.0",
        "trl>=0.7.0",
        "scipy",
        "sentencepiece",
        "protobuf",
    )
    .apt_install("git")
)

# Persistent volume for checkpoints
vol = modal.Volume("options-training-vol")

@app.function(
    image=image,
    gpu=modal.gpu.A10G(),  # A10G at $1.10/hr
    timeout=24 * 60 * 60,  # 24 hour timeout
    volumes={"/vol": vol},
    secrets=[
        modal.Secret.from_dict({
            "HF_TOKEN": "",  # Set via: modal secret create options-training HF_TOKEN=your_token
            "HF_CHECKPOINT_REPO": "Rohan5commit/options-llm-checkpoints",
            "GPU_PRICE_HR": "1.10",
            "TRAINING_BUDGET": "30.0",
            "GPU_NAME": "A10G",
        }),
    ],
)
def train():
    """Run training on Modal A10G GPU."""
    import json
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

    # Configuration
    BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    HF_REPO = os.environ.get("HF_CHECKPOINT_REPO", "Rohan5commit/options-llm-checkpoints")
    OUTPUT_DIR = "/vol/finetuned_models/options_llm"
    TRAINING_DATA = "/vol/training_data/train.jsonl"
    EVAL_DATA = "/vol/training_data/test.jsonl"
    GPU_PRICE_PER_HR = float(os.environ.get("GPU_PRICE_HR", "1.10"))
    
    # LoRA configuration
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # Training hyperparameters
    NUM_EPOCHS = 8
    PER_DEVICE_BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_SEQ_LENGTH = 2048
    LOGGING_STEPS = 50
    SAVE_STEPS = 1000
    EVAL_STEPS = 1000

    print("=== Modal Phase 2 Training ===")
    print(f"GPU: A10G at ${GPU_PRICE_PER_HR}/hr")
    print(f"Budget: $30")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    print("Loading model with QLoRA...")
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

    # Apply LoRA
    print("Applying LoRA...")
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
    print("Loading training data...")
    def load_jsonl(path):
        data = []
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        return data

    train_data = load_jsonl(TRAINING_DATA)
    eval_data = load_jsonl(EVAL_DATA) if os.path.exists(EVAL_DATA) else None
    print(f"Loaded {len(train_data)} training examples")

    # Format data
    def format_prompt(example):
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

    def formatting_func(examples):
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
        resume_from_checkpoint=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Check for existing checkpoint on HF Hub
    resume_from = None
    if HF_TOKEN and HF_REPO:
        try:
            api = HfApi(token=HF_TOKEN)
            # Download latest checkpoint from Hub
            print("Checking for existing checkpoint on Hub...")
            # TODO: Implement checkpoint download from Hub
        except Exception as e:
            print(f"Could not check Hub: {e}")

    # Start training
    print("\nStarting training...")
    start_time = datetime.utcnow()

    trainer.train(resume_from_checkpoint=resume_from)

    end_time = datetime.utcnow()
    duration_hours = (end_time - start_time).total_seconds() / 3600
    cost = duration_hours * GPU_PRICE_PER_HR

    # Save adapter
    print("Saving adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save config
    config_path = Path(OUTPUT_DIR) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "base_model": BASE_MODEL,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "trained_at": datetime.utcnow().isoformat(),
            "duration_hours": round(duration_hours, 2),
            "estimated_cost_usd": round(cost, 2),
            "gpu": "A10G",
            "gpu_price_per_hr": GPU_PRICE_PER_HR,
            "platform": "Modal",
        }, f, indent=2)

    # Push to HF Hub
    if HF_TOKEN and HF_REPO:
        try:
            api = HfApi(token=HF_TOKEN)
            api.create_repo(repo_id=HF_REPO, exist_ok=True, private=True)
            api.upload_folder(
                folder_path=OUTPUT_DIR,
                repo_id=HF_REPO,
                path_in_repo="final",
            )
            print(f"Pushed final adapter to {HF_REPO}")
        except Exception as e:
            print(f"Failed to push to Hub: {e}")

    print(f"\nTraining complete!")
    print(f"Duration: {duration_hours:.1f} hours")
    print(f"Cost: ${cost:.2f}")
    print(f"Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    app.run()
