"""Automatic batch size detection for QLoRA training.

Detects GPU VRAM and calculates optimal batch size targeting ~90% utilization.
Works with any GPU (L4, RTXP 6000, A100, H100, etc.).
"""

import gc
import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class BatchSizeRecommendation:
    """Result of automatic batch size detection."""
    per_device_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    estimated_utilization: float
    gpu_name: str
    total_vram_gb: float
    static_memory_gb: float
    activation_per_sample_gb: float


def _nearest_power_of_2(n: int) -> int:
    """Round down to nearest power of 2 for GPU tensor core efficiency."""
    if n <= 0:
        return 1
    p = 1
    while p * 2 <= n:
        p *= 2
    return max(1, p)


def auto_batch_size(
    model,
    tokenizer,
    max_seq_length: int = 2048,
    target_effective_batch_size: int = 64,
    target_utilization: float = 0.9,
    use_gradient_checkpointing: bool = True,
    lora_rank: int = 16,
    num_target_modules: int = 4,
    device: int = 0,
) -> BatchSizeRecommendation:
    """Automatically calculate optimal batch size for QLoRA training.

    Uses formula-based estimation to find the batch size that targets
    ~90% GPU VRAM utilization, then validates with a single test step.

    Args:
        model: The PEFT/LoRA-wrapped model.
        tokenizer: The tokenizer (used for dummy input generation).
        max_seq_length: Maximum sequence length for training.
        target_effective_batch_size: Desired effective batch size (bs * grad_accum).
        target_utilization: Target GPU VRAM utilization (0.0 - 1.0).
        use_gradient_checkpointing: Whether gradient checkpointing is enabled.
        lora_rank: LoRA rank (r) for parameter estimation.
        num_target_modules: Number of LoRA target modules.
        device: CUDA device index.

    Returns:
        BatchSizeRecommendation with optimal settings.
    """
    # Step 1: Detect GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot auto-detect batch size.")

    gpu_name = torch.cuda.get_device_name(device)
    free_vram, total_vram = torch.cuda.mem_get_info(device)
    target_vram = total_vram * target_utilization

    logger.info("GPU: %s (%.1f GB total, %.1f free)", gpu_name, total_vram / 1e9, free_vram / 1e9)

    # Step 2: Estimate static memory (model + optimizer + gradients)
    num_params = sum(p.numel() for p in model.parameters())
    adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # NF4 base model: ~0.5 bytes/param (4-bit quantization)
    base_model_gb = (num_params * 0.5) / 1e9

    # LoRA adapters: BF16 = 2 bytes/param
    adapter_gb = (adapter_params * 2) / 1e9

    # AdamW optimizer: 8 bytes/adapter_param (fp32 momentum + variance)
    optimizer_gb = (adapter_params * 8) / 1e9

    # Gradients: BF16 = 2 bytes/adapter_param
    gradients_gb = (adapter_params * 2) / 1e9

    # CUDA framework overhead
    overhead_gb = 1.0

    static_memory_gb = base_model_gb + adapter_gb + optimizer_gb + gradients_gb + overhead_gb

    logger.info("Static memory: %.2f GB (model=%.2f, adapter=%.3f, optimizer=%.3f, grads=%.3f)",
                static_memory_gb, base_model_gb, adapter_gb, optimizer_gb, gradients_gb)

    # Step 3: Estimate activation memory per sample
    # With gradient checkpointing, only ~35% of activations are stored
    checkpoint_factor = 0.35 if use_gradient_checkpointing else 1.0

    # Llama-3-8B architecture: hidden_size=4096, num_layers=32
    hidden_size = 4096
    num_layers = 32

    # Activation per sample (simplified transformer formula)
    # Each layer stores: attention + FFN activations
    # With gradient checkpointing, ~34x the hidden_size per token per layer
    activation_per_sample_gb = (
        num_layers * max_seq_length * hidden_size * 2 * 34 * checkpoint_factor / 1e9
    )

    # Add attention overhead (O(seq_len^2) for self-attention)
    attention_per_sample_gb = (
        num_layers * max_seq_length ** 2 * 4 * 2 * checkpoint_factor / 1e9
    )

    total_activation_per_sample_gb = activation_per_sample_gb + attention_per_sample_gb

    logger.info("Activation per sample: %.4f GB (at seq_len=%d, checkpoint=%s)",
                total_activation_per_sample_gb, max_seq_length, use_gradient_checkpointing)

    # Step 4: Calculate max batch size
    available_for_activations_gb = (target_vram / 1e9) - static_memory_gb

    if available_for_activations_gb <= 0:
        raise ValueError(
            f"Model + optimizer ({static_memory_gb:.2f} GB) exceeds "
            f"target VRAM ({target_vram / 1e9:.2f} GB). "
            f"Reduce LoRA rank, use fewer target modules, or use a larger GPU."
        )

    max_batch = max(1, int(available_for_activations_gb / total_activation_per_sample_gb))

    # Round to nearest power of 2 for GPU tensor core efficiency
    per_device_batch_size = _nearest_power_of_2(max_batch)

    # Step 5: Calculate gradient accumulation for target effective batch size
    grad_accum = max(1, target_effective_batch_size // per_device_batch_size)
    effective_bs = per_device_batch_size * grad_accum

    # Estimate final utilization
    estimated_vram_gb = static_memory_gb + (total_activation_per_sample_gb * per_device_batch_size)
    estimated_util = estimated_vram_gb / (total_vram / 1e9)

    rec = BatchSizeRecommendation(
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        effective_batch_size=effective_bs,
        estimated_utilization=estimated_util,
        gpu_name=gpu_name,
        total_vram_gb=total_vram / 1e9,
        static_memory_gb=static_memory_gb,
        activation_per_sample_gb=total_activation_per_sample_gb,
    )

    logger.info("Recommended: batch_size=%d, grad_accum=%d, effective_bs=%d, utilization=%.1f%%",
                per_device_batch_size, grad_accum, effective_bs, estimated_util * 100)

    return rec


def validate_batch_size(
    model,
    tokenizer,
    batch_size: int,
    max_seq_length: int,
    device: int = 0,
) -> dict:
    """Run one training step to empirically validate batch size fits in VRAM.

    Args:
        model: The PEFT/LoRA-wrapped model.
        tokenizer: The tokenizer.
        batch_size: Batch size to validate.
        max_seq_length: Maximum sequence length.
        device: CUDA device index.

    Returns:
        Dict with peak_vram_gb, total_vram_gb, utilization, and fits (bool).
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Create dummy batch with realistic token distribution
    dummy_texts = [
        "You are an expert options trader. Given market context, output a JSON trade decision. "
        "Market Context: SPY 520.50, VIX 15.2, AAPL 195.30"
        for _ in range(batch_size)
    ]

    dummy_input = tokenizer(
        dummy_texts,
        return_tensors="pt",
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
    ).to(device)

    # Forward pass
    outputs = model(**dummy_input, labels=dummy_input["input_ids"])
    loss = outputs.loss

    # Backward pass
    loss.backward()

    peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
    free_vram, total_vram = torch.cuda.mem_get_info(device)

    # Cleanup
    del outputs, loss, dummy_input
    model.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "peak_vram_gb": peak_memory,
        "total_vram_gb": total_vram / 1e9,
        "utilization": peak_memory / (total_vram / 1e9),
        "fits": peak_memory < (total_vram / 1e9) * 0.95,
    }

    logger.info("Validation: peak=%.2f GB, utilization=%.1f%%, fits=%s",
                peak_memory, result["utilization"] * 100, result["fits"])

    return result


def find_optimal_batch_size(
    model,
    tokenizer,
    max_seq_length: int = 2048,
    target_effective_batch_size: int = 64,
    target_utilization: float = 0.9,
    use_gradient_checkpointing: bool = True,
    lora_rank: int = 16,
    num_target_modules: int = 4,
    device: int = 0,
) -> BatchSizeRecommendation:
    """Find optimal batch size with empirical validation and fallback.

    This is the main entry point. It:
    1. Estimates batch size via formula
    2. Validates with a single forward+backward pass
    3. Falls back (halve batch, double grad_accum) if validation fails

    Returns:
        BatchSizeRecommendation with validated settings.
    """
    # Phase 1: Formula-based estimate
    rec = auto_batch_size(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        target_effective_batch_size=target_effective_batch_size,
        target_utilization=target_utilization,
        use_gradient_checkpointing=use_gradient_checkpointing,
        lora_rank=lora_rank,
        num_target_modules=num_target_modules,
        device=device,
    )

    logger.info("Phase 1 estimate: batch=%d, grad_accum=%d, effective=%d",
                rec.per_device_batch_size, rec.gradient_accumulation_steps, rec.effective_batch_size)

    # Phase 2: Empirical validation
    result = validate_batch_size(model, tokenizer, rec.per_device_batch_size, max_seq_length, device)

    if result["fits"]:
        logger.info("Phase 1 estimate validated! Using batch_size=%d", rec.per_device_batch_size)
        return rec

    # Phase 3: Fallback — halve batch size, double gradient accumulation
    logger.warning("Phase 1 estimate failed validation (peak=%.2f GB). Falling back...", result["peak_vram_gb"])

    fallback_batch = max(1, rec.per_device_batch_size // 2)
    fallback_grad_accum = rec.gradient_accumulation_steps * 2

    # Ensure we don't exceed target effective batch size too much
    while fallback_batch > 1 and fallback_grad_accum > 128:
        fallback_batch = max(1, fallback_batch // 2)
        fallback_grad_accum *= 2

    result2 = validate_batch_size(model, tokenizer, fallback_batch, max_seq_length, device)

    if result2["fits"]:
        logger.info("Fallback validated: batch=%d, grad_accum=%d", fallback_batch, fallback_grad_accum)
        return BatchSizeRecommendation(
            per_device_batch_size=fallback_batch,
            gradient_accumulation_steps=fallback_grad_accum,
            effective_batch_size=fallback_batch * fallback_grad_accum,
            estimated_utilization=result2["utilization"],
            gpu_name=rec.gpu_name,
            total_vram_gb=rec.total_vram_gb,
            static_memory_gb=rec.static_memory_gb,
            activation_per_sample_gb=rec.activation_per_sample_gb,
        )

    # Last resort: batch_size=1, maximize gradient accumulation
    logger.error("Fallback also failed. Using batch_size=1 with gradient accumulation.")
    final_grad_accum = min(128, target_effective_batch_size)

    return BatchSizeRecommendation(
        per_device_batch_size=1,
        gradient_accumulation_steps=final_grad_accum,
        effective_batch_size=final_grad_accum,
        estimated_utilization=result2["utilization"],
        gpu_name=rec.gpu_name,
        total_vram_gb=rec.total_vram_gb,
        static_memory_gb=rec.static_memory_gb,
        activation_per_sample_gb=rec.activation_per_sample_gb,
    )
