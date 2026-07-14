"""
Modal serverless GPU endpoint for LLM inference.
Loads Llama-3-8B-Instruct + LoRA adapter with 4-bit quantization.
Deploy once: modal deploy modal_inference.py
Call from llm_trader.py via modal.Inference.run().
"""
import logging
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

# ── Modal App ──────────────────────────────────────────────────────────────────

app = modal.App("options-llm-inference")

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.25.0",
        "sentencepiece",
        "protobuf",
    )
)


@app.cls(
    image=inference_image,
    gpu="L4",
    container_idle_timeout=300,
    timeout=180,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
class OptionsLLM:
    """
    Persistent container that loads the model once and serves multiple inference calls.
    Scales to zero when idle, scales up on demand.
    """

    @modal.enter()
    def load_model(self):
        """Load Llama-3-8B + LoRA adapter on container start."""
        import os

        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        hf_token = os.environ.get("HF_TOKEN", "")
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        adapter_repo = "Rohan556/options-llm-lora"

        logger.info("Loading tokenizer from %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=hf_token or None
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading base model in 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token or None,
        )

        logger.info("Loading LoRA adapter from %s", adapter_repo)
        try:
            self.model = PeftModel.from_pretrained(
                base_model, adapter_repo, token=hf_token or None
            )
        except Exception as exc:
            logger.warning(
                "Could not load LoRA adapter (%s). Using base model only.", exc
            )
            self.model = base_model

        self.model.eval()
        logger.info("Model loaded successfully")

    @modal.method()
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Run inference and return the LLM response text.
        Expects the prompt to contain the market context JSON.
        Returns raw text (should be valid JSON from the LLM).
        """
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (skip the input prompt)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()


# ── Local helper for calling the deployed endpoint ─────────────────────────────


def call_inference(context_json: str, system_prompt: str = "") -> str:
    """
    Call the deployed Modal endpoint from any environment.
    This function is imported by llm_trader.py.
    """
    import modal as _modal

    options_llm = _modal.Cls.from_name("options-llm-inference", "OptionsLLM")
    instance = options_llm()
    return instance.generate.remote(context_json, system_prompt=system_prompt)


# ── Standalone entry point for testing ─────────────────────────────────────────

if __name__ == "__main__":
    import json

    @app.local_entrypoint()
    def main():
        test_context = json.dumps({
            "symbol": "SPY",
            "underlying": {"price": 590.0, "rsi_14": 72.0},
            "iv_metrics": {"iv_rank": 0.8, "current_iv": 0.22},
            "options_chain": {"calls": [], "puts": []},
        })
        llm = OptionsLLM()
        result = llm.generate.remote(test_context)
        print(result)
