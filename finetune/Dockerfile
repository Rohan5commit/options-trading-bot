FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    peft>=0.7.0 \
    bitsandbytes>=0.41.0 \
    accelerate>=0.25.0 \
    datasets>=2.16.0 \
    huggingface-hub>=0.20.0 \
    trl>=0.7.0 \
    scipy \
    sentencepiece \
    protobuf

# Copy training code
COPY finetune/ /workspace/finetune/
COPY requirements.txt /workspace/requirements.txt

# Set environment variables
ENV BASE_MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
ENV OUTPUT_DIR=/workspace/output/options_llm
ENV TRAINING_DATA=/workspace/finetune/training_data/train.jsonl
ENV EVAL_DATA=/workspace/finetune/training_data/test.jsonl
ENV GPU_PRICE_HR=0.60
ENV MAX_BUDGET=45.0

# Default command
CMD ["bash", "/workspace/finetune/run_lightning.sh"]
