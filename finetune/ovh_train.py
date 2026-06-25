"""
OVH Cloud AI Training job submission.
Builds Docker image, pushes to registry, and submits training job via ovhai CLI.
"""
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── OVH Configuration ─────────────────────────────────────────────────────────

OVH_REGION = os.environ.get("OVH_REGION", "GRA")
# H100 80GB at $3.50/hr — 100K examples × 10 epochs ≈ 8 hours ≈ $28
OVH_GPU_FLAVOR = os.environ.get("OVH_TRAINING_GPU", "h100-380")
OVH_TIMEOUT = os.environ.get("OVH_TRAINING_TIMEOUT", "12h")
DOCKER_IMAGE = os.environ.get("OVH_DOCKER_IMAGE", "rohan5commit/options-trainer:latest")
DATASET_CONTAINER = os.environ.get("OVH_DATASET_CONTAINER", "options-training-data")
OUTPUT_CONTAINER = os.environ.get("OVH_OUTPUT_CONTAINER", "options-model-output")


def _run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with logging."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600
    )
    if check and result.returncode != 0:
        logger.error("Command failed: %s", result.stderr)
        raise RuntimeError(f"Command failed: {result.stderr}")
    return result


def build_and_push_image() -> None:
    """Build the Docker image and push to Docker Hub."""
    logger.info("Building Docker image: %s", DOCKER_IMAGE)
    _run_command(["docker", "build", "-t", DOCKER_IMAGE, "-f", "finetune/Dockerfile", "."])
    logger.info("Pushing image to registry")
    _run_command(["docker", "push", DOCKER_IMAGE])
    logger.info("Image pushed successfully")


def submit_training_job(
    dataset_path: str = "./training_data",
    output_path: str = "./output",
    env_vars: dict[str, str] | None = None,
) -> str:
    """
    Submit a training job to OVH AI Training via ovhai CLI.
    Returns the job ID.
    """
    cmd = [
        "ovhai", "job", "run",
        "--name", f"options-llm-train-{int(time.time())}",
        "--gpu", "1",
        "--flavor", OVH_GPU_FLAVOR,
        "--timeout", OVH_TIMEOUT,
    ]

    # Mount dataset volume (read-only)
    cmd.extend([
        "--volume", f"{DATASET_CONTAINER}@{OVH_REGION}:/workspace/finetune/training_data:RO",
    ])

    # Mount output volume (read-write)
    cmd.extend([
        "--volume", f"{OUTPUT_CONTAINER}@{OVH_REGION}:/workspace/output:RW",
    ])

    # Environment variables
    if env_vars:
        for key, value in env_vars.items():
            cmd.extend(["--env", f"{key}={value}"])

    # Add HF token if available
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        cmd.extend(["--env", f"HF_TOKEN={hf_token}"])

    # Docker image and command
    cmd.append(DOCKER_IMAGE)

    logger.info("Submitting OVH training job")
    result = _run_command(cmd, check=False)

    if result.returncode != 0:
        logger.error("Job submission failed: %s", result.stderr)
        raise RuntimeError(f"Job submission failed: {result.stderr}")

    # Parse job ID from output
    try:
        job_info = json.loads(result.stdout)
        job_id = job_info.get("id", "unknown")
        logger.info("Training job submitted: %s", job_id)
        return job_id
    except json.JSONDecodeError:
        logger.info("Job submitted (output: %s)", result.stdout[:200])
        return result.stdout.strip()


def wait_for_job(job_id: str, poll_interval: int = 60) -> str:
    """Poll job status until completion."""
    logger.info("Waiting for job %s to complete", job_id)
    while True:
        result = _run_command(
            ["ovhai", "job", "get", job_id, "-o", "json"],
            check=False,
        )
        if result.returncode == 0:
            try:
                info = json.loads(result.stdout)
                state = info.get("status", {}).get("state", "unknown")
                logger.info("Job %s state: %s", job_id, state)
                if state in ("DONE", "FAILED", "ERROR"):
                    return state
            except json.JSONDecodeError:
                pass
        time.sleep(poll_interval)


def run_full_pipeline() -> None:
    """Execute the complete training pipeline on OVH Cloud."""
    logger.info("Starting full OVH training pipeline")

    # Step 1: Build and push Docker image
    logger.info("Step 1: Building Docker image")
    build_and_push_image()

    # Step 2: Submit training job
    logger.info("Step 2: Submitting training job")
    env_vars = {
        "BASE_MODEL_NAME": "meta-llama/Meta-Llama-3-8B-Instruct",
    }
    job_id = submit_training_job(env_vars=env_vars)

    # Step 3: Wait for completion
    logger.info("Step 3: Waiting for training to complete")
    final_state = wait_for_job(job_id)

    if final_state == "DONE":
        logger.info("Training completed successfully!")
        print(f"Training complete! Model saved to OVH output container: {OUTPUT_CONTAINER}")
    else:
        logger.error("Training failed with state: %s", final_state)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_full_pipeline()
