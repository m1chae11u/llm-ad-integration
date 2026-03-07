# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Trains a language model (Llama 3.1-8B) to seamlessly insert advertisements into conversational responses using **d-RLAIF** (Reinforcement Learning from AI Feedback). An LLM judge (Google Gemini 1.5 Flash) scores ad-inserted responses, and those scores are used as rewards for PPO fine-tuning.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install wheel setuptools ninja
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
cp .env.example .env  # then fill in keys
```

Required `.env` keys:
- `GOOGLE_API_KEY` — required for Gemini judge and embeddings
- `HF_TOKEN` — required for gated Llama model download
- `OPENAI_API_KEY` — optional, used as fallback for embeddings

Optional `.env` tuning keys (OOM mitigations): `USE_LORA`, `USE_8BIT`, `PPO_BATCH_SIZE`, `MAX_NEW_TOKENS`, `LORA_R`, `LORA_ALPHA`, `LORA_TARGET_MODULES`, etc. See `.env.example` for full list.

## Running

```bash
# PPO training (main pipeline)
python src/main.py

# Baseline (no RL, just generate + judge)
python src/baseline/baseline_loop.py

# All tests
bash run_tests.sh

# Individual test file
python3 tests/test_components.py
python3 tests/test_judge_integration.py
python3 tests/test_training_integration.py
python3 tests/test_training_step.py
```

## Architecture

All source code is under `src/`. Scripts must be run from the repo root so relative paths (e.g., `data/merged_queries_ads.csv`) resolve correctly.

### Key modules

- **`src/config.py`** — Centralized config. Reads `.env` and defines `BASE_MODEL`, `CHECKPOINT_DIR`, `DATA_FILE`, and all API key constants. Import from here, not from `os.getenv` directly.

- **`src/main.py`** — Entry point. Loads ad data from CSV, builds the `MyPPOTrainer`, loads the latest checkpoint (if any), rebinds model/optimizer to the Accelerate trainer, then calls `trainer.ppo_train()`.

- **`src/training/ppo_training.py`** — Core PPO loop. Extends `CustomPPOTrainer` (from LlamaFactory). Key class is `MyPPOTrainer`. Uses LoRA via PEFT. Overrides `batched_forward_pass` to re-apply a forward-patch on whatever model instance is passed (needed because `accelerate.prepare()` creates new wrapper objects; see `EXPLANATION.md`). Async judge calls via `asyncio.gather`. Collects logs in `TRAINING_LOGS`.

- **`src/training/checkpoint_manager.py`** — Saves/loads `AutoModelForCausalLMWithValueHead` checkpoints (preserves `v_head`). Tracks `last_query_position.json` and `training_metrics.json`. Checkpoint directories are named `checkpoint-{step}`.

- **`src/training/prompts.py`** — System prompts for the ad-insertion model (`get_prompt_with_ad`, `get_prompt_without_ad`).

- **`src/judge/`** — Four async judge functions, each calling Gemini:
  - `coherence.py` — how well the ad fits the conversation
  - `helpfulness.py` — whether the overall response stays helpful
  - `salience.py` — clarity/noticeability of the ad
  - `detectability.py` — how obviously it reads as an ad (uses embedding similarity)
  - `utils.py` — shared Gemini/OpenAI client setup, async API wrappers, embedding cache

- **`src/generate/generator.py`** — Generates responses with and without ads from the base LLM.

- **`src/baseline/baseline_loop.py`** — Non-RL baseline: generates responses, judges them, logs scores. Uses `BaselineDataProcessor`.

- **`src/retriever/`** — RAG utilities (`EmbeddingManager`, `RAGManager`) for embedding-based ad retrieval (currently supplementary to main pipeline).

### Data

- `data/merged_queries_ads.csv` — Primary training data. Required columns: `ad_id`, `ad_product`, `brand`, `url`, `ad_description`.
- `data/user_queries.json` — User queries.
- `scripts/merged_queries_n_ads.py` — Script to generate `merged_queries_ads.csv` from raw data.

### Checkpoints

Saved under `checkpoints/ppo_llama/checkpoint-{step}/`. When resuming training after stopping, drop a saved checkpoint folder there and restart with `python src/main.py`. The trainer auto-detects the latest checkpoint.

### OOM Notes

The repo has iterative OOM fixes for Llama 3.1-8B on a single GPU:
- LoRA is enabled by default (`USE_LORA=true`) — disabling it requires ~64GB optimizer states
- 8-bit quantization available via `USE_8BIT=true`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256` is set at startup
- `_rebind_model_and_optimizer_for_accelerate` in `main.py` carefully avoids double `.to(cuda)` calls when loading checkpoints
