import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import json
import gc
import time
import signal
import logging
import asyncio
import copy

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler
from datasets import Dataset as HFDataset
import torch.nn as nn

from src.generate.generator import generate_responses, clear_response_cache
from src.judge.utils import clear_caches
from src.judge import (
    judge_coherence_async,
    judge_helpfulness_async,
    judge_ad_salience_async,
    judge_detectability_async,
)
gc.collect()
torch.cuda.empty_cache()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# CSV logging setup
# ──────────────────────────────────────────────────────────────────────────────
LOG_DIR   = Path("logs")
GEN_LOG   = LOG_DIR / "generations.csv"
JUDGE_LOG = LOG_DIR / "judgments.csv"
TRAIN_LOG = LOG_DIR / "training.csv"
STATS_LOG = LOG_DIR / "stats.csv"
LOG_DIR.mkdir(exist_ok=True, parents=True)

def _init_csv(path, columns):
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False)

_init_csv(GEN_LOG, [
    "batch_idx","query","ad_facts","response_without_ad","response_with_ad",
    "generation_time","token_count"
])
_init_csv(JUDGE_LOG, [
    "batch_idx","response_without_ad","response_with_ad",
    "C1","C2","C3","C4","Coherence Score","Coherence Explanation",
    "H1","Helpfulness Explanation",
    "S1","S2","S3","Ad Salience Score","Ad Salience Explanation",
    "detectability_cosine","similarity_cosine","total_score"
])
_init_csv(TRAIN_LOG, ["batch_idx","avg_reward","ppo_loss","kl_divergence"])
_init_csv(STATS_LOG, ["timestamp","total_queries","avg_reward_overall"])

gen_buffer   = []
judge_buffer = []
train_buffer = []
stats_buffer = []

def flush_logs():
    global gen_buffer, judge_buffer, train_buffer, stats_buffer
    if gen_buffer:
        pd.DataFrame(gen_buffer).to_csv(GEN_LOG, mode="a", header=False, index=False)
        gen_buffer.clear()
    if judge_buffer:
        pd.DataFrame(judge_buffer).to_csv(JUDGE_LOG, mode="a", header=False, index=False)
        judge_buffer.clear()
    if train_buffer:
        pd.DataFrame(train_buffer).to_csv(TRAIN_LOG, mode="a", header=False, index=False)
        train_buffer.clear()
    if stats_buffer:
        pd.DataFrame(stats_buffer).to_csv(STATS_LOG, mode="a", header=False, index=False)
        stats_buffer.clear()

def compute_rewards(records, responses, batch_idx):
    """
    Run the four judges asynchronously, log subscores + explanations,
    and return a tensor of total scores.
    """
    async def score_one(r, resp):
        ad_text = (
            f"Product: {r['ad_product']}\n"
            f"Brand: {r['brand']}\n"
            f"URL: {r['url']}\n"
            f"Description: {r['ad_description']}"
        )
        sc, sh, ss, sd = await asyncio.gather(
            judge_coherence_async(r["vague_query"], resp),
            judge_helpfulness_async(r["vague_query"], resp),
            judge_ad_salience_async(r["vague_query"], resp, ad_text),
            judge_detectability_async(resp, r["response_without_ad"]),
        )
        total = (
            sc.get("Coherence Score", 0)
            + sh.get("H1", 0)
            + ss.get("Ad Salience Score", 0)
            + (sd.get("detectability_cosine", 0) or 0)
        )
        judge_buffer.append({
            "batch_idx": batch_idx,
            "response_without_ad": r["response_without_ad"],
            "response_with_ad": resp,
            "C1": sc.get("C1", 0), "C2": sc.get("C2", 0),
            "C3": sc.get("C3", 0), "C4": sc.get("C4", 0),
            "Coherence Score": sc.get("Coherence Score", 0),
            "Coherence Explanation": sc.get("Coherence Explanation", ""),
            "H1": sh.get("H1", 0),
            "Helpfulness Explanation": sh.get("Helpfulness Explanation", ""),
            "S1": ss.get("S1", 0), "S2": ss.get("S2", 0), "S3": ss.get("S3", 0),
            "Ad Salience Score": ss.get("Ad Salience Score", 0),
            "Ad Salience Explanation": ss.get("Ad Salience Explanation", ""),
            "detectability_cosine": sd.get("detectability_cosine", 0) or 0,
            "similarity_cosine": sd.get("similarity_cosine", 0) or 0,
            "total_score": total,
        })
        return total

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    totals = loop.run_until_complete(
        asyncio.gather(*(score_one(r, resp) for r, resp in zip(records, responses)))
    )
    loop.close()
    return torch.tensor(totals, dtype=torch.float32, device=DEVICE)

# ──────────────────────────────────────────────────────────────────────────────
# Ctrl-C checkpointing
# ──────────────────────────────────────────────────────────────────────────────
_interrupted = False
def _handle_sigint(signum, frame):
    global _interrupted
    _interrupted = True
signal.signal(signal.SIGINT, _handle_sigint)

def run_manual_ppo(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    collect_batch_size: int = 4,      # smaller for debugging
    ppo_batch_size:    int = 2,      # smaller for debugging
    ppo_epochs:        int = 4,
    lr:                float = 1e-5,
    kl_target:         float = 0.1,
    hf_token:         str  = None,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_path = out_dir / "last_batch_idx.json"
    if pos_path.exists():
        start_batch = json.loads(pos_path.read_text()).get("last_batch_idx", 0)
        logger.info(f"Resuming from batch {start_batch}")
    else:
        start_batch = 0

    df_full = pd.read_csv(dataset_path)
    hf_ds    = HFDataset.from_pandas(df_full)
    df       = df_full.iloc[start_batch * collect_batch_size : ].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token, use_fast=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, token=hf_token
    ).to(DEVICE)
    # Create a separate value model instance for PPOTrainer
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, token=hf_token
    ).to(DEVICE)
    # Expose backbone prefix, backbone module, and score method for the PPO trainer
    value_model.base_model_prefix = value_model.pretrained_model.base_model_prefix
    # attach the actual backbone module so getattr(wrapper, base_model_prefix) finds it
    setattr(value_model, value_model.base_model_prefix, value_model.pretrained_model.model)
    # attach a score method that uses the v_head
    value_model.score = lambda hidden_states: value_model.v_head(hidden_states).squeeze(-1)

    # Dummy reward model to satisfy PPOTrainer (we feed actual rewards manually)
    dummy_reward = nn.Identity().to(DEVICE)

    # ensure generation_config exists
    if not hasattr(policy, "generation_config"):
        policy.generation_config = GenerationConfig.from_pretrained(model_name, token=hf_token)
    policy.generation_config.eos_token_id = tokenizer.eos_token_id

    ppo_cfg = PPOConfig(
        output_dir=str(out_dir),
        exp_name="ppo_run",
        reward_model_path=model_name,
        per_device_train_batch_size=ppo_batch_size,
        gradient_accumulation_steps=1,
        num_ppo_epochs=ppo_epochs,
        cliprange=0.2,
        vf_coef=0.5,
        kl_coef=kl_target,
        report_to=None,
    )
    ppo = PPOTrainer(
        args=ppo_cfg,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=dummy_reward,
        train_dataset=hf_ds,
        value_model=value_model,
    )

    length_sampler = LengthSampler(32, 128)
    total_queries = start_batch * collect_batch_size

    try:
        for batch_idx, i in enumerate(
            tqdm(range(0, len(df), collect_batch_size), desc="PPO"),
            start=start_batch
        ):
            if _interrupted:
                break

            slice_df = df.iloc[i : i + collect_batch_size]
            records  = slice_df.to_dict("records")
            queries  = [r["vague_query"] for r in records]

            # 1) generate
            t0 = time.time()
            no_ads, with_ads = [], []
            for r in records:
                no_ad, with_ad = generate_responses(
                    r["vague_query"],
                    {
                        "ad_product":   r["ad_product"],
                        "brand":        r["brand"],
                        "url":          r["url"],
                        "description":  r["ad_description"],
                    },
                    policy, tokenizer,
                )
                no_ads.append(no_ad)
                with_ads.append(with_ad)
            gen_time_each = (time.time() - t0) / len(records)

            # inject the no-ad baseline into each record so compute_rewards can see it
            for idx, r in enumerate(records):
                r["response_without_ad"] = no_ads[idx]

            # 2) log generation
            for idx, r in enumerate(records):
                gen_buffer.append({
                    "batch_idx":           batch_idx,
                    "query":               r["vague_query"],
                    "ad_facts":            json.dumps({
                        "ad_product":    r["ad_product"],
                        "brand":         r["brand"],
                        "url":           r["url"],
                        "ad_description":r["ad_description"],
                    }),
                    "response_without_ad": no_ads[idx],
                    "response_with_ad":    with_ads[idx],
                    "generation_time":     gen_time_each,
                    "token_count":         len(tokenizer.encode(with_ads[idx])),
                })

            # 3) judge & reward
            rewards = compute_rewards(records, with_ads, batch_idx)

            # 4) PPO step
            stats = ppo.step(queries, with_ads, rewards)
            total_queries += len(queries)

            # 5) training log
            train_buffer.append({
                "batch_idx":     batch_idx,
                "avg_reward":    rewards.mean().item(),
                "ppo_loss":      stats.get("ppo/loss/total"),
                "kl_divergence": stats.get("ppo/kl_divergence"),
            })

            # 6) periodic flush
            if batch_idx % 5 == 0:
                stats_buffer.append({
                    "timestamp":          time.time(),
                    "total_queries":      total_queries,
                    "avg_reward_overall": sum(tb["avg_reward"] for tb in train_buffer) / len(train_buffer),
                })
                flush_logs()

            # 7) checkpoint every 20 batches
            if batch_idx % 20 == 0:
                ckpt_dir = out_dir / f"ckpt_{batch_idx}"
                ppo.save_pretrained(str(ckpt_dir))
                pos_path.write_text(json.dumps({"last_batch_idx": batch_idx}))
                logger.info(f"🔖 Saved checkpoint @ batch {batch_idx}")

            clear_caches()
            clear_response_cache()
            gc.collect()
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.info("🛑 KeyboardInterrupt received — saving final checkpoint…")
        ckpt_dir = out_dir / f"ckpt_{batch_idx}_interrupt"
        ppo.save_pretrained(str(ckpt_dir))
        pos_path.write_text(json.dumps({"last_batch_idx": batch_idx}))
        logger.info(f"🔖 Saved interrupt-checkpoint @ batch {batch_idx}")
        raise

    # final save & flush
    final_dir = out_dir / "final_model"
    ppo.save_pretrained(str(final_dir))
    pos_path.write_text(json.dumps({"last_batch_idx": batch_idx}))
    stats_buffer.append({
        "timestamp":          time.time(),
        "total_queries":      total_queries,
        "avg_reward_overall": sum(tb["avg_reward"] for tb in train_buffer) / len(train_buffer),
    })
    flush_logs()
    logger.info("✅ PPO training complete.")


    ## note: the embedding API is rate-limited, so we need to throttle requests
    ## the current version of ppo doesnt support step() with a reward model. willl try to figure out a way to do this. 