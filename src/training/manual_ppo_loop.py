import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import json
import gc
import time
import signal
import logging
import asyncio

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModelForCausalLM,
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler
from datasets import Dataset as HFDataset
import torch.nn as nn
from types import SimpleNamespace
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig

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

# Dummy backbone for v_head compliance
class DummyBackbone(nn.Module):
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        hidden = torch.zeros(batch_size, seq_len, 1, device=input_ids.device)
        return SimpleNamespace(hidden_states=[hidden])

class JudgeRewardModel(nn.Module):
    def __init__(self, tokenizer, records, pad_token_id, context_length):
        super().__init__()
        self.tokenizer = tokenizer
        self.records = records
        self.base_model_prefix = "judge_backbone"
        self.judge_backbone = DummyBackbone()
    async def _score_one(self, rec):
        sal = await judge_ad_salience_async(
            rec["vague_query"], rec["response_with_ad"], {
              "ad_product": rec["ad_product"],
              "brand":      rec["brand"],
              "url":        rec["url"],
              "description":rec["ad_description"],
            }
        )
        det = await judge_detectability_async(
            rec["response_with_ad"], rec["response_without_ad"]
        )
        return sal.get("salience_score", 0.0) - det.get("detectability_score", 0.0)
    def score(self, hidden_states):
        rewards = asyncio.run(
            asyncio.gather(*(self._score_one(r) for r in self.records))
        )
        r = torch.tensor(rewards, device=hidden_states.device)
        return r.unsqueeze(1).expand(-1, hidden_states.shape[1])

def run_manual_ppo(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    collect_batch_size: int = 4,
    ppo_batch_size:    int = 2,
    ppo_epochs:        int = 4,
    lr:                float = 1e-5,
    kl_target:         float = 0.1,
    hf_token:         str  = None,
):
    out_dir = Path(output_dir); out_dir.mkdir(exist_ok=True, parents=True)
    pos_path = out_dir / "last_batch_idx.json"
    start_batch = json.loads(pos_path.read_text()).get("last_batch_idx",0) if pos_path.exists() else 0

    # ──────────────────────────────────────────────
    # Load & patch CSV
    # ──────────────────────────────────────────────
    df_full = pd.read_csv(dataset_path)
    # ensure we have both fields so compute_rewards & logging won't blow up
    df_full["response_without_ad"] = ""
    df_full["response_with_ad"]    = ""
    hf_ds = HFDataset.from_pandas(df_full)

    # ──────────────────────────────────────────────
    # Tokenizer & Models
    # ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token, use_fast=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=hf_token,
    )

    # ensure generation_config exists so PPOTrainer.__init__ won't error
    policy.generation_config = GenerationConfig.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token
    )
    policy.generation_config.eos_token_id = tokenizer.eos_token_id

    value_model = policy

    ref_model = create_reference_model(policy)
    ref_model.to("cpu")
    ref_model.config.return_dict = True
    ref_model.eval()
    for p in ref_model.parameters(): p.requires_grad=False

    ppo_cfg = PPOConfig(
        exp_name="ppo_run",
        seed=42,
        batch_size=collect_batch_size,
        mini_batch_size=collect_batch_size,
        gradient_accumulation_steps=1,
        ppo_epochs=ppo_epochs,
        cliprange=0.2,
        vf_coef=0.5,
        kl_penalty="full",
        optimize_device_cache=True,
        gradient_checkpointing=True,
    )

    ds = hf_ds.map(
        lambda e: tokenizer(e["vague_query"], truncation=True, padding="longest"),
        batched=True,
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    judge_records = df_full.to_dict("records")
    judge_rm = JudgeRewardModel(tokenizer, judge_records, tokenizer.pad_token_id, None)

    ppo = PPOTrainer(
        model     = policy,     # your AutoModelForCausalLMWithValueHead
        config    = ppo_cfg,    # the config you just built
        dataset   = ds,         # your tokenized HF dataset
        tokenizer = tokenizer,  # HF tokenizer
        ref_model = ref_model,  # optional: your frozen LM for KL
    )

    # ──────────────────────────────────────────────
    # Manual on-policy loop
    # ──────────────────────────────────────────────
    loader = DataLoader(ds, batch_size=collect_batch_size, shuffle=False)
    for batch_idx, batch in enumerate(loader, start=start_batch):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        # ---------------------------------------------------------------------
        # 1) Generate both no-ad & with-ad via your helper, per record
        # ---------------------------------------------------------------------
        # Switch to eval mode for generation to avoid checkpointing conflicts
        policy.eval()
        clear_response_cache()
        t0 = time.time()
        records_batch = df_full.iloc[
            batch_idx*collect_batch_size : (batch_idx+1)*collect_batch_size
        ].to_dict("records")

        no_ad_list, with_ad_list = [], []
        for rec in records_batch:
            ad_facts = {
                "ad_product":   rec["ad_product"],
                "brand":        rec["brand"],
                "url":          rec["url"],
                "description":  rec["ad_description"],
            }
            try:
                with torch.no_grad():
                    try:
                        no_ad, with_ad = generate_responses(
                            rec["vague_query"],
                            ad_facts,
                            policy,
                            tokenizer
                        )
                    except RuntimeError as gen_err:
                        msg = str(gen_err)
                        if "probability tensor contains" in msg or "device-side assert triggered" in msg:
                            print("🔥 Skipping bad generation and clearing CUDA cache.")
                            torch.cuda.empty_cache()
                            no_ad = ""
                            with_ad = ""
                        else:
                            raise
            except RuntimeError as e:
                print(f"🔥 Generation RuntimeError: {e}")
                torch.cuda.empty_cache()
                no_ad = ""
                with_ad = ""
            rec["response_without_ad"] = no_ad
            rec["response_with_ad"]    = with_ad
            no_ad_list.append(no_ad)
            with_ad_list.append(with_ad)
        gen_time = time.time() - t0
        # Return to train mode for PPO updates
        policy.train()

        # ---------------------------------------------------------------------
        # 2) Log generations
        # ---------------------------------------------------------------------
        for i, rec in enumerate(records_batch):
            gen_buffer.append({
                "batch_idx":           batch_idx,
                "query":               rec["vague_query"],
                "ad_facts":            rec["ad_description"],
                "response_without_ad": rec["response_without_ad"],
                "response_with_ad":    rec["response_with_ad"],
                "generation_time":     gen_time,
                "token_count":         len(tokenizer(rec["response_with_ad"]).input_ids),
            })

        # ---------------------------------------------------------------------
        # 3) Write back into judge_records & compute rewards
        # ---------------------------------------------------------------------
        start = batch_idx*collect_batch_size
        end   = start + len(records_batch)
        for i, rec in enumerate(records_batch):
            judge_records[start + i]["response_without_ad"] = rec["response_without_ad"]
            judge_records[start + i]["response_with_ad"]    = rec["response_with_ad"]

        rewards = compute_rewards(
            judge_records[start:end],
            with_ad_list,
            batch_idx
        )
        
        # Print out reward scores for this batch
        print(f"Batch {batch_idx} reward scores: {rewards.detach().cpu().tolist()}")

        # ---------------------------------------------------------------------
        # 4) PPO update step (batch-encode and pad all responses)
        # ---------------------------------------------------------------------
        # Prepare prompts as token tensors for PPO step
        queries_list = [rec["vague_query"] for rec in records_batch]
        tokenized_queries = tokenizer(
            queries_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        query_tensors = tokenized_queries.input_ids.to(DEVICE)
        # Tokenize and pad responses
        tokenized_resps = tokenizer(
            with_ad_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        response_tensors = tokenized_resps.input_ids.to(DEVICE)
        # Split rewards into list of scalar tensors for PPOTrainer 0.11 step()
        score_list = list(rewards.unbind())
        stats = ppo.step(
            list(query_tensors),     # List[torch.Tensor]
            list(response_tensors),  # List[torch.Tensor]
            score_list               # List[torch.Tensor]
        )

        # ---------------------------------------------------------------------
        # 5) Log training stats & flush
        # ---------------------------------------------------------------------
        train_buffer.append({
            "batch_idx":     batch_idx,
            "avg_reward":    rewards.mean().item(),
            "ppo_loss":      float(stats.get("ppo/loss/total", 0.0)),
            "kl_divergence": float(stats.get("objective/kl", 0.0)),
        })
        flush_logs()

        # Skip batch if any generation failed to avoid GPU-side asserts
        if any(resp == "" for resp in no_ad_list) or any(resp == "" for resp in with_ad_list):
            print(f"⚠️ Skipping batch {batch_idx} due to generation error.")
            continue

        if _interrupted:
            break

    logger.info("✅ PPO training complete.")
    ppo.save_pretrained(str(out_dir / "final_model"))
    pos_path.write_text(json.dumps({"last_batch_idx": batch_idx}))
    stats_buffer.append({
        "timestamp":          time.time(),
        "total_queries":      (batch_idx+1)*collect_batch_size,
        "avg_reward_overall": sum(tb["avg_reward"] for tb in train_buffer) / len(train_buffer),
    })
    flush_logs()