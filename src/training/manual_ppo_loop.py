import os
import torch
import pandas as pd
import gc
from tqdm import tqdm
from torch.nn import functional as F

from judge import (
    judge_coherence,
    judge_helpfulness,
    judge_ad_salience,
    judge_detectability
)
from generate.generator import generate_responses

# ✅ Prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def compute_ppo_loss(old_log_probs, new_log_probs, advantages, clip_range=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    return -torch.min(ratio * advantages, clipped * advantages).mean()

def compute_advantages(reward, value):
    return reward - value

def run_manual_ppo(model, tokenizer):
    device = model.device
    model.eval()

    df = pd.read_csv("data/merged_queries_ads.csv")
    df = df.iloc[:3]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.4e-6)

    log_path = "logs/ppo_manual_log.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        pd.DataFrame(columns=["idx", "query", "reward", "loss", "response", "C1", "C2", "C3", "C4",
                              "H1", "S1", "S2", "S3", "Detect_Cosine", "Detect_BERT"]).to_csv(log_path, index=False)

    pbar = tqdm(df.itertuples(), total=len(df), desc="Manual PPO Training", dynamic_ncols=True)
    for idx, row in enumerate(pbar):
        query = str(row.vague_query)
        ad_facts = {
            "ad_product": str(row.ad_product),
            "brand": str(row.brand),
            "url": str(row.url),
            "description": str(row.ad_description),
        }

        try:
            with torch.no_grad():
                response_without_ad, response_with_ad = generate_responses(query, ad_facts, model, tokenizer)
                print(f"\n🟦 [Row {idx}] Response WITHOUT ad:\n{response_without_ad}\n")
                print(f"🟨 [Row {idx}] Response WITH ad:\n{response_with_ad}\n")

            ad_text = f"""Product: {ad_facts['ad_product']}
                        Brand: {ad_facts['brand']}
                        URL: {ad_facts['url']}
                        Description: {ad_facts['description']}"""

            # Evaluate with judges
            with torch.no_grad():
                score_coh = judge_coherence(query, response_with_ad)
                score_help = judge_helpfulness(query, response_with_ad)
                score_sal = judge_ad_salience(query, response_with_ad, ad_text)
                score_det = judge_detectability(response_with_ad, response_without_ad)

            # Compute reward
            reward_values = [
                score_coh.get("Coherence Score", 0),
                score_help.get("Helpfulness Score", 0),
                score_sal.get("Ad Salience Score", 0),
                score_det.get("detectability_cosine", 0) or 0
            ]
            reward = torch.tensor(sum(reward_values) / len(reward_values), dtype=torch.float32).to(device)

            # Debug output
            debug_log = f"""
            ===============================
            📝 Query: {query}
            📦 Ad Facts: {ad_facts}
            📤 Response WITHOUT ad:
            {response_without_ad}

            📥 Response WITH ad:
            {response_with_ad}

            📊 Judge Scores:
            - Coherence Score: {score_coh}
            - Helpfulness Score: {score_help}
            - Ad Salience Score: {score_sal}
            - Detectability Score: {score_det}
            🎯 Final Reward: {reward.item():.4f}
            ===============================
            """

            print(debug_log)

            # 🔽 Optional: save debug output to file
            with open("logs/ppo_debug_log.txt", "a") as f:
                f.write(debug_log)
            pbar.set_postfix({
                "Reward": f"{reward.item():.3f}",
                "Loss": "---",
                "C": reward_values[0],
                "H": reward_values[1],
                "S": reward_values[2],
                "D": round(reward_values[3], 4)
            })

        except Exception as e:
            print(f"❌ Skipping row {idx} due to error: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        try:
            input_ids = tokenizer(query, return_tensors="pt", truncation=True, max_length=384).input_ids.to(device)
            response_ids = tokenizer(response_with_ad, return_tensors="pt", truncation=True, max_length=128).input_ids.to(device)[0]

            if input_ids.shape[1] + response_ids.shape[0] > 512:
                print(f"⚠️ Skipping row {idx}: combined input too long")
                continue

            input_plus_response = torch.cat([input_ids[0], response_ids])
            inputs = input_plus_response.unsqueeze(0)
            labels = input_plus_response[1:]

            logits = model(inputs).logits[0, :-1]
            logits = torch.clamp(logits, -50, 50)
            log_probs = F.log_softmax(logits, dim=-1)
            chosen_log_probs = log_probs[torch.arange(len(labels)), labels]
            old_logprob = chosen_log_probs.sum()
            value = reward.detach()
            advantage = compute_advantages(reward, value)

            model.train()
            logits = model(inputs).logits[0, :-1]
            new_log_probs = F.log_softmax(logits, dim=-1)[torch.arange(len(labels)), labels].sum()
            loss = compute_ppo_loss(old_logprob.detach(), new_log_probs, advantage)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.eval()

            del input_ids, response_ids, inputs, labels, logits, log_probs, new_log_probs, chosen_log_probs
            torch.cuda.empty_cache()
            gc.collect()

            pbar.set_postfix({
                "Reward": f"{reward.item():.3f}",
                "Loss": f"{loss.item():.4f}",
                "C": reward_values[0],
                "H": reward_values[1],
                "S": reward_values[2],
                "D": round(reward_values[3], 4)
            })

            pd.DataFrame([{
                "idx": idx,
                "query": query,
                "reward": reward.item(),
                "loss": loss.item(),
                "response": response_with_ad,
                "C1": score_coh.get("C1", 0),
                "C2": score_coh.get("C2", 0),
                "C3": score_coh.get("C3", 0),
                "C4": score_coh.get("C4", 0),
                "H1": score_help.get("H1", 0),
                "S1": score_sal.get("S1", 0),
                "S2": score_sal.get("S2", 0),
                "S3": score_sal.get("S3", 0),
                "Detect_Cosine": score_det.get("detectability_cosine"),
                "Detect_BERT": score_det.get("detectability_bert")
            }]).to_csv(log_path, mode="a", header=False, index=False)
            if idx % 50 == 0:
                os.makedirs("checkpoints/ppo_manual", exist_ok=True)
                model.save_pretrained("checkpoints/ppo_manual")
                tokenizer.save_pretrained("checkpoints/ppo_manual")
                print(f"💾 Saved checkpoint & logs at step {idx}")

        except torch.cuda.OutOfMemoryError as oom:
            print(f"🔥 CUDA OOM at row {idx}: {oom}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

    print("✅ PPO training complete. Log saved to logs/ppo_manual_log.csv")
