import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.nn import functional as F

from judge import (
    judge_coherence,
    judge_helpfulness,
    judge_ad_salience,
    judge_detectability
)
from generate.generator import clean_response, generate_responses

# PPO Helper (simplified)
def compute_ppo_loss(old_log_probs, new_log_probs, advantages, clip_range=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    return -torch.min(ratio * advantages, clipped * advantages).mean()

def compute_advantages(reward, value):
    return reward - value  # Simple baseline

def run_manual_ppo(model, tokenizer):
    device = model.device
    model.eval()

    df = pd.read_csv("data/merged_queries_ads.csv")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.4e-5)

    logs = []

    for idx, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="Manual PPO Training", dynamic_ncols=True)):
        query = row.vague_query
        ad_facts = {
            "ad_product": str(row.ad_product),
            "brand": str(row.brand),
            "url": str(row.url),
            "description": str(row.ad_description),
        }

        try:
            print(f"\n[{idx}] Generating responses for query: {query[:60]}...")
            response_without_ad, response_with_ad = generate_responses(query, ad_facts, model, tokenizer)
            cleaned = clean_response(response_with_ad)

            ad_text = f"""Product: {ad_facts['ad_product']}
Brand: {ad_facts['brand']}
URL: {ad_facts['url']}
Description: {ad_facts['description']}"""

            print("Judging response quality...")
            score_coh = judge_coherence(query, cleaned)
            score_help = judge_helpfulness(query, cleaned)
            score_sal = judge_ad_salience(query, cleaned, ad_text)
            score_det = judge_detectability(response_with_ad, response_without_ad)

            reward_values = [
                score_coh.get("Coherence Score", 0),
                score_help.get("Helpfulness Score", 0),
                score_sal.get("Ad Salience Score", 0),
                score_det.get("detectability_bert", 0) or 0
            ]
            reward = torch.tensor(sum(reward_values) / len(reward_values), dtype=torch.float32).to(device)

            print(f"Subscores → C: {reward_values[0]} | H: {reward_values[1]} | S: {reward_values[2]} | D: {reward_values[3]}")
        except Exception as e:
            print(f"Skipping row {idx} due to judge error: {e}")
            continue

        # Tokenization and log probs
        input_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
        response_ids = tokenizer(cleaned, return_tensors="pt").input_ids.to(device)[0]
        input_plus_response = torch.cat([input_ids[0], response_ids])
        inputs = input_plus_response.unsqueeze(0)
        labels = input_plus_response[1:]

        logits = model(inputs).logits[0, :-1]
        log_probs = F.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs[torch.arange(len(labels)), labels]
        old_logprob = chosen_log_probs.sum()

        # PPO optimization
        value = reward.detach()
        advantage = compute_advantages(reward, value)

        model.train()
        logits = model(inputs).logits[0, :-1]
        new_log_probs = F.log_softmax(logits, dim=-1)[torch.arange(len(labels)), labels].sum()
        loss = compute_ppo_loss(old_logprob.detach(), new_log_probs, advantage)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()

        print(f"🔧 PPO Update → Reward: {reward.item():.3f} | Loss: {loss.item():.4f} | Adv: {advantage.item():.4f}")

        # Store training log
        logs.append({
            "idx": idx,
            "reward": reward.item(),
            "loss": loss.item(),
            "response": cleaned,
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
        })

        if idx % 50 == 0:
            os.makedirs("checkpoints/ppo_manual", exist_ok=True)
            model.save_pretrained("checkpoints/ppo_manual")
            tokenizer.save_pretrained("checkpoints/ppo_manual")
            pd.DataFrame(logs).to_csv("logs/ppo_manual_log.csv", index=False)
            print(f"Saved model & log checkpoint at step {idx}")

    # Final save
    pd.DataFrame(logs).to_csv("logs/ppo_manual_log.csv", index=False)
    print("PPO training complete. Final log saved to logs/ppo_manual_log.csv")