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
from concurrent.futures import ThreadPoolExecutor


# Prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def compute_ppo_loss(old_log_probs, new_log_probs, advantages, clip_range=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    return -torch.min(ratio * advantages, clipped * advantages).mean()

def compute_advantages(reward, value):
    return reward - value

def run_manual_ppo(model, tokenizer):
    device = model.device
    # model.gradient_checkpointing_enable() # Disabled for debugging stability
    model.eval()

    df = pd.read_csv("data/merged_queries_ads.csv")
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1.4e-7) # Switched optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1.4e-7)

    log_path = "logs/ppo_manual_log.csv" # Main training log
    periodic_eval_log_path = "logs/periodic_eval_log.csv" # Log for periodic evals
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(periodic_eval_log_path), exist_ok=True)

    # Header for main training log (including sub-scores)
    if not os.path.exists(log_path):
        pd.DataFrame(columns=["idx", "query", "response_trained", "reward", "loss",
                              "C1", "C2", "C3", "C4", "Coherence Score",
                              "H1", "Helpfulness Score",
                              "S1", "S2", "S3", "Ad Salience Score",
                              "Detect_Cosine"])\
            .to_csv(log_path, index=False)
            
    # Header for periodic evaluation log
    if not os.path.exists(periodic_eval_log_path):
        pd.DataFrame(columns=["step_idx", "query", "eval_base_response", "eval_ppo_response",
                              "ad_product", "brand", "url", "ad_description",
                              "C1", "C2", "C3", "C4", "Coherence Score",
                              "H1", "Helpfulness Score",
                              "S1", "S2", "S3", "Ad Salience Score",
                              "Detect_Cosine"])\
            .to_csv(periodic_eval_log_path, index=False)

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
            with torch.no_grad(), ThreadPoolExecutor(max_workers=4) as executor:
                future_coh = executor.submit(judge_coherence, query, response_with_ad)
                future_help = executor.submit(judge_helpfulness, query, response_with_ad)
                future_sal = executor.submit(judge_ad_salience, query, response_with_ad, ad_text)
                future_det = executor.submit(judge_detectability, response_with_ad, response_without_ad)

                score_coh = future_coh.result()
                score_help = future_help.result()
                score_sal = future_sal.result()
                score_det = future_det.result()

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
            📊 Judge Scores:
            - Coherence Score: {score_coh}
            - Helpfulness Score: {score_help}
            - Ad Salience Score: {score_sal}
            - Detectability Score: {score_det}
            🎯 Final Reward: {reward.item():.4f}
            ===============================
            """

            print(debug_log)

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
            log_probs = F.log_softmax(logits, dim=-1)
            chosen_log_probs = log_probs[torch.arange(len(labels)), labels]
            old_logprob = chosen_log_probs.sum()

            model.train()
            logits = model(inputs).logits[0, :-1]
            new_log_probs = F.log_softmax(logits, dim=-1)[torch.arange(len(labels)), labels].sum()
            
            # Simple REINFORCE-like loss
            loss = -new_log_probs * reward.detach()

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

            # Log training step data including sub-scores
            pd.DataFrame([{
                "idx": idx,
                "query": query,
                "response_trained": response_with_ad, # The response used for this step's update
                "reward": reward.item(),
                "loss": loss.item(),
                "C1": score_coh.get("C1", 0),
                "C2": score_coh.get("C2", 0),
                "C3": score_coh.get("C3", 0),
                "C4": score_coh.get("C4", 0),
                "Coherence Score": score_coh.get("Coherence Score", 0),
                "H1": score_help.get("H1", 0),
                "Helpfulness Score": score_help.get("Helpfulness Score", 0),
                "S1": score_sal.get("S1", 0),
                "S2": score_sal.get("S2", 0),
                "S3": score_sal.get("S3", 0),
                "Ad Salience Score": score_sal.get("Ad Salience Score", 0),
                "Detect_Cosine": score_det.get("detectability_cosine"),
            }]).to_csv(log_path, mode="a", header=False, index=False)

            # Periodic Evaluation & Checkpoint Saving (every 25 steps)
            if idx % 25 == 0:
                print(f"\n🔄 Running Periodic Evaluation at step {idx}...")
                os.makedirs("checkpoints/ppo_manual", exist_ok=True)
                model.save_pretrained("checkpoints/ppo_manual")
                tokenizer.save_pretrained("checkpoints/ppo_manual")
                print(f"💾 Saved checkpoint at step {idx}")
                
                # Store current eval results
                periodic_eval_results = []
                model.eval() # Ensure model is in eval mode
            
                for eval_idx, eval_row in enumerate(df.itertuples()): 
                    eval_query = str(eval_row.vague_query)
                    eval_ad_facts = {
                        "ad_product": str(eval_row.ad_product),
                        "brand": str(eval_row.brand),
                        "url": str(eval_row.url),
                        "description": str(eval_row.ad_description),
                    }
                    eval_ad_text = f"""Product: {eval_ad_facts['ad_product']}
                                    Brand: {eval_ad_facts['brand']}
                                    URL: {eval_ad_facts['url']}
                                    Description: {eval_ad_facts['description']}"""

                    try:
                        with torch.no_grad():
                            # Generate NEW responses with the current model state
                            eval_response_without_ad, eval_response_with_ad = generate_responses(eval_query, eval_ad_facts, model, tokenizer)
                            
                            # Run all judges
                            eval_score_coh = judge_coherence(eval_query, eval_response_with_ad)
                            eval_score_help = judge_helpfulness(eval_query, eval_response_with_ad)
                            eval_score_sal = judge_ad_salience(eval_query, eval_response_with_ad, eval_ad_text)
                            eval_score_det = judge_detectability(eval_response_with_ad, eval_response_without_ad)
                            
                        periodic_eval_results.append({
                            "step_idx": idx,
                            "query": eval_query,
                            "eval_base_response": eval_response_without_ad,
                            "eval_ppo_response": eval_response_with_ad,
                            "ad_product": eval_ad_facts.get("ad_product", ""),
                            "brand": eval_ad_facts.get("brand", ""),
                            "url": eval_ad_facts.get("url", ""),
                            "ad_description": eval_ad_facts.get("description", ""),
                            "C1": eval_score_coh.get("C1", 0), "C2": eval_score_coh.get("C2", 0), "C3": eval_score_coh.get("C3", 0), "C4": eval_score_coh.get("C4", 0),
                            "Coherence Score": eval_score_coh.get("Coherence Score", 0),
                            "H1": eval_score_help.get("H1", 0),
                            "Helpfulness Score": eval_score_help.get("Helpfulness Score", 0),
                            "S1": eval_score_sal.get("S1", 0), "S2": eval_score_sal.get("S2", 0), "S3": eval_score_sal.get("S3", 0),
                            "Ad Salience Score": eval_score_sal.get("Ad Salience Score", 0),
                            "Detect_Cosine": eval_score_det.get("detectability_cosine"),
                        
                        })
                    except Exception as eval_e:
                        print(f"❌ Error during periodic evaluation for query {eval_idx}: {eval_e}")
                        continue # Skip this row on error
                
                # Append evaluation results to the specific log file
                if periodic_eval_results:
                    pd.DataFrame(periodic_eval_results).to_csv(periodic_eval_log_path, mode="a", header=False, index=False)
                    print(f"💾 Periodic evaluation results appended to {periodic_eval_log_path}")
                else:
                    print("⚠️ No results to save from periodic evaluation.")

        except torch.cuda.OutOfMemoryError as oom:
            print(f"🔥 CUDA OOM at row {idx}: {oom}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

    print("✅ PPO training complete. Log saved to logs/ppo_manual_log.csv")
