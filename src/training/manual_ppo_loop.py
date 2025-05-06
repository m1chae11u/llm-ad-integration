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
    df = df.iloc[:100] # Keep the 100 limit for testing
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1.4e-7) # Switched optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1.4e-7)

    log_path = "logs/ppo_manual_log.csv" # Main training log
    periodic_eval_log_path = "logs/periodic_eval_log.csv" # Log for periodic evals
    checkpoint_dir = "checkpoints/ppo_manual" # Checkpoint directory
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt") # Optimizer state path

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(periodic_eval_log_path), exist_ok=True)
    # Note: Checkpoint dir creation is handled later

    # --- Check for resuming --- 
    start_idx = 0
    if os.path.exists(log_path):
        try:
            log_df = pd.read_csv(log_path)
            if not log_df.empty:
                last_idx = log_df['idx'].iloc[-1]
                start_idx = int(last_idx) + 1 
                print(f"✅ Found previous log. Resuming training from index {start_idx}")
        except Exception as e:
            print(f"⚠️ Error reading log file {log_path}: {e}. Starting from index 0.")
            # Consider deleting or renaming corrupted log file
    else:
        print(f"ℹ️ No previous log file found. Starting from index 0.")
        # Create headers only if starting fresh
        # Header for main training log (including sub-scores)
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

    # Load optimizer state if resuming and state file exists
    if start_idx > 0 and os.path.exists(optimizer_path):
        try:
            optimizer.load_state_dict(torch.load(optimizer_path))
            print(f"✅ Loaded optimizer state from {optimizer_path}")
        except Exception as e:
             print(f"⚠️ Could not load optimizer state: {e}. Using fresh state.")
    elif start_idx > 0:
        print(f"⚠️ Resuming run, but optimizer state file not found at {optimizer_path}. Using fresh state.")

    # Prepare data slice and progress bar for current run
    total_rows = len(df)
    if start_idx >= total_rows:
         print("✅ Training already completed!")
         return # Exit if already done
         
    df_to_process = df.iloc[start_idx:]
    pbar = tqdm(enumerate(df_to_process.itertuples(index=False), start=start_idx), 
                total=total_rows, initial=start_idx, 
                desc="Manual PPO Training", dynamic_ncols=True)

    # Main loop starting from resume point
    for idx, row in pbar:
        # query = str(row.vague_query) # Adapt if itertuples(index=False) is used
        # Assuming standard itertuples which includes Index as first element
        # Correct access depends on how df.itertuples is called, adjust if needed
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

            # Periodic Evaluation & Checkpoint Saving (now every 25 steps)
            if idx > 0 and idx % 2 == 0: # Also ensure idx > 0 to avoid double save at start if resuming
                print(f"\n🔄 Running Periodic Evaluation at step {idx}...")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                torch.save(optimizer.state_dict(), optimizer_path) # <-- Save optimizer state
                print(f"💾 Saved checkpoint and optimizer state at step {idx}")
                
                # Store current eval results
                periodic_eval_results = []
                model.eval() # Ensure model is in eval mode
            
                eval_data_slice = df.iloc[:10] # Example: Use first 10 rows for eval
                for eval_idx, eval_row in enumerate(eval_data_slice.itertuples()): 
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
