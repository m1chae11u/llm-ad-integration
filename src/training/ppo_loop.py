import os
import gc
import torch
import csv
import pandas as pd
from tqdm import tqdm
from trl import PPOTrainer, PPOConfig

from generate.generator import generate_responses
from training.reward import compute_reward

def run_ppo_finetuning(model, tokenizer, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
    # --- PPO config & trainer ---
    ppo_config = PPOConfig(
        batch_size=1,
        mini_batch_size=1,
        learning_rate=1.4e-5,
        optimize_cuda_cache=True
    )
    ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

    # --- Load data ---
    print("Loading data...")
    df = pd.read_csv("data/merged_queries_ads.csv")
    print(f"Loaded {len(df)} rows\n")

    # --- Set up logging ---
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/ppo_training_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Step", "Query", "Response Without Ad", "Response With Ad",
                "C1", "C2", "C3", "C4", "Coherence Explanation", "Coherence Score",
                "H1", "Helpfulness Explanation", "Helpfulness Score",
                "S1", "S2", "S3", "Ad Salience Explanation", "Ad Salience Score",
                "Detectability Cosine", "Similarity Cosine", "Detectability BERT", "BERT F1",
                "Total Reward"
            ])

    resample_log_file = "logs/resampled_rollouts.csv"
    if not os.path.exists(resample_log_file):
        with open(resample_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Step When Resampled", "Original Step", "Query", "New Response With Ad", "New Reward"
            ])

    # --- PPO loop ---
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="PPO Fine-tuning"):
        try:
            user_query = row["vague_query"]
            ad_facts = {
                "ad_product": row["ad_product"],
                "brand": row["brand"],
                "url": row["url"],
                "description": row["ad_description"]
            }

            query_tensor = tokenizer(user_query, return_tensors="pt").input_ids.to(model.device)

            # Generate responses
            response_without_ad, response_with_ad = generate_responses(user_query, ad_facts, model, tokenizer)

            # Compute reward
            reward_details = compute_reward(user_query, response_with_ad, response_without_ad, ad_facts)
            reward_score = reward_details.get("Total Score", 0)
            reward_tensor = torch.tensor([reward_score]).to(model.device)

            # PPO step
            ppo_trainer.step(
                [query_tensor.squeeze(0)],
                [tokenizer.encode(response_with_ad, return_tensors="pt").squeeze(0).to(model.device)],
                [reward_tensor]
            )

            # Log full reward details
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx, user_query, response_without_ad, response_with_ad,
                    reward_details.get("C1", ""), reward_details.get("C2", ""), reward_details.get("C3", ""), reward_details.get("C4", ""),
                    reward_details.get("Coherence Explanation", ""), reward_details.get("Coherence Score", 0),
                    reward_details.get("H1", ""), reward_details.get("Helpfulness Explanation", ""), reward_details.get("Helpfulness Score", 0),
                    reward_details.get("S1", ""), reward_details.get("S2", ""), reward_details.get("S3", ""), reward_details.get("Ad Salience Explanation", ""), reward_details.get("Ad Salience Score", 0),
                    reward_details.get("detectability_cosine", ""), reward_details.get("similarity_cosine", ""),
                    reward_details.get("detectability_bert", ""), reward_details.get("bert_f1", ""),
                    reward_score
                ])

            # Periodically resample and log updated rollout
            if idx % 20 == 0:  # Every 20 steps, resample a past query
                sample_idx = max(0, idx - 10)
                past_query = df.iloc[sample_idx]["vague_query"]
                past_ad_facts = {
                    "ad_product": df.iloc[sample_idx]["ad_product"],
                    "brand": df.iloc[sample_idx]["brand"],
                    "url": df.iloc[sample_idx]["url"],
                    "description": df.iloc[sample_idx]["ad_description"]
                }
                _, resampled_response = generate_responses(past_query, past_ad_facts, model, tokenizer)
                resampled_reward = compute_reward(past_query, resampled_response, "", past_ad_facts).get("Total Score", 0)

                with open(resample_log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([idx, sample_idx, past_query, resampled_response, resampled_reward])

            print(f"[{idx}] Reward: {reward_score:.2f} | Query: {user_query[:40]}")

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            # Periodic model saving
            if idx % 50 == 0 or idx == len(df) - 1:
                output_dir = "checkpoints/ppo_finetuned_deepseek"
                os.makedirs(output_dir, exist_ok=True)
                print(f"\nSaving PPO fine-tuned model to {output_dir}...")
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print("PPO model and tokenizer saved.")

        except Exception as e:
            print(f"Error on row {idx}: {e}")
            continue

    print("\nPPO fine-tuning complete.")
