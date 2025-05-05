import os
import gc
import torch
import csv
import pandas as pd
from tqdm import tqdm
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from copy import deepcopy
from generate.generator import generate_responses
from training.reward import compute_reward
import torch.nn as nn
import json

class DummyRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, queries, responses):
        # Return dummy rewards, actual rewards computed in training loop
        return torch.ones(len(queries))


class DummyProcessing:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, queries, responses):
        # Minimal processing, actual tokenization happens in loop
        return queries, responses

    def to(self, device):
        return self

    def eval(self):
        return self

 # dummy reward, will be ignored
def run_ppo_finetuning(model, tokenizer, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
    # model is AutoModelForCausalLMWithValueHead

    # --- PPO config & trainer ---
    reward_model = DummyRewardModel()
    processing_class = DummyProcessing(tokenizer)

    # Define PPOConfig
    ppo_config = PPOConfig(
        learning_rate=1.4e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        seed=42,
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        total_episodes=100,
        max_grad_norm=1.0,
        kl_coef=0.05,
        whiten_rewards=False
    )

    # --- Load data ---
    print("Loading data...")
    df = pd.read_csv("data/merged_queries_ads.csv")
    print(f"Loaded {len(df)} rows\n")

    # Format dataset for PPO trainer
    dataset = [{
        "query": row["vague_query"],
        "ad_facts": {
            "ad_product": row["ad_product"],
            "brand": row["brand"],
            "url": row["url"],
            "description": row["ad_description"]
        }
    } for _, row in df.iterrows()]

    # Inspect the model object attributes before trainer init
    print("--- Inspecting model attributes ---")
    print(dir(model))
    print("---------------------------------")

    # Create PPOTrainer
    ppo_trainer = PPOTrainer(
        ppo_config,
        model=model, # Use the value-head model passed in
        ref_model=None, # Let TRL handle reference model creation
        value_model=model, # Explicitly pass the value head model here too
        reward_model=reward_model,
        processing_class=processing_class,
        train_dataset=dataset
    )

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
    print("\nStarting PPO training...")
    for idx, data in enumerate(tqdm(dataset, desc="PPO Fine-tuning")):
        try:
            query = data["query"]
            ad_facts = data["ad_facts"]

            # Get device from model parameters
            device = next(model.parameters()).device

            query_tensor = tokenizer(query, return_tensors="pt").input_ids.to(device)
            response_wo_ad, response_with_ad = generate_responses(query, ad_facts, model, tokenizer)

            reward_details = compute_reward(query, response_with_ad, response_wo_ad, ad_facts)
            reward = torch.tensor([reward_details.get("Total Score", 0)]).to(device)

            # Print reward details for the first iteration
            if idx == 0:
                print("--- Reward Details (First Iteration) ---")
                print(json.dumps(reward_details, indent=2))
                print("--------------------------------------")

            ppo_trainer.step(
                [query_tensor.squeeze(0)],
                [tokenizer.encode(response_with_ad, return_tensors="pt").squeeze(0).to(device)],
                [reward]
            )

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx, query, response_wo_ad, response_with_ad,
                    reward_details.get("C1", ""), reward_details.get("C2", ""), reward_details.get("C3", ""), reward_details.get("C4", ""),
                    reward_details.get("Coherence Explanation", ""), reward_details.get("Coherence Score", 0),
                    reward_details.get("H1", ""), reward_details.get("Helpfulness Explanation", ""), reward_details.get("Helpfulness Score", 0),
                    reward_details.get("S1", ""), reward_details.get("S2", ""), reward_details.get("S3", ""), reward_details.get("Ad Salience Explanation", ""), reward_details.get("Ad Salience Score", 0),
                    reward_details.get("detectability_cosine", ""), reward_details.get("similarity_cosine", ""),
                    reward_details.get("detectability_bert", ""), reward_details.get("bert_f1", ""),
                    reward.item()
                ])

            if idx % 20 == 0:
                sample_idx = max(0, idx - 10)
                past_query = dataset[sample_idx]["query"]
                past_ad_facts = dataset[sample_idx]["ad_facts"]
                _, resampled_response = generate_responses(past_query, past_ad_facts, model, tokenizer)
                resampled_reward = compute_reward(past_query, resampled_response, "", past_ad_facts).get("Total Score", 0)

                with open(resample_log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([idx, sample_idx, past_query, resampled_response, resampled_reward])

            print(f"[{idx}] Reward: {reward.item():.2f} | Query: {query[:40]}")

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            if idx % 50 == 0 or idx == len(dataset) - 1:
                output_dir = "checkpoints/ppo_finetuned_deepseek"
                os.makedirs(output_dir, exist_ok=True)
                print(f"\nSaving PPO fine-tuned model to {output_dir}...")
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print("PPO model and tokenizer saved.")

        except Exception as e:
            print(f"Error on row {idx}: {e}")
            continue

    summary_file = "logs/ppo_reward_summary.csv"
    rewards_df = pd.read_csv(log_file)
    if not rewards_df.empty:
        rewards_df["Step"] = pd.to_numeric(rewards_df["Step"], errors='coerce')
        rewards_df["Total Reward"] = pd.to_numeric(rewards_df["Total Reward"], errors='coerce')
        rewards_df.dropna(subset=["Total Reward"], inplace=True)
        rewards_df.groupby(rewards_df["Step"] // 10 * 10)["Total Reward"].mean().to_csv(summary_file)
        print(f"\n📈 PPO reward summary saved to {summary_file}")

    print("\nPPO fine-tuning complete.")
