import os
import gc
import torch
import csv
import pandas as pd
from tqdm import tqdm
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from copy import deepcopy
# Remove unused imports if generate_responses and compute_reward are no longer called directly
# from generate.generator import generate_responses
# from training.reward import compute_reward
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
        batch_size=1, # Keep batch size 1 if processing one query at a time
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        seed=42,
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        # total_episodes=100, # TRL might use num_train_epochs or max_steps instead
        # Let TRL handle training duration based on dataset size and epochs/steps
        max_grad_norm=1.0,
        kl_coef=0.05,
        whiten_rewards=False,
        gradient_checkpointing=True,
        # Add logging directory for TRL internal logging
        log_with="tensorboard", # Or "wandb" if you prefer
        logging_dir="logs/ppo_trainer_logs",
        ppo_epochs=4 # Number of PPO epochs per batch
    )

    # --- Load data ---
    print("Loading data...")
    df = pd.read_csv("data/merged_queries_ads.csv")
    print(f"Loaded {len(df)} rows\n")

    # Format dataset for PPO trainer - Ensure it's compatible with TRL's train method
    # TRL typically expects a Hugging Face Dataset object or similar structure.
    # Let's try keeping the list of dicts for now, but might need adjustment.
    # We need 'query' and potentially the inputs tokenized.
    # TRL's train usually expects tokenized inputs in the dataset.
    # Let's tokenize the queries here.
    def tokenize_query(example):
        example["input_ids"] = tokenizer(example["query"], return_tensors="pt").input_ids.squeeze(0)
        return example

    dataset = [{
        "query": row["vague_query"],
        # Add ad_facts if needed by custom generation/reward logic *within* TRL (unlikely with dummy)
        # "ad_facts": { ... }
    } for _, row in df.iterrows()]
    dataset = [tokenize_query(d) for d in dataset]

    # Create PPOTrainer
    ppo_trainer = PPOTrainer(
        ppo_config,
        model=model, # Use the value-head model passed in
        ref_model=None, # Let TRL handle reference model creation
        # value_model=model, # Remove this explicit passing
        tokenizer=tokenizer, # Pass tokenizer needed for generation
        reward_model=reward_model,
        processing_class=processing_class,
        dataset=dataset # Pass the formatted dataset
    )

    # --- Run PPO training using TRL's train method ---
    print("Starting PPO training using ppo_trainer.train()...")
    # The train method handles the loop, generation, reward, stepping, logging
    ppo_trainer.train()
    print("PPO training complete.")

    # --- Save final model ---
    output_dir = "checkpoints/ppo_finetuned_deepseek"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving final PPO fine-tuned model to {output_dir}...")
    # Use the save_pretrained from the trainer which saves adapter/model correctly
    ppo_trainer.save_model(output_dir)
    # model.save_pretrained(output_dir) # Use trainer's save method
    tokenizer.save_pretrained(output_dir)
    print("PPO model and tokenizer saved.")

    # Remove manual logging setup and loop
    # --- Set up logging ---
    # os.makedirs("logs", exist_ok=True)
    # log_file = "logs/ppo_training_log.csv"
    # ... (removed log file setup) ...
    # resample_log_file = "logs/resampled_rollouts.csv"
    # ... (removed resample log file setup) ...

    # --- PPO loop ---
    # print("Starting PPO training...")
    # for idx, data in enumerate(tqdm(dataset, desc="PPO Fine-tuning")):
    #     try:
    #         query = data["query"]
    #         ad_facts = data["ad_facts"]
    #
    #         # Get device from model parameters
    #         device = next(model.parameters()).device
    #
    #         query_tensor = tokenizer(query, return_tensors="pt").input_ids.to(device)
    #         # Manual generation removed
    #         # response_wo_ad, response_with_ad = generate_responses(query, ad_facts, model, tokenizer)
    #
    #         # Manual reward calculation removed
    #         # reward_details = compute_reward(query, response_with_ad, response_wo_ad, ad_facts)
    #         # reward = torch.tensor([reward_details.get("Total Score", 0)]).to(device)
    #
    #         # Print reward details removed
    #         # if idx == 0: ...
    #
    #         # Manual step removed
    #         # ppo_trainer.step(...)
    #
    #         # Manual logging removed
    #         # with open(log_file, "a", newline="") as f: ...
    #
    #         # Manual resampling removed
    #         # if idx % 20 == 0: ...
    #
    #         # Print statement removed
    #         # print(f"[{idx}] Reward: {reward.item():.2f} | Query: {query[:40]}")
    #
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()
    #
    #         # Manual saving removed
    #         # if idx % 50 == 0 or idx == len(dataset) - 1: ...
    #
    #     except Exception as e:
    #         print(f"Error on row {idx}: {e}")
    #         continue

    # Remove manual summary creation - TRL logs metrics automatically
    # summary_file = "logs/ppo_reward_summary.csv"
    # ... (removed summary logic) ...

    # print("PPO fine-tuning complete.") # Moved earlier

    # Note: evaluate_logged_responses in main.py will need adjustment
    # as the log file format/location will change based on TRL's logging.
    # It might need to read from tensorboard logs or be removed if TRL provides sufficient eval.
