# from generate.generator import generate_responses
# from reward import compute_reward  
# import pandas as pd
# import torch
# import gc
# import os
# from tqdm import tqdm
# import time

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from trl import PPOTrainer, PPOConfig

# # --- Load model and tokenizer ---
# print("Loading DeepSeek model...")
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto",
#     trust_remote_code=True
# )
# print("Model loaded\n")

# # --- PPO config & trainer ---
# ppo_config = PPOConfig(
#     model_name=MODEL_NAME,
#     batch_size=1,
#     mini_batch_size=1,
#     learning_rate=1.4e-5,
#     optimize_cuda_cache=True
# )
# ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

# # --- Load data ---
# print("Loading data...")
# df = pd.read_csv("data/merged_queries_ads.csv")
# print(f"Loaded {len(df)} rows\n")

# # --- PPO loop ---
# for idx, row in tqdm(df.iterrows(), total=len(df), desc="PPO Fine-tuning"):
#     try:
#         user_query = row["vague_query"]
#         ad_facts = {
#             "ad_product": row["ad_product"],
#             "brand": row["brand"],
#             "url": row["url"],
#             "description": row["ad_description"]
#         }

#         # Tokenize query (for PPO step)
#         query_tensor = tokenizer(user_query, return_tensors="pt").input_ids.to(model.device)

#         # Generate responses
#         response_without_ad, response_with_ad = generate_responses(user_query, ad_facts, model, tokenizer)

#         # Compute reward using judges
#         reward_score = compute_reward(user_query, response_with_ad, response_without_ad, ad_facts)
#         reward_tensor = torch.tensor([reward_score]).to(model.device)

#         # PPO step: fine-tune on one query-response pair
#         ppo_trainer.step(
#             [query_tensor.squeeze(0)],
#             [tokenizer.encode(response_with_ad, return_tensors="pt").squeeze(0).to(model.device)],
#             [reward_tensor]
#         )

#         print(f"[{idx}] Reward: {reward_score:.2f} | Query: {user_query[:40]}")

#         # Optional cleanup
#         gc.collect()
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()

#     except Exception as e:
#         print(f"Error on row {idx}: {e}")
#         continue

# print("\n🏁 PPO fine-tuning complete.")


from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation import evaluate_logged_responses
from training.ppo_loop import run_ppo_finetuning

def load_model_and_tokenizer():
    print("Loading DeepSeek model...")
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ Model loaded\n")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    run_ppo_finetuning(model, tokenizer)
    evaluate_logged_responses("logs/full_responses_log.csv")