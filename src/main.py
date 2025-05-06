import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from training.evaluation import evaluate_logged_responses
from training.manual_ppo_loop import run_manual_ppo

CHECKPOINT_DIR = "checkpoints/ppo_manual"

def load_model_and_tokenizer():
    model_path = CHECKPOINT_DIR
    # Check if a local checkpoint exists and seems valid (e.g., contains config.json)
    if os.path.exists(os.path.join(model_path, "config.json")):
        print(f"✅ Found local checkpoint. Loading model from: {model_path}")
    else:
        print(f"ℹ️ No local checkpoint found at {model_path}. Loading base model...")
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # Original base model

    print(f"Loading tokenizer from: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from: {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32, # Use FP32 for stability
        device_map="auto",
        trust_remote_code=True
    )
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception: # Handle cases where generation_config might be missing from older checkpoints
         print("⚠️ Could not load generation_config from checkpoint. Using default.")
         # Use generation_config from the original base model name as a fallback
         base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
         model.generation_config = GenerationConfig.from_pretrained(base_model_name)

    print("✅ Models loaded\n")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    run_manual_ppo(model, tokenizer)
    # Optionally evaluate logged responses (Now handled periodically within the loop)
    # try:
    #     evaluate_logged_responses("logs/ppo_manual_log.csv") 
    # except FileNotFoundError:
    #     print("Log file not found, skipping evaluation.")  