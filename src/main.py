import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import gc

from src.training.manual_ppo_loop import run_manual_ppo
from src.config import BASE_MODEL, CHECKPOINT_DIR, DATA_FILE, HF_TOKEN_ENV_VAR, GOOGLE_API_KEY_ENV_VAR, PROJECT_ID_ENV_VAR, CUSTOM_JUDGE_MODEL_ID_ENV_VAR
from src.judge.utils import clear_caches 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load environment variables from .env file
load_dotenv()

def load_model_and_tokenizer():
    model_path = CHECKPOINT_DIR
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print("⚠️ Hugging Face token not found in .env file. Will attempt to load model without it.")
        print("   This may fail for gated models. Please add HF_TOKEN to your .env file.")

    # Check if a local checkpoint exists by looking for checkpoint_info.json
    checkpoint_info_path = os.path.join(model_path, "checkpoint_info.json")
    if os.path.exists(checkpoint_info_path):
        print(f"✅ Found local checkpoint info. Loading model from: {model_path}")
        effective_model_path_for_tokenizer = BASE_MODEL
        effective_model_path_for_model = BASE_MODEL # run_manual_ppo will override this if checkpoint exists
    else:
        print(f"ℹ️ No local checkpoint found at {model_path}. Loading base model...")
        effective_model_path_for_tokenizer = BASE_MODEL
        effective_model_path_for_model = BASE_MODEL

    print(f"Loading tokenizer from: {effective_model_path_for_tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(effective_model_path_for_tokenizer, trust_remote_code=True, token=hf_token)

    print(f"Loading model from: {effective_model_path_for_model}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            effective_model_path_for_model,
            torch_dtype=torch.float16,  # Use FP16 for efficiency
            device_map="auto",
            trust_remote_code=True,
            use_cache=True,  # Enable KV cache for faster inference
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
            token=hf_token
        )
        try:
            model.generation_config = GenerationConfig.from_pretrained(effective_model_path_for_model, token=hf_token)
        except Exception: # Handle cases where generation_config might be missing
            print(f"⚠️ Could not load generation_config from {effective_model_path_for_model}. Using default from {BASE_MODEL}.")
            model.generation_config = GenerationConfig.from_pretrained(BASE_MODEL, token=hf_token)
    except Exception as e:
        print(f"❌ Failed to load base model {effective_model_path_for_model} in main.py: {e}")
        print("   This might be a placeholder if checkpoints are used. run_manual_ppo will attempt the actual load.")
        model = None # So run_manual_ppo knows to load it.

    print("✅ Model and tokenizer loading attempted in main.py.\n")
    return model, tokenizer, hf_token

if __name__ == "__main__":
    # Clear memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    model, tokenizer, hf_token = load_model_and_tokenizer()
    # Let run_manual_ppo handle checkpoint management and definitive model loading
    run_manual_ppo(model, tokenizer, base_model_name=BASE_MODEL, checkpoint_dir_str=CHECKPOINT_DIR, hf_token=hf_token) # Pass CHECKPOINT_DIR
    