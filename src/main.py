import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from training.evaluation import evaluate_logged_responses
from training.manual_ppo_loop import run_manual_ppo
import gc

CHECKPOINT_DIR = "checkpoints/ppo_manual"
BASE_MODEL = "meta-llama/Llama-3.1-8B"  # Changed to Llama 3.1-8B

def load_model_and_tokenizer():
    model_path = CHECKPOINT_DIR
    # Check if a local checkpoint exists by looking for checkpoint_info.json
    checkpoint_info_path = os.path.join(model_path, "checkpoint_info.json")
    if os.path.exists(checkpoint_info_path):
        print(f"✅ Found local checkpoint info. Loading model from: {model_path}")
        # We'll let run_manual_ppo handle the actual loading of the latest checkpoint
        model_path = BASE_MODEL  # Use Llama 3.1-8B as base model
    else:
        print(f"ℹ️ No local checkpoint found at {model_path}. Loading base model...")
        model_path = BASE_MODEL  # Use Llama 3.1-8B as base model

    print(f"Loading tokenizer from: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from: {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use FP16 for efficiency
        device_map="auto",
        trust_remote_code=True,
        use_cache=True,  # Enable KV cache for faster inference
        low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
    )
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception: # Handle cases where generation_config might be missing from older checkpoints
         print("⚠️ Could not load generation_config from checkpoint. Using default.")
         # Use generation_config from the base model as fallback
         model.generation_config = GenerationConfig.from_pretrained(BASE_MODEL)

    print("✅ Models loaded\n")
    return model, tokenizer

if __name__ == "__main__":
    # Clear memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    model, tokenizer = load_model_and_tokenizer()
    # Let run_manual_ppo handle checkpoint management
    run_manual_ppo(model, tokenizer)
    